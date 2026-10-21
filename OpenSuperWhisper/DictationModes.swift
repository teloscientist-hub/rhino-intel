import AppKit
import ApplicationServices
import Foundation
import KeyboardShortcuts

/// What a recording is for. The hotkey decides, so the app never guesses intent from the words.
///
/// - dictate: today's behavior — type what was said (after the usual cleanup).
/// - prompt: turn a rough spoken idea into a structured prompt for an AI assistant.
/// - rewrite: rewrite the text that was selected when the key went down, following the spoken
///   instruction ("make this friendlier"); the result replaces the selection.
enum DictationMode: String {
    case dictate, prompt, rewrite
}

extension KeyboardShortcuts.Name {
    /// Hold to speak an idea; Rhino types a structured prompt.
    static let promptMode = Self("promptMode", default: .init(.p, modifiers: [.control, .option]))
    /// Select text, hold, say how to change it; Rhino replaces the selection.
    static let rewriteMode = Self("rewriteMode", default: .init(.r, modifiers: [.control, .option]))
}

enum DictationModes {
    /// The larger local model, when it's on disk. The 1.5B cleanup model wrote flat, unstructured
    /// prompts and reply drafts that contradicted the dictated gist (i3 bench, Xeon W-3245,
    /// 2026-10-21); the 3B wrote usable ones at ~30 tok/s on 16 threads. Falls back to the
    /// cleanup model if the 3B file is missing.
    static let preferredModelFile = "qwen2.5-3b-instruct-q4_k_m.gguf"

    static func systemPrompt(for mode: DictationMode) -> String? {
        switch mode {
        case .dictate:
            return nil
        case .prompt:
            return """
            You turn a person's spoken, rambling idea into a clear prompt for an AI assistant. \
            Write it in the second person, addressed to the assistant, as four labeled sections, \
            each on its own line: Goal, Context, Constraints, Expected output. One or two \
            sentences each. Keep every requirement the speaker stated; add none of your own. \
            Start your answer with "Goal:" and output nothing but the four sections.
            """
        case .rewrite:
            return """
            You rewrite a passage of text following the user's spoken instruction. Change only \
            what the instruction asks for and keep everything else, including the author's voice. \
            Output only the rewritten passage, with no preamble or quotation marks.
            """
        }
    }

    static func userMessage(for mode: DictationMode, spoken: String, selected: String?) -> String {
        switch mode {
        case .dictate, .prompt:
            return spoken
        case .rewrite:
            return "PASSAGE:\n\(selected ?? "")\n\nINSTRUCTION:\n\(spoken)"
        }
    }

    /// Runs the mode over the transcript. Never throws and never loses the dictation: on any
    /// failure (no model, empty output, rewrite with nothing selected) it returns `spoken`.
    static func apply(_ mode: DictationMode, spoken: String, selected: String?) async -> String {
        guard let system = systemPrompt(for: mode) else { return spoken }
        if mode == .rewrite, (selected ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            print("DictationModes: rewrite with no selection; inserting the dictation as is")
            return spoken
        }
        let user = userMessage(for: mode, spoken: spoken, selected: selected)
        let start = CFAbsoluteTimeGetCurrent()
        let output = await ModeLlama.shared.generate(system: system, user: user)
        print(String(format: "DictationModes: %@ in %.0fms", mode.rawValue,
                     (CFAbsoluteTimeGetCurrent() - start) * 1000))
        var trimmed = output?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        // Small models sometimes open with "Here is the prompt:" chatter despite the instruction.
        if mode == .prompt, let goal = trimmed.range(of: "Goal:"), goal.lowerBound != trimmed.startIndex {
            trimmed = String(trimmed[goal.lowerBound...])
        }
        return trimmed.isEmpty ? spoken : trimmed
    }

    /// The text selected in the focused element of the frontmost app, via Accessibility.
    /// nil when nothing is selected or the app doesn't expose it (many web views don't).
    static func selectedTextInFrontmostApp() -> String? {
        guard let pid = NSWorkspace.shared.frontmostApplication?.processIdentifier else { return nil }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.25)
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(app, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let focused else { return nil }
        let element = focused as! AXUIElement
        var selected: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextAttribute as CFString, &selected) == .success,
              let text = selected as? String, !text.isEmpty else { return nil }
        return text
    }
}

/// A llama context for modes, separate from the cleanup backend so the larger model and its
/// longer outputs never disturb the cleanup path's prompt cache. Loaded lazily, released after
/// idleUnloadDelay without use. All context access happens on `queue` (llama_context is not
/// thread-safe).
final class ModeLlama {
    static let shared = ModeLlama()

    private static let idleUnloadDelay: TimeInterval = 5 * 60
    private let queue = DispatchQueue(label: "com.noahkagan.rhino.mode-llm", qos: .userInitiated)
    private var context: LlamaContext?
    private var unloadWork: DispatchWorkItem?

    private init() {}

    private func modelPath() -> String? {
        let dir = LLMModelManager.shared.modelsDirectory
        let preferred = dir.appendingPathComponent(DictationModes.preferredModelFile).path
        if FileManager.default.fileExists(atPath: preferred) { return preferred }
        let fallback = dir.appendingPathComponent("qwen2.5-1.5b-instruct-q4_k_m.gguf").path
        return FileManager.default.fileExists(atPath: fallback) ? fallback : nil
    }

    /// Starts loading the model while the user is still speaking.
    func preload() {
        queue.async { [weak self] in _ = self?.loadOnQueue() }
    }

    func generate(system: String, user: String, maxTokens: Int = 300) async -> String? {
        await withCheckedContinuation { continuation in
            queue.async { [weak self] in
                guard let self, let ctx = self.loadOnQueue() else {
                    continuation.resume(returning: nil)
                    return
                }
                let out = ctx.generate(system: system, user: user, maxTokens: maxTokens)
                self.scheduleUnloadOnQueue()
                continuation.resume(returning: out)
            }
        }
    }

    private func loadOnQueue() -> LlamaContext? {
        unloadWork?.cancel()
        if let context { return context }
        guard let path = modelPath() else {
            print("ModeLlama: no model on disk")
            return nil
        }
        context = LlamaContext(modelPath: path)
        print("ModeLlama: loaded \((path as NSString).lastPathComponent): \(context != nil)")
        return context
    }

    private func scheduleUnloadOnQueue() {
        unloadWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.context = nil }
        unloadWork = work
        queue.asyncAfter(deadline: .now() + Self.idleUnloadDelay, execute: work)
    }
}
