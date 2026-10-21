import Foundation

/// Parakeet on Intel Macs, through sherpa-onnx (ONNX Runtime on the CPU).
///
/// FluidAudio runs Parakeet as CoreML models built for the Neural Engine. Intel Macs have no
/// Neural Engine, and CoreML's CPU fallback crashes (SIGFPE in
/// Espresso::general_padding_kernel_cpu, Xeon W-3245, macOS 26.7) whichever compute units are
/// requested. sherpa-onnx runs the same NVIDIA model exported to ONNX and transcribes the JFK
/// sample correctly in ~1s on that machine.
///
/// The library is loaded at runtime (dlopen) from the app's Frameworks folder, so the Xcode
/// project needs no link settings and Apple-silicon builds never touch it.
/// `Scripts/vendor-sherpa.sh` puts the dylibs there.
final class SherpaParakeetEngine: TranscriptionEngine {
    var engineName: String { "Parakeet (sherpa-onnx)" }
    var onProgressUpdate: ((Float) -> Void)?

    private var recognizer: OpaquePointer?
    private var isCancelled = false
    private let version: String

    init(versionOverride: String? = nil) {
        version = versionOverride ?? AppPreferences.shared.fluidAudioModelVersion
    }

    var isModelLoaded: Bool { recognizer != nil }

    func initialize() async throws {
        let api = try SherpaAPI.shared()
        guard let dir = SherpaParakeetModels.directoryIfPresent(version: version) else {
            throw TranscriptionError.contextInitializationFailed
        }
        let files = SherpaParakeetModels.files(in: dir)
        var cStrings: [UnsafeMutablePointer<CChar>] = []
        func c(_ s: String) -> UnsafePointer<CChar> {
            let p = strdup(s)!
            cStrings.append(p)
            return UnsafePointer(p)
        }
        defer { cStrings.forEach { free($0) } }

        var config = SherpaOnnxOfflineRecognizerConfig()
        config.feat_config.sample_rate = 16000
        config.feat_config.feature_dim = 80
        config.model_config.transducer.encoder = c(files.encoder)
        config.model_config.transducer.decoder = c(files.decoder)
        config.model_config.transducer.joiner = c(files.joiner)
        config.model_config.tokens = c(files.tokens)
        config.model_config.num_threads = Int32(max(2, min(ProcessInfo.processInfo.activeProcessorCount, 8)))
        config.model_config.provider = c("cpu")
        config.model_config.model_type = c("nemo_transducer")
        config.decoding_method = c("greedy_search")

        recognizer = withUnsafePointer(to: &config) { api.createRecognizer($0) }
        guard recognizer != nil else { throw TranscriptionError.contextInitializationFailed }

        // First decode pays ONNX Runtime's one-time setup; do it here, not on the first dictation.
        _ = decode(api: api, samples: [Float](repeating: 0, count: 8000))
    }

    func transcribeAudio(url: URL, settings: Settings) async throws -> String {
        guard recognizer != nil else { throw TranscriptionError.contextInitializationFailed }
        let api = try SherpaAPI.shared()
        isCancelled = false
        onProgressUpdate?(0.05)

        let samples = try LocalAudioReader.samples(at: url)
        if WhisperEngine.isNearSilence(samples) {
            onProgressUpdate?(1.0)
            return TranscriptionResult.noSpeech
        }
        guard !isCancelled else { throw CancellationError() }

        let start = CFAbsoluteTimeGetCurrent()
        let raw = decode(api: api, samples: samples)
        print(String(format: "SherpaParakeetEngine: %.1fs audio in %.0fms",
                     Double(samples.count) / 16000, (CFAbsoluteTimeGetCurrent() - start) * 1000))
        guard !isCancelled else { throw CancellationError() }

        onProgressUpdate?(1.0)
        // Timestamps are not wired for this engine yet; plain text either way.
        let text = settings.applyCustomDictionary(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        return text.isEmpty ? TranscriptionResult.noSpeech : text
    }

    private func decode(api: SherpaAPI, samples: [Float]) -> String {
        guard let stream = api.createStream(recognizer) else { return "" }
        defer { api.destroyStream(stream) }
        samples.withUnsafeBufferPointer {
            api.acceptWaveform(stream, 16000, $0.baseAddress, Int32($0.count))
        }
        api.decode(recognizer, stream)
        guard let result = api.getResult(stream) else { return "" }
        defer { api.destroyResult(result) }
        guard let text = result.pointee.text else { return "" }
        return String(cString: text)
    }

    func cancelTranscription() { isCancelled = true }

    func getSupportedLanguages() -> [String] {
        EngineCapabilities.supportedLanguages(engine: "fluidaudio", fluidAudioModelVersion: version)
    }

    deinit {
        if let recognizer, let api = try? SherpaAPI.shared() { api.destroyRecognizer(recognizer) }
    }
}

/// The handful of sherpa-onnx C entry points the engine uses, resolved with dlsym.
final class SherpaAPI {
    typealias CreateRecognizer = @convention(c) (UnsafePointer<SherpaOnnxOfflineRecognizerConfig>?) -> OpaquePointer?
    typealias DestroyRecognizer = @convention(c) (OpaquePointer?) -> Void
    typealias CreateStream = @convention(c) (OpaquePointer?) -> OpaquePointer?
    typealias DestroyStream = @convention(c) (OpaquePointer?) -> Void
    typealias AcceptWaveform = @convention(c) (OpaquePointer?, Int32, UnsafePointer<Float>?, Int32) -> Void
    typealias Decode = @convention(c) (OpaquePointer?, OpaquePointer?) -> Void
    typealias GetResult = @convention(c) (OpaquePointer?) -> UnsafePointer<SherpaOnnxOfflineRecognizerResult>?
    typealias DestroyResult = @convention(c) (UnsafePointer<SherpaOnnxOfflineRecognizerResult>?) -> Void

    let createRecognizer: CreateRecognizer
    let destroyRecognizer: DestroyRecognizer
    let createStream: CreateStream
    let destroyStream: DestroyStream
    let acceptWaveform: AcceptWaveform
    let decode: Decode
    let getResult: GetResult
    let destroyResult: DestroyResult

    struct LoadError: LocalizedError {
        let errorDescription: String?
    }

    private static let lock = NSLock()
    private static var cached: Result<SherpaAPI, Error>?

    static func shared() throws -> SherpaAPI {
        lock.lock(); defer { lock.unlock() }
        if let cached { return try cached.get() }
        let result = Result { try SherpaAPI() }
        cached = result
        return try result.get()
    }

    /// Where the dylib is looked for: the app bundle first, then RHINO_SHERPA_LIB_DIR (tests, CLI).
    static var candidatePaths: [String] {
        var paths: [String] = []
        if let frameworks = Bundle.main.privateFrameworksPath {
            paths.append(frameworks + "/libsherpa-onnx-c-api.dylib")
        }
        if let dir = ProcessInfo.processInfo.environment["RHINO_SHERPA_LIB_DIR"] {
            paths.append(dir + "/libsherpa-onnx-c-api.dylib")
        }
        return paths
    }

    static var isAvailable: Bool {
        candidatePaths.contains { FileManager.default.fileExists(atPath: $0) }
    }

    private init() throws {
        guard let path = Self.candidatePaths.first(where: { FileManager.default.fileExists(atPath: $0) }),
              let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "libsherpa-onnx-c-api.dylib not found"
            throw LoadError(errorDescription: "sherpa-onnx unavailable: \(reason)")
        }
        func sym<T>(_ name: String, _: T.Type) throws -> T {
            guard let p = dlsym(handle, name) else {
                throw LoadError(errorDescription: "sherpa-onnx is missing \(name)")
            }
            return unsafeBitCast(p, to: T.self)
        }
        createRecognizer = try sym("SherpaOnnxCreateOfflineRecognizer", CreateRecognizer.self)
        destroyRecognizer = try sym("SherpaOnnxDestroyOfflineRecognizer", DestroyRecognizer.self)
        createStream = try sym("SherpaOnnxCreateOfflineStream", CreateStream.self)
        destroyStream = try sym("SherpaOnnxDestroyOfflineStream", DestroyStream.self)
        acceptWaveform = try sym("SherpaOnnxAcceptWaveformOffline", AcceptWaveform.self)
        decode = try sym("SherpaOnnxDecodeOfflineStream", Decode.self)
        getResult = try sym("SherpaOnnxGetOfflineStreamResult", GetResult.self)
        destroyResult = try sym("SherpaOnnxDestroyOfflineRecognizerResult", DestroyResult.self)
    }
}

/// Parakeet ONNX models for sherpa-onnx: where they live and how they are fetched.
enum SherpaParakeetModels {
    static func archiveName(version: String) -> String {
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-\(version == "v2" ? "v2" : "v3")-int8"
    }

    static func downloadURL(version: String) -> URL {
        URL(string: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/\(archiveName(version: version)).tar.bz2")!
    }

    static var baseDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "com.noahkagan.rhino")
            .appendingPathComponent("sherpa-models")
    }

    struct Files { let encoder, decoder, joiner, tokens: String }

    static func files(in dir: URL) -> Files {
        Files(encoder: dir.appendingPathComponent("encoder.int8.onnx").path,
              decoder: dir.appendingPathComponent("decoder.int8.onnx").path,
              joiner: dir.appendingPathComponent("joiner.int8.onnx").path,
              tokens: dir.appendingPathComponent("tokens.txt").path)
    }

    static func directoryIfPresent(version: String) -> URL? {
        let dir = baseDirectory.appendingPathComponent(archiveName(version: version))
        let f = files(in: dir)
        let all = [f.encoder, f.decoder, f.joiner, f.tokens]
        return all.allSatisfy { FileManager.default.fileExists(atPath: $0) } ? dir : nil
    }

    /// Downloads and unpacks the model archive (~480 MB). Progress covers the download only.
    static func download(version: String, progress: @escaping (Double) -> Void) async throws {
        if directoryIfPresent(version: version) != nil { return }
        try FileManager.default.createDirectory(at: baseDirectory, withIntermediateDirectories: true)

        let delegate = ProgressDelegate(progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let (tmp, response) = try await session.download(from: downloadURL(version: version))
        defer { try? FileManager.default.removeItem(at: tmp) }
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw TranscriptionError.contextInitializationFailed
        }

        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["xjf", tmp.path, "-C", baseDirectory.path]
        try tar.run()
        tar.waitUntilExit()
        guard tar.terminationStatus == 0, directoryIfPresent(version: version) != nil else {
            throw TranscriptionError.contextInitializationFailed
        }
    }

    private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate {
        let progress: (Double) -> Void
        init(progress: @escaping (Double) -> Void) { self.progress = progress }
        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                        totalBytesExpectedToWrite: Int64) {
            guard totalBytesExpectedToWrite > 0 else { return }
            progress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
        }
        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didFinishDownloadingTo location: URL) {}
    }
}
