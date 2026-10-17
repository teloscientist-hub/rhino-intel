//
// Created by user on 07.02.2025.
//

import Foundation

public struct WhisperContextParams {
    // Intel Macs: ggml's Metal backend on discrete AMD GPUs (no simdgroup ops)
    // returns garbled transcripts and is slower than the CPU. Use the CPU there.
    #if arch(x86_64)
    public var useGPU: Bool = false
    #else
    public var useGPU: Bool = true
    #endif
    public var flashAttention: Bool = true
    public var gpuDevice: Int32 = 0
    public var dtwTokenTimestamps: Bool = false
    public var dtwAheadsPreset: WhisperAlignmentHeadsPreset = .none
    public var dtwNTop: Int32 = 0
    public var dtwAheads: WhisperAheads = .init(heads: [])
    public var dtwMemSize: Int = 0 // remove

    public init() {}

    func toC() -> whisper_context_params {
        var cParams = whisper_context_params()
        cParams.use_gpu = useGPU
        cParams.flash_attn = flashAttention
        cParams.gpu_device = gpuDevice
        cParams.dtw_token_timestamps = dtwTokenTimestamps
        cParams.dtw_aheads_preset = whisper_alignment_heads_preset(rawValue: UInt32(dtwAheadsPreset.rawValue))
        cParams.dtw_n_top = dtwNTop
        let swiftAheads = dtwAheads
        cParams.dtw_aheads = swiftAheads.toC() // Use the toC() method
        cParams.dtw_mem_size = dtwMemSize
        return cParams
    }
}