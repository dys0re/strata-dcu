#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "strata/kernels/execution_weight.hpp"

namespace strata::core {

/// Opt-in native output projection. Owns one device execution layout and its scratch.
/// GGUF quantization and device layout are separate; execution() describes both.
/// Plan v0.3 P6: any type the native MMVQ takes (Q5_K in the Q2_0 / IQ3_XXS files, IQ4_XS in IQ2_XS).
/// `load` takes the model's shards (strata::gguf_split_paths): the architecture is checked on the first, the
/// tensor is read from the shard that holds it.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    bool load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err);
    void unload();
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool loaded() const { return weights_ != nullptr; }
    // Raw access is GGUF-only (e.g. the BF16 DFlash head). General consumers use execution().
    const void* weights() const { return execution_.gguf() ? weights_ : nullptr; }
    const void* storage() const { return weights_; }
    strata::kernels::ExecutionWeight execution() const {
        return execution_.data ? execution_ : strata::kernels::ExecutionWeight{weights_, type_, n_in_, n_out_};
    }
    int type() const { return type_; }
    /// The device whose memory holds the weights (the one current when `load` ran; -1 before).
    int device() const { return device_; }
    /// Bytes of one reconstructed GGUF vocabulary row (independent of execution padding).
    size_t row_bytes() const { return row_bytes_ ? row_bytes_ : n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }

private:
    strata::kernels::ExecutionWeight execution_;
    size_t row_bytes_ = 0;
    void* weights_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int n_in_ = 0, n_out_ = 0;
    int type_ = -1;
    int device_ = -1;
};

/// Plan v0.3 P6: `token_embd.weight` in its GGUF form (the IQ model files), in mapped pinned host memory: a row
/// is read over PCIe per token, so the table costs no VRAM.  `embed_row`, the verify window and the MTP drafter
/// use it instead of the canonical table when it is set (`set_native_embed`).
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    bool load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err);
    /// Rows for device token ids.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }

private:
    void* host_ = nullptr;
    const void* dev_ = nullptr;
    uint64_t bytes_ = 0;
    size_t row_ = 0;
    int64_t n_embd_ = 0, n_vocab_ = 0;
    int type_ = -1;
};
void set_native_embed(const NativeEmbed* e);
const NativeEmbed* native_embed();

}  // namespace strata::core
