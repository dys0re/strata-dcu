// Exercise disk-backed external BF16 PLE at unaligned offsets, EOF and cache
// reuse; reject changed source sizes and out-of-bounds descriptors.
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
static std::string le(uint64_t v, int n) {
    std::string s(n, '\0');
    for (int i = 0; i < n; ++i) s[i] = char(v >> (8 * i));
    return s;
}
static std::string str(const std::string& s) { return le(s.size(), 8) + s; }
static void descriptor(const fs::path& p, uint64_t offset, uint64_t rows, uint64_t size) {
    std::string h = "GGUF" + le(3, 4) + le(0, 8) + le(5, 8);
    h += str("general.architecture") + le(8, 4) + str("strata-ple");
    h += str("strata.ple.external.path") + le(8, 4) + str("source.safetensors");
    h += str("strata.ple.external.offset") + le(10, 4) + le(offset, 8);
    h += str("strata.ple.external.rows") + le(10, 4) + le(rows, 8);
    h += str("strata.ple.external.file_size") + le(10, 4) + le(size, 8);
    std::ofstream(p, std::ios::binary).write(h.data(), h.size());
}
int main() {
    using namespace strata::kernels;
    const auto dir = fs::temp_directory_path() /
        ("strata-ple-external-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(dir);
        constexpr uint64_t offset = 20736, rows = 29;
        std::vector<uint16_t> weights(rows * 160);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = bf16_from_f32(float(int(i % 101) - 50) / 16);
        {
            std::ofstream f(dir / "source.safetensors", std::ios::binary);
            const std::string prefix(offset, 'h');
            f.write(prefix.data(), prefix.size());
            f.write(reinterpret_cast<const char*>(weights.data()), weights.size() * 2);
        }
        const uint64_t size = offset + weights.size() * 2;
        descriptor(dir / "ple.gguf", offset, rows, size);
        std::string err;
        PleIoOptions io;
        io.cache_rows = 32;
        {
            PleTable table;
            if (!table.open((dir / "ple.gguf").string(), err, io)) throw std::runtime_error(err);
            uint32_t ids[16];
            float values[2560];
            for (int repeat = 0; repeat < 2; ++repeat) {
                for (int i = 0; i < 16; ++i) ids[i] = (i * 13) % rows;
                ids[0] = 0; ids[1] = rows - 1;
                if (!table.issue(ids) || !table.collect(values, err)) throw std::runtime_error(err);
                for (int i = 0; i < 16; ++i) for (int j = 0; j < 160; ++j)
                    if (values[i * 160 + j] != f32_from_bf16(weights[ids[i] * 160 + j]))
                        throw std::runtime_error("external PLE read differs from original BF16 bits");
            }
        }
        {
            PleTable t; io.mode = PleIo::Mmap;
            if (t.open((dir / "ple.gguf").string(), err, io)) throw std::runtime_error("mmap unexpectedly accepted");
            io.mode = PleIo::Direct;
            descriptor(dir / "ple.gguf", offset, rows, size + 1);
            if (t.open((dir / "ple.gguf").string(), err, io)) throw std::runtime_error("changed file accepted");
            descriptor(dir / "ple.gguf", offset + 320, rows, size);
            if (t.open((dir / "ple.gguf").string(), err, io)) throw std::runtime_error("out of bounds accepted");
        }
        fs::remove_all(dir);
        std::puts("ple_external_test OK: exact disk reads, EOF, relative path, cache, size/bounds guards");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        fs::remove_all(dir);
        return 1;
    }
}
