// Exercise Strata's actual PLE loader and row gather against an independent float fixture.
#include "strata/kernels/ngram.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error(std::string("cannot open ") + path);
    const auto length = stream.tellg();
    if (length < 0 || length % sizeof(T)) throw std::runtime_error("invalid fixture length");
    std::vector<T> result(static_cast<size_t>(length) / sizeof(T));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(result.data()), length)) throw std::runtime_error("short fixture read");
    return result;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: check_ple_table table.gguf rows.u32 expected.f32\n");
        return 2;
    }
    try {
        const auto rows = read_file<uint32_t>(argv[2]);
        const auto expected = read_file<float>(argv[3]);
        if (rows.empty() || rows.size() % 16 || expected.size() != rows.size() * 160)
            throw std::runtime_error("fixture needs complete groups of 16 rows");
        for (const auto mode : {strata::kernels::PleIo::Direct, strata::kernels::PleIo::Mmap}) {
            strata::kernels::PleIoOptions io;
            io.mode = mode;
            io.cache_rows = 0;
            io.io_thread = false;
            io.max_inflight = 16;
            strata::kernels::PleTable table;
            std::string error;
            if (!table.open(argv[1], error, io)) throw std::runtime_error(error);
            if (std::strcmp(table.format(), "IQ4_NL") || table.rows() != 320001536)
                throw std::runtime_error("unexpected table format or geometry");
            std::vector<float> actual(16 * 160);
            for (size_t first = 0; first < rows.size(); first += 16) {
                if (!table.issue(rows.data() + first) || !table.collect(actual.data(), error))
                    throw std::runtime_error("gather failed: " + error);
                if (std::memcmp(actual.data(), expected.data() + first * 160, actual.size() * sizeof(float)))
                    throw std::runtime_error("row data differs from independent decoder at fixture row " + std::to_string(first));
            }
            std::printf("PLE_TABLE_PASS mode=%s rows=%zu format=%s total_rows=%llu\n",
                        mode == strata::kernels::PleIo::Direct ? "direct" : "mmap", rows.size(), table.format(),
                        static_cast<unsigned long long>(table.rows()));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_TABLE_FAIL: %s\n", error.what());
        return 1;
    }
}
