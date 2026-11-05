// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <fmt/format.h>
#include "common/common_types.h"

namespace VideoCommon {
// EDEN_NATIVE_BUFFER_TRACE=<GPU address>:<length>, both hexadecimal. Captures GPU-facing
// host uploads and DMA requests only; direct guest CPU stores and compute writes are not covered.
inline void TraceNativeBufferWrite(const char* tag, u64 destination, const void* data,
                                   size_t size, u64 source_gpu = 0) {
    struct State {
        State() {
            const char* value = std::getenv("EDEN_NATIVE_BUFFER_TRACE");
            if (!value) return;
            char* tail{};
            const u64 address = std::strtoull(value, &tail, 16);
            if (!tail || *tail != ':') return;
            const u64 length = std::strtoull(tail + 1, &tail, 16);
            if (*tail || !address || !length || length > 64 * 1024 * 1024 ||
                address > std::numeric_limits<u64>::max() - length) return;
            begin = address;
            end = address + length;
        }
        u64 begin{}, end{};
        u64 payload_total{};
        u32 records{};
        std::mutex mutex;
    };
    static State state;
    if (!state.end || destination >= state.end ||
        size > std::numeric_limits<u64>::max() - destination ||
        destination + size <= state.begin) return;
    const std::scoped_lock lock{state.mutex};
    if (state.records >= 32768) {
        if (state.records == 32768) {
            ++state.records;
            std::ofstream("native-buffer-write-limit.txt") << "Trace limit reached; subsequent writes were omitted.\n";
        }
        return;
    }
    const auto prefix = fmt::format("native-buffer-write-{:04}", state.records++);
    const u64 begin = (std::max)(destination, state.begin);
    const u64 end = (std::min)(destination + size, state.end);
    const u64 payload_bytes = data ? (std::min)({end - begin, u64{1024 * 1024},
                                               u64{128 * 1024 * 1024} - state.payload_total}) : 0;
    state.payload_total += payload_bytes;
    if (payload_bytes) {
        std::ofstream payload(prefix + ".bin", std::ios::binary);
        payload.write(static_cast<const char*>(data) + (begin - destination), payload_bytes);
    }
    std::ofstream metadata(prefix + ".txt");
    metadata << "tag " << tag << "\ndestination " << std::hex << destination
             << "\nsource_gpu " << source_gpu << "\ncaptured_gpu " << begin
             << "\nwatch_begin " << state.begin << "\nwatch_end " << state.end
             << std::dec << "\nrequested_bytes " << size << "\ncaptured_bytes " << end - begin
             << "\npayload " << (data != nullptr) << "\npayload_bytes " << payload_bytes << '\n';
}
} // namespace VideoCommon
