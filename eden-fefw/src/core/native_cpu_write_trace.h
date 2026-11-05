// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include "common/common_types.h"

namespace Core::NativeCpuWriteTrace {
inline bool Enabled() {
    static const bool enabled = std::getenv("EDEN_NATIVE_CPU_WRITE_TRACE") != nullptr;
    return enabled;
}

struct Range {
    u64 gpu{}, cpu{}, size{}, host{};
};
struct State {
    std::array<Range, 256> ranges{};
    std::atomic<u32> count{};
    std::atomic<u64> focus_gpu{};
    std::mutex mutex;
    u32 records{};
    u32 host_records{};
    std::ofstream writes;
};
inline State& GetState() {
    static State state;
    return state;
}

inline Range Find(u64 address, u64 size) {
    if (!Enabled()) return {};
    auto& state = GetState();
    const u32 count = state.count.load(std::memory_order_acquire);
    for (u32 i = count; i > 0; --i) {
        const auto range = state.ranges[i - 1];
        if (address < range.cpu + range.size && address + size > range.cpu) return range;
    }
    return {};
}

// This isolated trace follows the object-matrix page across the observed 2 MiB relocations.
// Initial snapshots also expose data written before the GPU mapping was established.
template <typename Memory>
void RegisterMapping(Memory& memory, u64 gpu, u64 cpu, u64 size) {
    if (!Enabled() || !cpu || !size || gpu + size < gpu || cpu + size < cpu) return;
    constexpr u64 window_begin = 0x19fe000000;
    constexpr u64 window_end = 0x1a00000000;
    const u64 begin = (std::max)(gpu, window_begin);
    const u64 end = (std::min)(gpu + size, window_end);
    if (begin >= end) return;
    auto& state = GetState();
    const std::scoped_lock lock{state.mutex};
    for (u64 block = begin & ~u64{0x1fffff}; block < end; block += 0x200000) {
        const u64 page = block + 0x1e2000;
        if (page < begin || page + 0x1000 > end) continue;
        const u64 address = cpu + page - gpu;
        if (!memory.IsValidVirtualAddressRange(address, 0x1000)) continue;
        const u32 index = state.count.load(std::memory_order_relaxed);
        if (index >= state.ranges.size()) {
            std::ofstream("native-cpu-map-limit.txt") << "Mapping trace capacity exceeded.\n";
            break;
        }
        state.ranges[index] = {page, address, 0x1000,
            reinterpret_cast<u64>(memory.template GetPointer<u8>(address))};
        state.count.store(index + 1, std::memory_order_release);
        const std::string prefix = "native-cpu-map-" + std::to_string(index);
        std::array<u8, 0x1000> initial{};
        memory.ReadBlockUnsafe(address, initial.data(), initial.size());
        std::ofstream(prefix + ".bin", std::ios::binary).write(
            reinterpret_cast<const char*>(initial.data()), initial.size());
        std::ofstream(prefix + ".txt") << "gpu " << std::hex << page << "\ncpu " << address
                                       << "\nsize 1000\n";
        memory.MarkRegionDebug(address, 0x1000, true);
    }
}

inline thread_local const char* host_caller = "unknown";
struct HostWriteScope {
    explicit HostWriteScope(const char* caller) : previous(host_caller) { host_caller = caller; }
    ~HostWriteScope() { host_caller = previous; }
    const char* previous;
};

inline void HostStore(const char* tag, const void* destination, const void* source, u64 size) {
    if (!Enabled()) return;
    auto& state = GetState();
    const u64 pointer = reinterpret_cast<u64>(destination);
    const u32 count = state.count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        const auto range = state.ranges[i];
        if (range.gpu != state.focus_gpu.load(std::memory_order_relaxed)) continue;
        if (!range.host || pointer >= range.host + range.size || pointer + size <= range.host) continue;
        const u64 begin = (std::max)(pointer, range.host);
        const u64 end = (std::min)(pointer + size, range.host + range.size);
        if (std::memcmp(reinterpret_cast<const void*>(begin),
                        static_cast<const char*>(source) + begin - pointer, end - begin) == 0) continue;
        const std::scoped_lock lock{state.mutex};
        if (state.host_records >= 8192) {
            std::ofstream("native-host-write-limit.txt") << "Host write trace capacity exceeded.\n";
            return;
        }
        const std::string prefix = "native-host-write-" + std::to_string(state.host_records++);
        std::ofstream(prefix + ".txt") << "tag " << tag << "\ncaller " << host_caller << "\ngpu " << std::hex
            << range.gpu + begin - range.host << "\ncpu " << range.cpu + begin - range.host
            << std::dec << "\nbytes " << end - begin << "\nfull_bytes " << size
            << "\ncpu_records " << state.records << '\n';
        std::ofstream(prefix + ".bin", std::ios::binary).write(
            static_cast<const char*>(source) + begin - pointer, end - begin);
        std::ofstream(prefix + "-before.bin", std::ios::binary).write(
            reinterpret_cast<const char*>(begin), end - begin);
    }
}

template <typename Jit>
void Store(Jit& jit, u64 address, u64 size, u64 low, u64 high = 0) {
    const auto range = Find(address, size);
    if (!range.size) return;
    auto& state = GetState();
    const std::scoped_lock lock{state.mutex};
    // Diagnostic scene anchor: the wall's first matrix row is (1, 0, 0, 65).
    // Focus host tracing only after the CPU attempts that row, without touching guest data.
    if (size == 16 && low == 0x000000003f800000ULL && high == 0x4282000000000000ULL) {
        state.focus_gpu.store(range.gpu, std::memory_order_relaxed);
    }
    if (state.records >= 1000000) {
        if (state.records == 1000000) {
            ++state.records;
            std::ofstream("native-cpu-write-limit.txt") << "Write trace capacity exceeded.\n";
        }
        return;
    }
    if (!state.writes.is_open()) state.writes.open("native-cpu-writes.bin", std::ios::binary);
    // Eleven little-endian u64 fields; PC can name the current JIT block, so retain LR and FP.
    const std::array<u64, 11> record{range.gpu + address - range.cpu, address, size, low, high,
        jit->GetPC(), jit->GetRegister(30), jit->GetRegister(29), jit->GetRegister(0),
        jit->GetRegister(1), jit->GetRegister(2)};
    state.writes.write(reinterpret_cast<const char*>(record.data()), sizeof(record));
    if ((++state.records & 255) == 0) state.writes.flush();
}
} // namespace Core::NativeCpuWriteTrace
