// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/assert.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename T>
class DescriptorTable {
public:
    [[nodiscard]] bool Synchronize(GPUVAddr gpu_addr, u32 limit) noexcept {
        bool ret = !(current_gpu_addr == gpu_addr && current_limit == limit);
        if (ret) {
            Refresh(gpu_addr, limit);
        }
        return ret;
    }

    void Invalidate() noexcept {
        std::ranges::fill(read_descriptors, 0);
    }

    // Always read current guest bytes, but let hot callers avoid returning a
    // descriptor-sized copy when they only need to check whether it changed.
    [[nodiscard]] bool Update(Tegra::MemoryManager const& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        T value;
        bool changed;
        const u8* source = nullptr;
        if (sizeof(T) <= Core::DEVICE_PAGESIZE - (gpu_addr & Core::DEVICE_PAGEMASK)) {
            source = gpu_memory.GetPointer(gpu_addr);
        }
        if (source) {
            std::memcpy(std::addressof(value), source, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(value), sizeof(T));
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            changed = value != descriptors[index];
        } else {
            read_descriptors[index / 64] |= 1ULL << (index % 64);
            changed = true;
        }
        if (changed) {
            descriptors[index] = value;
        }
        return changed;
    }

    [[nodiscard]] std::pair<T, bool> Read(Tegra::MemoryManager const& gpu_memory, u32 index) noexcept {
        const bool changed = Update(gpu_memory, index);
        return {descriptors[index], changed};
    }

    void Refresh(GPUVAddr gpu_addr, u32 limit) noexcept {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        size_t old_size = read_descriptors.size();
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        old_size = (std::min)(old_size, read_descriptors.size());
        std::fill(read_descriptors.begin(), read_descriptors.begin() + old_size, 0);
        //
        descriptors.resize(num_descriptors);
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
};

} // namespace VideoCommon
