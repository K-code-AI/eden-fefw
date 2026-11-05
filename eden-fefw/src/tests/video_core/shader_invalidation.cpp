// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <random>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/device_memory.h"
#include "video_core/native_draw_optimizations.h"
#include "video_core/shader_cache.h"

namespace VideoCommon {
struct ShaderCacheTestAccess {
    static std::unique_ptr<ShaderCache> Create(Tegra::MaxwellDeviceMemoryManager& memory) {
        return std::unique_ptr<ShaderCache>{new ShaderCache{memory}};
    }
    static void Register(ShaderCache& cache, VAddr address, size_t size, u64 hash) {
        auto shader = std::make_unique<ShaderInfo>();
        shader->unique_hash = hash;
        cache.Register(std::move(shader), address, size);
    }
    static std::optional<u64> Find(const ShaderCache& cache, VAddr address) {
        const auto* shader = cache.TryGet(address);
        return shader ? std::optional{shader->unique_hash} : std::nullopt;
    }
    static void Reject(ShaderCache& cache, VAddr address, size_t size) {
        cache.rejected_shaders.insert_or_assign(address,
            ShaderCache::RejectedShader{123, size, "test rejection"});
        cache.max_rejected_prefix = (std::max)(cache.max_rejected_prefix, size);
        cache.has_rejections.store(true);
    }
    static bool HasRejections(const ShaderCache& cache) {
        return cache.has_rejections.load();
    }
    static bool IsRejected(const ShaderCache& cache, VAddr address) {
        return cache.rejected_shaders.contains(address);
    }
};
} // namespace VideoCommon

using VideoCommon::ShaderCacheTestAccess;

TEST_CASE("Shader invalidation batches match sequential overlapping writes", "[shader_invalidation]") {
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager device_memory{memory};
    device_memory.RegisterProcess(nullptr);
    auto batched = ShaderCacheTestAccess::Create(device_memory);
    auto sequential = ShaderCacheTestAccess::Create(device_memory);
    std::mt19937 random{0x16'67};
    constexpr VAddr base = 0x100000;
    for (unsigned iteration = 0; iteration < 600; ++iteration) {
        for (unsigned slot = 0; slot < 128; ++slot) {
            const auto address = base + slot * 4096;
            if (!ShaderCacheTestAccess::Find(*batched, address)) {
                const auto size = 1 + random() % 65536;
                ShaderCacheTestAccess::Register(*batched, address, size, iteration + slot);
                ShaderCacheTestAccess::Register(*sequential, address, size, iteration + slot);
            }
            ShaderCacheTestAccess::Reject(*batched, address, 1 + slot * 17);
            ShaderCacheTestAccess::Reject(*sequential, address, 1 + slot * 17);
        }
        if (iteration % 3 == 0) {
            const auto address = base + random() % (128 * 4096);
            batched->OnCacheInvalidation(address, 256);
            sequential->OnCacheInvalidation(address, 256);
        }
        std::vector<std::pair<DAddr, size_t>> writes;
        for (unsigned i = 0, count = random() % 24; i < count; ++i) {
            writes.emplace_back(base + random() % (128 * 4096), 1 + random() % 32768);
            if (i % 4 == 0) {
                writes.push_back(writes.back());
            }
        }
        for (const auto& [address, size] : writes) {
            sequential->InvalidateRegion(address, size);
        }
        batched->InvalidateRegions(writes);
        for (unsigned slot = 0; slot < 128; ++slot) {
            const auto address = base + slot * 4096;
            REQUIRE(ShaderCacheTestAccess::Find(*batched, address) ==
                    ShaderCacheTestAccess::Find(*sequential, address));
            REQUIRE(ShaderCacheTestAccess::IsRejected(*batched, address) ==
                    ShaderCacheTestAccess::IsRejected(*sequential, address));
        }
    }
}

TEST_CASE("Empty shader batches retain pending removals", "[shader_invalidation]") {
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager device_memory{memory};
    device_memory.RegisterProcess(nullptr);
    auto cache = ShaderCacheTestAccess::Create(device_memory);
    constexpr VAddr address = 0x100000;
    ShaderCacheTestAccess::Register(*cache, address, 32768, 123);
    cache->OnCacheInvalidation(address, 1);
    cache->InvalidateRegions({});
    REQUIRE(ShaderCacheTestAccess::Find(*cache, address) == u64{123});
    const std::array unrelated{std::pair<DAddr, size_t>{address + 0x100000, 4096}};
    cache->InvalidateRegions(unrelated);
    REQUIRE_FALSE(ShaderCacheTestAccess::Find(*cache, address));
}

TEST_CASE("Shader batches invalidate rejection prefixes at exact boundaries", "[shader_invalidation]") {
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager device_memory{memory};
    device_memory.RegisterProcess(nullptr);
    auto cache = ShaderCacheTestAccess::Create(device_memory);
    constexpr VAddr base = 0x100000;
    ShaderCacheTestAccess::Reject(*cache, base, 64);
    ShaderCacheTestAccess::Reject(*cache, base + 64, 64);
    ShaderCacheTestAccess::Reject(*cache, base + 0x4000 - 4, 64);
    const std::array writes{
        std::pair<DAddr, size_t>{base + 64, 1},
        std::pair<DAddr, size_t>{base + 0x4000, 1}};
    cache->InvalidateRegions(writes);
    REQUIRE(ShaderCacheTestAccess::IsRejected(*cache, base));
    REQUIRE_FALSE(ShaderCacheTestAccess::IsRejected(*cache, base + 64));
    REQUIRE_FALSE(ShaderCacheTestAccess::IsRejected(*cache, base + 0x4000 - 4));
    const std::array last_byte{std::pair<DAddr, size_t>{base + 63, 1}};
    cache->InvalidateRegions(last_byte);
    REQUIRE_FALSE(ShaderCacheTestAccess::IsRejected(*cache, base));
}

TEST_CASE("Rejection flag skips only writes with no remembered rejections", "[shader_invalidation]") {
    namespace Optimizations = VideoCommon::NativeDrawOptimizations;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager device_memory{memory};
    device_memory.RegisterProcess(nullptr);
    auto skipping = ShaderCacheTestAccess::Create(device_memory);
    auto locked = ShaderCacheTestAccess::Create(device_memory);
    std::mt19937 random{0x2026'1104};
    constexpr VAddr base = 0x100000;
    const bool previous = Optimizations::skip_empty_rejections.load();
    for (unsigned iteration = 0; iteration < 2000; ++iteration) {
        if (random() % 8 == 0) {
            const auto address = base + (random() % 64) * 4096;
            const auto size = 1 + random() % 256;
            ShaderCacheTestAccess::Reject(*skipping, address, size);
            ShaderCacheTestAccess::Reject(*locked, address, size);
        }
        const auto address = base + random() % (64 * 4096);
        const auto size = 1 + random() % 8192;
        Optimizations::skip_empty_rejections.store(true);
        skipping->InvalidateRegion(address, size);
        Optimizations::skip_empty_rejections.store(false);
        locked->InvalidateRegion(address, size);
        bool any = false;
        for (unsigned slot = 0; slot < 64; ++slot) {
            const auto rejected = ShaderCacheTestAccess::IsRejected(*skipping, base + slot * 4096);
            REQUIRE(rejected == ShaderCacheTestAccess::IsRejected(*locked, base + slot * 4096));
            any |= rejected;
        }
        REQUIRE(ShaderCacheTestAccess::HasRejections(*skipping) == any);
    }
    Optimizations::skip_empty_rejections.store(previous);
}
