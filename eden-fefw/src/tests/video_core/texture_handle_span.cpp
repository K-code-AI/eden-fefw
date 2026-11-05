// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "video_core/renderer_vulkan/draw_sampler_cache.h"
#include "video_core/renderer_vulkan/texture_handle_span.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
using Run = std::array<std::uint32_t, 3>;
struct Memory {
    static constexpr std::uint64_t First = 0x100000;
    static constexpr std::uint64_t Second = 0x200000;
    const std::uint8_t* first{};
    const std::uint8_t* second{};
    std::size_t size{};
    const void* GetSpan(std::uint64_t address, std::uint64_t bytes) {
        if (address == First && bytes <= size) return first;
        if (address == Second && bytes <= size) return second;
        return nullptr;
    }
};

void Write(std::uint8_t* bytes, std::size_t offset, std::uint32_t value) {
    std::memcpy(bytes + offset, &value, sizeof(value));
}

std::vector<Run> Collect(const Vulkan::TextureHandleSpan& handles, std::uint32_t count, bool vector) {
    std::vector<Run> runs;
    handles.ForEachRun(count, [&](auto start, auto length, auto value) {
        runs.push_back({start, length, value});
    }, vector);
    return runs;
}
} // namespace

TEST_CASE("Texture handle vector scan matches scalar across layouts", "[texture-handle]") {
    std::mt19937 random{0xa71c53};
    for (unsigned trial = 0; trial < 3000; ++trial) {
        const std::uint32_t count = 1 + random() % 1024;
        const unsigned stride = trial % 7 == 0 ? 3 : 2;
        const unsigned alignment = trial % 4;
        const bool secondary = trial % 2 != 0;
        const unsigned lhs_shift = trial % 32;
        const unsigned rhs_shift = (trial * 7) % 32;
        std::vector<std::uint8_t> lhs((std::size_t{count} << stride) + 4);
        auto rhs = lhs;
        std::uint32_t left = random(), right = random();
        for (std::uint32_t i = 0; i < count; ++i) {
            if (i % (1 + trial % 53) == 0) { left = random(); right = random(); }
            Write(lhs.data() + alignment, std::size_t{i} << stride, left);
            Write(rhs.data() + alignment, std::size_t{i} << stride, right);
        }
        Memory memory{lhs.data() + alignment, rhs.data() + alignment,
                      (std::size_t{count - 1} << stride) + 4};
        Vulkan::TextureHandleSpan handles;
        REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, count, stride,
                                secondary, lhs_shift, rhs_shift));
        REQUIRE(Collect(handles, count, true) == Collect(handles, count, false));
        REQUIRE(Collect(handles, 0, true).empty());
    }
}

TEST_CASE("Texture handle vector scan rereads after callbacks", "[texture-handle]") {
    for (bool secondary : {false, true}) {
        for (std::uint32_t boundary = 4; boundary < 37; ++boundary) {
            const auto scan = [&](bool vector) {
                std::array<std::uint8_t, 4 * 131> lhs{}, rhs{};
                for (std::uint32_t i = 0; i < 131; ++i) {
                    Write(lhs.data(), i * 4, i < boundary ? 3 : 7);
                    Write(rhs.data(), i * 4, 0x123000);
                }
                Memory memory{lhs.data(), rhs.data(), lhs.size()};
                Vulkan::TextureHandleSpan handles;
                REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 131, 2,
                                        secondary, 0, 1));
                std::vector<Run> runs;
                handles.ForEachRun(131, [&](auto start, auto length, auto value) {
                    runs.push_back({start, length, value});
                    const auto next = start + length;
                    if (next + 2 < 131) {
                        // The boundary handle is read before the callback in both paths.
                        // Handles after it must not reuse the vector's earlier lookahead.
                        Write(lhs.data(), (next + 1) * 4, 0x80000000U + next);
                        Write(rhs.data(), (next + 2) * 4, 0x40000000U + next);
                    }
                }, vector);
                return runs;
            };
            REQUIRE(scan(true) == scan(false));
        }
    }
}

TEST_CASE("Texture handle scan keeps mapping and overflow guards", "[texture-handle]") {
    std::vector<std::uint8_t> storage(65536);
    Memory memory{storage.data(), storage.data(), storage.size()};
    Vulkan::TextureHandleSpan handles;
    REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 16384, 2, true, 0, 0));
    REQUIRE(Collect(handles, 16384, true) == std::vector<Run>{{0, 16384, 0}});
    REQUIRE_FALSE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 16385, 2, false, 0, 0));
    REQUIRE_FALSE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 0, 2, false, 0, 0));
    REQUIRE_FALSE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 64, 32, false, 0, 0));
    REQUIRE_FALSE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 64, 2, true, 32, 0));
    REQUIRE_FALSE(handles.Prepare(memory, Memory::First, UINT32_MAX - 1, Memory::Second, 0, 64, 2, false, 0, 0));
    REQUIRE_FALSE(handles.Prepare(memory, UINT64_MAX - 1, 8, Memory::Second, 0, 64, 2, false, 0, 0));
    REQUIRE_FALSE(handles.Prepare(memory, Memory::First + 1, 0, Memory::Second, 0, 64, 2, false, 0, 0));
    REQUIRE_FALSE(handles.Prepare(memory, Memory::First, 0, Memory::Second + 1, 0, 64, 2, true, 0, 0));

    REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 64, 2, false, 0, 0));
    const auto before = Collect(handles, 64, true);
    Write(storage.data(), 13 * 4, 0x123);
    REQUIRE(Collect(handles, 64, true) != before);
    REQUIRE(Collect(handles, 64, true) == Collect(handles, 64, false));
}

TEST_CASE("Packed texture handle words match the general reader", "[texture-handle]") {
    std::vector<std::uint8_t> lhs(4 * 300 + 4), rhs(4 * 300 + 4);
    for (std::uint32_t i = 0; i < 300; ++i) {
        Write(lhs.data() + 1, i * 4, i * 0x9e3779b9U);
        Write(rhs.data() + 1, i * 4, i * 0x85ebca6bU);
    }
    Memory memory{lhs.data() + 1, rhs.data() + 1, 4 * 300};
    Vulkan::TextureHandleSpan handles;
    REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 300, 2, false, 0, 0));
    const auto* words = handles.PackedWords();
    REQUIRE(words != nullptr);
    for (std::uint32_t i = 0; i < 300; ++i) {
        std::uint32_t value;
        std::memcpy(&value, words + i * 4, sizeof(value));
        REQUIRE(value == handles.Read(i));
    }
    REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 300, 2, true, 0, 3));
    REQUIRE(handles.PackedWords() == nullptr);
    REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, 150, 3, false, 0, 0));
    REQUIRE(handles.PackedWords() == nullptr);
}

TEST_CASE("Sampler cache skip bound matches lookups", "[texture-handle]") {
    for (bool enabled : {false, true}) {
        Vulkan::DrawSamplerCache<std::uint32_t> cache;
        cache.SetInvalidIndexLimit(40, 7, enabled);
        cache.BeginArray();
        REQUIRE(cache.InvalidValue() == 7);
        REQUIRE(cache.SkipAbove() == (enabled ? 40U : UINT32_MAX));
        for (std::uint32_t index : {0U, 39U, 40U, 41U, 300U, UINT32_MAX}) {
            unsigned resolves = 0;
            const auto value = cache.Get(index, [&](std::uint32_t) { ++resolves; return index + 100; });
            if (index > cache.SkipAbove()) {
                REQUIRE(value == cache.InvalidValue());
                REQUIRE(resolves == 0);
            } else {
                REQUIRE(value == index + 100);
            }
        }
    }
}

#ifdef _WIN32
TEST_CASE("Texture handle vector loads stay within mapped tails", "[texture-handle]") {
    struct Pages {
        std::uint8_t* data = static_cast<std::uint8_t*>(
            VirtualAlloc(nullptr, 8192, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        ~Pages() { if (data) VirtualFree(data, 0, MEM_RELEASE); }
    } lhs, rhs;
    REQUIRE(lhs.data);
    REQUIRE(rhs.data);
    DWORD previous{};
    REQUIRE(VirtualProtect(lhs.data + 4096, 4096, PAGE_NOACCESS, &previous));
    REQUIRE(VirtualProtect(rhs.data + 4096, 4096, PAGE_NOACCESS, &previous));
    for (std::uint32_t count = 1; count <= 257; ++count) {
        auto* first = lhs.data + 4096 - count * 4;
        auto* second = rhs.data + 4096 - count * 4;
        std::memset(first, 0x33, count * 4);
        std::memset(second, 0x55, count * 4);
        Memory memory{first, second, count * 4};
        for (bool secondary : {false, true}) {
            Vulkan::TextureHandleSpan handles;
            REQUIRE(handles.Prepare(memory, Memory::First, 0, Memory::Second, 0, count, 2,
                                    secondary, 3, 7));
            REQUIRE(Collect(handles, count, true) == Collect(handles, count, false));
        }
    }
}
#endif
