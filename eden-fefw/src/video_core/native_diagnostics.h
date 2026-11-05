// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>

#include "common/settings.h"

namespace VideoCommon {
// Local comparisons: bit 0 disables sparse uniforms, bit 1 copies live commands,
// bits 2/3 relax equal-depth tests, bit 4 reads fresh command entries, bit 5 resets
// Vulkan dynamic state, and bit 6 waits for GPU work before each draw.
// Uniform comparisons: bit 7 gathers all native uniforms with GPU writeback;
// bit 8 streams CPU data directly (diagnostic only), bit 9 disables the size-based
// streaming shortcut while retaining required alignment handling. Bit 10 disables deferred clears.
inline bool NativeRenderTestFlag(u8 bit) {
    static const bool enabled = std::getenv("EDEN_TEST_RENDER_AB") != nullptr;
    return enabled && Settings::GetDebugKnobAt(bit);
}

inline bool NativeDrawDiagnosticsEnabled() {
    static const bool enabled = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
    return enabled;
}
// EDEN_NATIVE_TEXTURE_REQUESTS: only the request-driven readbacks (native-gpu-texture.request,
// native-dispatch-trace.request, native-draw-ps.request), without the rest of the draw diagnostics,
// which slow the game to a few fps.
inline bool NativeTextureRequestsEnabled() {
    static const bool enabled =
        NativeDrawDiagnosticsEnabled() || std::getenv("EDEN_NATIVE_TEXTURE_REQUESTS") != nullptr;
    return enabled;
}
inline bool NativeFrameCaptureEnabled() {
    static const bool enabled = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    return enabled;
}
inline bool NativeDrawCaptureEnabled() {
    static const bool enabled = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
    return enabled;
}
// True when a "*.request" file may be waiting in the working directory. The directory is scanned at
// most every 100 ms; checking each request file on every draw and dispatch slowed diagnostic runs to
// about 1 fps. A request is therefore picked up up to 100 ms late.
inline bool NativeRequestPending() {
    using Clock = std::chrono::steady_clock;
    static std::atomic<Clock::rep> next_scan{};
    static std::atomic<bool> pending{};
    const Clock::rep now = Clock::now().time_since_epoch().count();
    Clock::rep due = next_scan.load(std::memory_order_relaxed);
    if (now >= due && next_scan.compare_exchange_strong(
                          due, now + std::chrono::duration_cast<Clock::duration>(
                                         std::chrono::milliseconds(100)).count())) {
        // Known names only: run directories hold thousands of dump files, too many to list often.
        static constexpr const char* names[] = {
            "native-gpu-texture.request", "native-dispatch-trace.request",
            "native-dlss-snapshot.request", "native-draw-ps.request", "native-draw.request",
            "native-twin.request", "native-zero.request", "native-layer-dump.request",
            "native-display-texture.request",
            "native-window-draw.request",
            "native-depth-trace.request",
            "native-compute.request", "native-memory.request", "native-textures.request",
        };
        bool found = false;
        std::error_code error;
        for (const char* name : names) {
            if (std::filesystem::exists(name, error)) {
                found = true;
                break;
            }
        }
        pending.store(found, std::memory_order_relaxed);
    }
    return pending.load(std::memory_order_relaxed);
}
// Remaining compute dispatches whose grid/kind is logged (armed by native-dispatch-trace.request).
inline std::atomic<std::uint32_t>& NativeDispatchDimsBudget() {
    static std::atomic<std::uint32_t> budget{};
    return budget;
}
} // namespace VideoCommon
