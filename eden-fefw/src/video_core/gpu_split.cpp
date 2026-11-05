// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#include <charconv>
#include <exception>
#include <string_view>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "common/windows_cpu_set_reservation.h"
#endif
#include <cstdio>

#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/gpu_split.h"
#include "video_core/native_draw_optimizations.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};
std::atomic<bool> g_back_busy{};

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    if (!g_back_busy.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::fprintf(stderr, "GPU split owner check: %s on thread %lu while the back runs\n", where,
                 thread);
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        std::fprintf(stderr, "  frame %s+%#llx\n", base_name ? base_name + 1 : name,
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(frames[i]) -
                                                     reinterpret_cast<uintptr_t>(module)));
    }
    std::fflush(stderr);
#else
    (void)where;
#endif
}

GpuSplit GpuSplit::instance;

bool GpuSplit::Requested() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_TEST_GPU_SPLIT");
        return value && (value[0] == '1' || value[0] == '2') &&
               std::getenv("EDEN_NATIVE_STRICT") == nullptr;
    }();
    return enabled;
}

void GpuSplit::MaybeStart(Core::System& system_, bool native_channel) {
    const bool allowed = Requested() &&
        NativeDrawOptimizations::gpu_split_enabled.load(std::memory_order_relaxed) &&
        !Settings::IsGPULevelHigh() &&
        Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "Ordered graphics worker stopped");
    }
    if (active || !allowed || !native_channel) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_TEST_GPU_SPLIT_CHECK") != nullptr;
    ring = std::make_unique<u64[]>(RingWords);
    back_thread = std::jthread([this](std::stop_token stop_token) {
#ifdef _WIN32
        Common::WindowsCpuSetReservation cpu_placement;
        if (const char* value = std::getenv("EDEN_RENDER_WORKER_CPU")) {
            const std::string_view requested{value};
            unsigned processor{};
            const auto parsed = std::from_chars(requested.data(), requested.data() + requested.size(),
                                                processor);
            try {
                if (parsed.ec == std::errc{} && parsed.ptr == requested.data() + requested.size() &&
                    cpu_placement.Enable(processor)) {
                    LOG_INFO(HW_GPU, "Render worker CPU placement enabled: logical processor {}",
                             processor);
                } else {
                    LOG_WARNING(HW_GPU, "Requested render worker CPU placement is unavailable");
                }
            } catch (const std::exception& error) {
                LOG_WARNING(HW_GPU, "Render worker CPU placement failed: {}", error.what());
            }
        }
#endif
        BackLoop(stop_token);
#ifdef _WIN32
        if (!cpu_placement.Reset()) {
            LOG_ERROR(HW_GPU, "Failed to restore render worker CPU placement");
        }
#endif
    });
    active = true;
#ifdef ARCHITECTURE_x86_64
    const char* mode = std::getenv("EDEN_TEST_GPU_SPLIT");
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode && mode[0] == '2' && (!jit || jit[0] != '0') &&
                   std::getenv("EDEN_REFRESH_MACRO_PARAMS") == nullptr;
#endif
    LOG_INFO(HW_GPU, "Front graphics macros: {}", front_macros);
    LOG_INFO(HW_GPU, "Ordered graphics worker started; coherent command reads preserved");
}

void GpuSplit::RegisterFrontend(Tegra::Engines::EngineInterface* engine) {
    if (std::find(frontend_engines.begin(), frontend_engines.end(), engine) == frontend_engines.end()) {
        frontend_engines.push_back(engine);
    }
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    Drain();
    for (auto* engine : frontend_engines) engine->EndFrontend(*system);
    frontend_engines.clear();
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    last_writer_position = 0;
    published_local = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    back_engine = nullptr;
    front_macros = false;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
    wait_target.store(~u64{0}, std::memory_order_relaxed);
    wake.store(0, std::memory_order_relaxed);
    front_waiting.store(false, std::memory_order_relaxed);
    front_wake.store(0, std::memory_order_relaxed);
    system = nullptr;
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();

    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::Publish() {
    if (published_local == write_position) {
        return;
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void GpuSplit::Drain() {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position) {
        return;
    }
    Publish();

    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            publish_every = 8;
            return;
        }
        if (spins < 16384) {
            Pause();
        } else if (NativeDrawOptimizations::park_front_waits.load(std::memory_order_relaxed)) {
            WaitForProgress(consumed_cache);
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::WaitForWriters() {
    const u64 target = std::min(last_writer_position, write_position);
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return;
    }
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            return;
        }
        if (spins < 16384) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    front_engine = engine;
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment) {
    SelectEngine(engine);
    // Some inline uploads require one complete call, even when the payload exceeds the ring's
    // small-entry limit. Own such payloads separately rather than changing the engine boundary.
    if (amount > MaxMultiWords) {
        std::vector<u32> owned(words, words + amount);
        const GPUVAddr saved_segment = segment != nullptr ? *segment : 0;
        DeferFunction([this, engine, method, methods_pending, saved_segment,
                       has_segment = segment != nullptr, owned = std::move(owned)] {
            engine->ConsumeSink(*system);
            if (has_segment) engine->current_dma_segment = saved_segment;
            engine->CallMultiMethod(*system, method, owned.data(),
                                    static_cast<u32>(owned.size()), methods_pending);
        });
        return;
    }
    if (amount == 0) return;
    const size_t head = segment != nullptr ? 3 : 2;
    const size_t size = head + (amount + 1) / 2;
    u64* const entry = Reserve(size);
    entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
               (u64{method & 0xffff} << 8) | (u64{amount} << 32);
    entry[1] = methods_pending;
    if (segment != nullptr) entry[2] = *segment;
    entry[size - 1] = 0;
    std::memcpy(entry + head, words, size_t{amount} * sizeof(u32));
    Commit(size);
}

void GpuSplit::DeferFunction(std::function<void()>&& func) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer([heap] {
        (*heap)();
        delete heap;
    });
}

size_t GpuSplit::Execute(const u64* entry) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if ((head & Direct) != 0) {

            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if (!engine->execution_mask[method]) {
            engine->method_sink.emplace_back(method, argument);
        } else {

            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;

        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::Call: {
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        return 2 + payload;
    }
    }
    return 1;
}

void GpuSplit::PublishConsumed(u64 position) {
    // Pair the completion store with waiter registration so neither side can miss the other.
    consumed.store(position, std::memory_order_seq_cst);
    if (front_waiting.load(std::memory_order_seq_cst)) {
        front_wake.fetch_add(1, std::memory_order_seq_cst);
        front_wake.notify_one();
    }
}

void GpuSplit::WaitForProgress(u64 observed) {
    front_waiting.store(true, std::memory_order_seq_cst);
    const u32 seen = front_wake.load(std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) == observed) {
        front_wake.wait(seen, std::memory_order_seq_cst);
    }
    front_waiting.store(false, std::memory_order_relaxed);
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    Common::SetCurrentThreadName("GPUBack");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    while (!stop_token.stop_requested()) {

        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {

            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < 8192 && available == read_position; ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                if (available == read_position && !stop_token.stop_requested()) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        u64 last_store = read_position;
        g_back_busy.store(true, std::memory_order_relaxed);
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            const size_t size = Execute(&ring[index]);
            read_position += size != 0 ? size : RingWords - index;
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                PublishConsumed(read_position);
                last_store = read_position;
            }
        }
        g_back_busy.store(false, std::memory_order_release);
        PublishConsumed(read_position);
    }
}

} // namespace VideoCommon

