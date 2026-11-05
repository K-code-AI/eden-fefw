// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/timeline.h"

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Common::Timeline {
namespace {

struct Event {
    u64 ns;
    const char* tag;
    u64 a;
    u64 b;
    u32 tid;
};

constexpr size_t kCapacity = 1 << 22;

struct State {
    const std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
    bool enabled{};
    u64 start_ns{};
    u64 end_ns{};
    std::vector<Event> events;
    std::atomic<size_t> next{};
    std::atomic<bool> written{};

    State() {
        const char* env = std::getenv("EDEN_TIMELINE");
        if (!env) {
            return;
        }
        double start = 0, duration = 0;
        if (std::sscanf(env, "%lf:%lf", &start, &duration) != 2) {
            return;
        }
        start_ns = static_cast<u64>(start * 1e9);
        end_ns = static_cast<u64>((start + duration) * 1e9);
        events.resize(kCapacity);
        enabled = true;
    }

    void Write() {
        if (written.exchange(true)) {
            return;
        }
        std::FILE* file = std::fopen("timeline.txt", "w");
        if (!file) {
            return;
        }
        const size_t count = std::min(next.load(), kCapacity);
        for (size_t i = 0; i < count; ++i) {
            const Event& e = events[i];
            if (!e.tag) {
                continue;
            }
            std::fprintf(file, "%llu %u %s %#llx %#llx\n", static_cast<unsigned long long>(e.ns),
                         e.tid, e.tag, static_cast<unsigned long long>(e.a),
                         static_cast<unsigned long long>(e.b));
        }
        std::fclose(file);
    }
};

// Constructed at static initialisation so the window is measured from process start.
State global_state;

State& GetState() {
    return global_state;
}

u32 ThreadId() {
#ifdef __linux__
    thread_local const u32 tid = static_cast<u32>(syscall(SYS_gettid));
    return tid;
#else
    return 0;
#endif
}

} // namespace

bool Enabled() {
    static const bool enabled = GetState().enabled;
    return enabled;
}

void Record(const char* tag, u64 a, u64 b) {
    State& state = GetState();
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - state.origin)
                                        .count());
    if (ns < state.start_ns) {
        return;
    }
    if (ns >= state.end_ns) {
        if (!state.written.load(std::memory_order_relaxed)) {
            state.Write();
        }
        return;
    }
    const size_t slot = state.next.fetch_add(1, std::memory_order_relaxed);
    if (slot >= kCapacity) {
        return;
    }
    state.events[slot] = Event{ns, tag, a, b, ThreadId()};
}

} // namespace Common::Timeline
