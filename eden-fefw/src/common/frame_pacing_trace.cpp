// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>
#include "common/frame_pacing_trace.h"
#ifdef _WIN32
#include <windows.h>
#endif

namespace Common::FramePacing {

std::atomic<bool> active{};

namespace {
using Clock = std::chrono::steady_clock;
struct Event {
    u64 ns;
    const char* tag;
    u64 a;
    u64 b;
    u32 tid;
};
constexpr size_t Capacity = 1 << 20;
std::vector<Event> events;
std::atomic<size_t> next{};
std::atomic<u32> writers{};
Clock::time_point origin;
Clock::time_point deadline;
} // namespace

void Record(const char* tag, u64 a, u64 b) {
    writers.fetch_add(1);
    if (active.load()) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - origin);
        const auto slot = next.fetch_add(1, std::memory_order_relaxed);
        if (slot < events.size()) {
#ifdef _WIN32
            const u32 tid = GetCurrentThreadId();
#else
            const u32 tid = static_cast<u32>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
            events[slot] = Event{static_cast<u64>(ns.count()), tag, a, b, tid};
        }
    }
    writers.fetch_sub(1);
}

void Poll() {
    static const bool enabled = std::getenv("EDEN_FRAME_PACING_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    static auto next_poll = Clock::now();
    const auto now = Clock::now();
    if (active.load(std::memory_order_relaxed)) {
        if (now < deadline) {
            return;
        }
        active.store(false);
        while (writers.load() != 0) {
            std::this_thread::yield();
        }
        if (auto* out = std::fopen("frame-pacing-trace.tsv", "w")) {
            const auto count = next.load();
            // origin_ns: steady clock epoch offset of ns=0, for aligning other steady-clock logs.
            const auto origin_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(origin.time_since_epoch());
            std::fprintf(out, "# events=%zu capacity=%zu origin_ns=%lld\nns\ttid\ttag\ta\tb\n",
                         count, Capacity, static_cast<long long>(origin_ns.count()));
            for (size_t i = 0; i < std::min(count, Capacity); ++i) {
                const auto& e = events[i];
                std::fprintf(out, "%llu\t%u\t%s\t%llu\t%llu\n",
                             static_cast<unsigned long long>(e.ns), e.tid, e.tag,
                             static_cast<unsigned long long>(e.a),
                             static_cast<unsigned long long>(e.b));
            }
            std::fclose(out);
        }
        next_poll = Clock::now() + std::chrono::seconds{1};
        return;
    }
    if (now < next_poll) {
        return;
    }
    next_poll = now + std::chrono::seconds{1};
    auto* request = std::fopen("frame-pacing-trace.request", "r");
    if (!request) {
        return;
    }
    unsigned seconds{};
    const auto parsed = std::fscanf(request, "%u", &seconds);
    std::fclose(request);
    std::error_code error;
    std::filesystem::remove("frame-pacing-trace.request", error);
    if (parsed != 1 || seconds < 1 || seconds > 60 || error) {
        return;
    }
    events.resize(Capacity);
    next.store(0);
    origin = Clock::now();
    deadline = origin + std::chrono::seconds{seconds};
    active.store(true);
}

} // namespace Common::FramePacing
