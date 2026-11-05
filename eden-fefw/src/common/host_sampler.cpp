// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/host_sampler.h"

#if defined(__linux__) && defined(__x86_64__)

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

namespace Common::HostSampler {
namespace {

/// EDEN_HOST_SAMPLER=CPUCore_0,GPU samples each named thread on its own CPU-time clock (1 kHz)
/// and writes host-samples-<thread>.txt (stack -> count, cumulative) plus host-samples-maps.txt
/// to the working directory every 4 s. Diagnostic only.
constexpr int kDepth = 8;
constexpr size_t kRing = 1 << 16;

struct Sample {
    std::array<void*, kDepth> frames;
};

struct ThreadState {
    std::string name;
    std::vector<Sample> ring = std::vector<Sample>(kRing);
    std::atomic<size_t> head{};
    size_t tail{};
    std::map<std::array<void*, kDepth>, size_t> counts;
};

std::mutex threads_mutex;
std::vector<ThreadState*> threads;
thread_local ThreadState* current;

void Handler(int, siginfo_t*, void* raw) {
    ThreadState* const state = current;
    if (!state) {
        return;
    }
    const int saved_errno = errno;
    Sample sample{};
    sample.frames[0] = reinterpret_cast<void*>(
        static_cast<ucontext_t*>(raw)->uc_mcontext.gregs[REG_RIP]);
    // backtrace() walks through the signal trampoline into the interrupted frame. It stops at
    // the first JIT frame (no unwind info), which is what we want.
    std::array<void*, kDepth + 3> trace{};
    const int n = backtrace(trace.data(), static_cast<int>(trace.size()));
    int out = 1;
    for (int i = 0; i < n && out < kDepth; ++i) {
        if (trace[i] == sample.frames[0]) {
            for (int j = i + 1; j < n && out < kDepth; ++j) {
                sample.frames[out++] = trace[j];
            }
            break;
        }
    }
    const size_t slot = state->head.load(std::memory_order_relaxed);
    state->ring[slot % kRing] = sample;
    state->head.store(slot + 1, std::memory_order_release);
    errno = saved_errno;
}

void Writer() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds{4});
        std::scoped_lock lock{threads_mutex};
        for (ThreadState* state : threads) {
            const size_t head = state->head.load(std::memory_order_acquire);
            for (; state->tail < head; ++state->tail) {
                ++state->counts[state->ring[state->tail % kRing].frames];
            }
            const std::string path = "host-samples-" + state->name + ".txt";
            if (FILE* file = std::fopen(path.c_str(), "w")) {
                std::fprintf(file, "# total %zu\n", head);
                for (const auto& [frames, count] : state->counts) {
                    std::fprintf(file, "%zu", count);
                    for (void* frame : frames) {
                        if (!frame) {
                            break;
                        }
                        std::fprintf(file, " %p", frame);
                    }
                    std::fputc('\n', file);
                }
                std::fclose(file);
            }
        }
        if (FILE* in = std::fopen("/proc/self/maps", "r")) {
            if (FILE* out = std::fopen("host-samples-maps.txt", "w")) {
                char buf[4096];
                size_t len;
                while ((len = std::fread(buf, 1, sizeof(buf), in)) > 0) {
                    std::fwrite(buf, 1, len, out);
                }
                std::fclose(out);
            }
            std::fclose(in);
        }
    }
}

bool Wanted(const char* name) {
    static const char* const list = std::getenv("EDEN_HOST_SAMPLER");
    if (!list) {
        return false;
    }
    const std::string_view names{list};
    size_t pos = 0;
    while (pos <= names.size()) {
        const size_t end = std::min(names.find(',', pos), names.size());
        if (names.substr(pos, end - pos) == name) {
            return true;
        }
        pos = end + 1;
    }
    return false;
}

} // namespace

void RegisterCurrentThread(const char* name) {
    if (!Wanted(name) || current) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, [] {
        void* warm[1];
        backtrace(warm, 1); // loads libgcc outside the signal handler
        struct sigaction action {};
        action.sa_sigaction = Handler;
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&action.sa_mask);
        sigaction(SIGPROF, &action, nullptr);
        std::thread{Writer}.detach();
    });
    auto* state = new ThreadState{};
    state->name = name;
    {
        std::scoped_lock lock{threads_mutex};
        threads.push_back(state);
    }
    current = state;
    clockid_t clock;
    if (pthread_getcpuclockid(pthread_self(), &clock) != 0) {
        return;
    }
    sigevent event{};
    event.sigev_notify = SIGEV_THREAD_ID;
    event.sigev_signo = SIGPROF;
    event._sigev_un._tid = static_cast<pid_t>(syscall(SYS_gettid));
    timer_t timer;
    if (timer_create(clock, &event, &timer) != 0) {
        return;
    }
    itimerspec spec{};
    spec.it_interval.tv_nsec = 1'000'000;
    spec.it_value.tv_nsec = 1'000'000;
    timer_settime(timer, 0, &spec, nullptr);
}

} // namespace Common::HostSampler

#else

namespace Common::HostSampler {
void RegisterCurrentThread(const char*) {}
} // namespace Common::HostSampler

#endif
