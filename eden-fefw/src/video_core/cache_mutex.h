// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <mutex>

#ifdef _WIN32
#include <cstdlib>
#include <system_error>
#include <windows.h>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "video_core/native_draw_optimizations.h"

namespace VideoCommon {

// Both backends retain recursive ownership and support std::scoped_lock.
class CacheMutex {
public:
    CacheMutex() {
#ifdef _WIN32
        if (native_enabled &&
            !InitializeCriticalSectionEx(&native_mutex, 1024, CRITICAL_SECTION_NO_DEBUG_INFO)) {
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                    "Cache critical section initialization failed");
        }
#endif
    }

    ~CacheMutex() {
#ifdef _WIN32
        if (native_enabled) DeleteCriticalSection(&native_mutex);
#endif
    }

    CacheMutex(const CacheMutex&) = delete;
    CacheMutex& operator=(const CacheMutex&) = delete;

    void lock() {
#ifdef _WIN32
        if (native_enabled) {
            EnterCriticalSection(&native_mutex);
            return;
        }
#endif
#if defined(_M_X64) || defined(__x86_64__)
        const auto spins = (std::min)(NativeDrawOptimizations::cache_lock_spins.load(
                                         std::memory_order_relaxed), std::uint32_t{4096});
        for (std::uint32_t attempt = 0; attempt < spins; ++attempt) {
            if (mutex.try_lock()) {
                return;
            }
            _mm_pause();
        }
#endif
        mutex.lock();
    }

    bool try_lock() {
#ifdef _WIN32
        if (native_enabled) return TryEnterCriticalSection(&native_mutex) != 0;
#endif
        return mutex.try_lock();
    }

    void unlock() {
#ifdef _WIN32
        if (native_enabled) {
            LeaveCriticalSection(&native_mutex);
            return;
        }
#endif
        mutex.unlock();
    }

private:
    std::recursive_mutex mutex;
#ifdef _WIN32
    // The backend is fixed for this object's lifetime, including recursive acquisitions.
    const bool native_enabled = std::getenv("EDEN_TEST_NATIVE_CACHE_LOCK") != nullptr;
    CRITICAL_SECTION native_mutex{};
#endif
};

} // namespace VideoCommon
