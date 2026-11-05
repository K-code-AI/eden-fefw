// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include "common/common_types.h"

namespace Common::FramePacing {

extern std::atomic<bool> active;
void Record(const char* tag, u64 a, u64 b);
void Poll(); // Called only by the composition thread.

inline void Mark(const char* tag, u64 a = 0, u64 b = 0) {
    if (active.load(std::memory_order_relaxed)) {
        Record(tag, a, b);
    }
}

} // namespace Common::FramePacing
