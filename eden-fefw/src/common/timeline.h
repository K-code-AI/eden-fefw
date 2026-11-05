// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

#ifdef _MSC_VER
#include <intrin.h>
#define EDEN_RETURN_ADDRESS() _ReturnAddress()
#else
#define EDEN_RETURN_ADDRESS() __builtin_return_address(0)
#endif

namespace Common::Timeline {

/// EDEN_TIMELINE=<start_s>:<duration_s> records Mark() calls made in that window (seconds since
/// process start) and writes timeline.txt to the working directory when the window closes.
/// Diagnostic only. `tag` must be a string literal.
bool Enabled();
void Record(const char* tag, u64 a, u64 b);

inline void Mark(const char* tag, u64 a = 0, u64 b = 0) {
    if (Enabled()) {
        Record(tag, a, b);
    }
}

} // namespace Common::Timeline
