// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/logging.h"

namespace Common::Log {

/// Tests the current filter without preparing a formatted message.
[[nodiscard]] bool IsMessageEnabled(Class log_class, Level level);

} // namespace Common::Log
