// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Common::HostSampler {

/// Starts sampling the calling thread if its name is listed in EDEN_HOST_SAMPLER.
void RegisterCurrentThread(const char* name);

} // namespace Common::HostSampler
