// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>

namespace VideoCommon::NativeCommandSync {

// Only register setters with no backend side effects may pass queued rendering work.
constexpr bool PullerRegisterOnly(std::uint32_t method) {
    switch (method) {
    case 0x02: // NOP
    case 0x04: // semaphore address high
    case 0x05: // semaphore address low
    case 0x06: // semaphore payload
    case 0x1c: // syncpoint payload
        return true;
    default:
        return false;
    }
}

constexpr bool ComputeRegisterOnly(std::uint32_t method) {
    // ProcessNativeMethod stores these fields without accessing shared macro code, launching
    // work, or writing guest memory. Keep all macro calls and executable methods synchronized.
    return (method >= 0x60 && method < 0x6c) || method == 0x5a ||
           method == 0xa7 || method == 0xad || (method >= 0xc6 && method < 0x107);
}

} // namespace VideoCommon::NativeCommandSync
