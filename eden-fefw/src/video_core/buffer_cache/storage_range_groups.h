// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <boost/container/small_vector.hpp>
#include "common/common_types.h"

namespace VideoCommon {
struct StorageAddressRange { u64 address; u32 size; };
struct StorageRangeGroup {
    u64 address{}, end{};
    boost::container::small_vector<size_t, 8> members;
};
// Grouping runs for every draw with storage buffers; inline capacity keeps it off the heap.
using StorageRangeGroups = boost::container::small_vector<StorageRangeGroup, 8>;
// Overlapping virtual views must share one backing allocation so writes through
// one descriptor remain visible through every alias in the same dispatch.
inline StorageRangeGroups GroupStorageRanges(std::span<const StorageAddressRange> ranges) {
    boost::container::small_vector<size_t, 16> order;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (!ranges[i].address || !ranges[i].size) continue;
        if (ranges[i].address > std::numeric_limits<u64>::max() - ranges[i].size)
            throw std::overflow_error("Storage address range overflow");
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ranges[a].address < ranges[b].address; });
    StorageRangeGroups groups;
    for (size_t i : order) {
        const u64 address = ranges[i].address, end = address + ranges[i].size;
        if (groups.empty() || address >= groups.back().end) {
            auto& group = groups.emplace_back();
            group.address = address;
            group.end = end;
            group.members.push_back(i);
        } else {
            groups.back().end = std::max(groups.back().end, end);
            groups.back().members.push_back(i);
        }
    }
    return groups;
}
} // namespace VideoCommon
