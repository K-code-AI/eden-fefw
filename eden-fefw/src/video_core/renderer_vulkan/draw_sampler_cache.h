// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace Vulkan {

template <typename Value, std::size_t Size = 256>
class DrawSamplerCache {
    static_assert(Size != 0 && (Size & (Size - 1)) == 0);
public:
    void SetInvalidIndexLimit(std::uint32_t limit, Value value, bool enabled) {
        index_limit = limit;
        invalid_value = value;
        skip_invalid = enabled;
    }

    // Get returns InvalidValue() without a lookup for indices above this bound.
    std::uint32_t SkipAbove() const { return skip_invalid ? index_limit : UINT32_MAX; }
    Value InvalidValue() const { return invalid_value; }

    void BeginArray() {
        if (++generation == 0) {
            for (auto& entry : entries) entry.generation = 0;
            generation = 1;
        }
    }

    template <typename Resolve>
    Value Get(std::uint32_t index, Resolve resolve) {
        // Out-of-table indices resolve to the same null sampler without a table read.
        if (skip_invalid && index > index_limit) return invalid_value;
        auto& entry = entries[index & (Size - 1)];
        if (entry.generation == generation && entry.index == index) return entry.value;
        const Value value = resolve(index);
        entry = {index, generation, value};
        return value;
    }

private:
    struct Entry {
        std::uint32_t index{}, generation{};
        Value value{};
    };
    std::array<Entry, Size> entries{};
    std::uint32_t generation{1};
    std::uint32_t index_limit{};
    Value invalid_value{};
    bool skip_invalid{};
};

} // namespace Vulkan
