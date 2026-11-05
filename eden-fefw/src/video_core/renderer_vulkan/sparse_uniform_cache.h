// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace Vulkan {

struct SparseUniformRange {
    std::uint32_t offset;
    std::uint32_t size;
    bool operator==(const SparseUniformRange&) const = default;
};

// Reuse an immutable upload only within its still-unsubmitted scheduler tick.
// Mapping layout and every mapped byte are read again before considering reuse.
template <class Ref>
class SparseUniformCache {
public:
    template <class Read, class Upload>
    std::optional<Ref> Resolve(std::uint64_t key, std::uint64_t tick, std::uint32_t size,
                               std::span<const SparseUniformRange> ranges, Read&& read,
                               Upload&& upload) {
        if (tick == 0 || size < 4096 || size > 65536 || ranges.size() > 32) return {};
        std::uint64_t end = 0, total = 0;
        for (const auto& range : ranges) {
            if (range.size == 0 || range.offset < end ||
                std::uint64_t{range.offset} + range.size > size) return {};
            end = std::uint64_t{range.offset} + range.size;
            total += range.size;
        }
        if (cooldown != 0) {
            --cooldown;
            return {};
        }
        scratch.resize(static_cast<std::size_t>(total));
        std::size_t position = 0;
        for (const auto& range : ranges) {
            read(range.offset, std::span(scratch).subspan(position, range.size));
            position += range.size;
        }
        auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
            return entry.valid && entry.key == key && entry.size == size;
        });
        if (found != entries.end() && found->tick == tick &&
            std::ranges::equal(found->ranges, ranges) && found->bytes == scratch) {
            misses = 0;
            return found->ref;
        }
        const Ref ref = upload(ranges, std::span<const std::uint8_t>(scratch));
        if (found == entries.end()) {
            found = entries.begin() + next;
            next = (next + 1) % entries.size();
        }
        found->valid = false;
        found->key = key;
        found->tick = tick;
        found->size = size;
        found->ref = ref;
        found->ranges.assign(ranges.begin(), ranges.end());
        found->bytes = scratch;
        found->valid = true;
        if (++misses >= 4) {
            misses = 0;
            cooldown = 64;
        }
        return ref;
    }

private:
    struct Entry {
        bool valid{};
        std::uint64_t key{}, tick{};
        std::uint32_t size{};
        Ref ref{};
        std::vector<SparseUniformRange> ranges;
        std::vector<std::uint8_t> bytes;
    };
    std::array<Entry, 4> entries;
    std::vector<std::uint8_t> scratch;
    std::size_t next{};
    unsigned misses{}, cooldown{};
};

} // namespace Vulkan
