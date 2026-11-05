// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

namespace Vulkan {

// Deferred descriptor updates keep raw pointers until the worker consumes them.
// Never grow or move an active allocation. Extra chunks follow the same frame
// lifetime as the base allocation and are retained for subsequent frame cycles.
template <class Entry, std::size_t FramesInFlight = 8>
class DescriptorPayloadArena {
    static_assert(FramesInFlight > 0);
    static constexpr std::size_t MAX_EXTRA_CHUNKS = 16;
    static constexpr std::size_t EXTRA_CHUNK_SIZE = 65536;

    struct Chunk {
        std::unique_ptr<Entry[]> entries;
        std::size_t capacity{};
    };
    struct Frame {
        std::array<Chunk, MAX_EXTRA_CHUNKS> extra;
        std::size_t extra_capacity{};
    };

public:
    explicit DescriptorPayloadArena(std::size_t frame_capacity_)
        : frame_capacity{frame_capacity_}, base(frame_capacity_ * FramesInFlight) {
        assert(frame_capacity != 0);
        ResetFrame();
    }

    void TickFrame() noexcept {
        frame_index = (frame_index + 1) % FramesInFlight;
        ResetFrame();
    }

    template <class Wait>
    void Acquire(std::size_t required, Wait&& wait) {
        assert(required != 0 && required <= frame_capacity);
        if (required > static_cast<std::size_t>(active_end - cursor)) {
            if (!SelectExtraChunk(required)) {
                // Memory remains bounded even if a frame generates arbitrarily
                // many descriptors. Reuse only after all queued readers finish.
                wait();
                ResetFrame();
            }
        }
        upload_start = cursor;
    }

    void Add(const Entry& entry) noexcept {
        assert(cursor < active_end);
        *(cursor++) = entry;
    }

    const Entry* UpdateData() const noexcept {
        return upload_start;
    }

    // Only valid when no queued worker command reads this acquisition.
    void Discard() noexcept {
        cursor = upload_start;
    }

    std::size_t UsedCount() const noexcept {
        return previous_used + static_cast<std::size_t>(cursor - active_start);
    }

    std::size_t ExtraCapacity() const noexcept {
        std::size_t result{};
        for (const auto& frame : frames) result += frame.extra_capacity;
        return result;
    }

private:
    void ResetFrame() noexcept {
        active_start = base.data() + frame_index * frame_capacity;
        active_end = active_start + frame_capacity;
        cursor = upload_start = active_start;
        previous_used = next_extra = 0;
    }

    bool SelectExtraChunk(std::size_t required) {
        auto& frame = frames[frame_index];
        if (next_extra == frame.extra.size()) return false;
        auto& chunk = frame.extra[next_extra];
        if (chunk.capacity < required) {
            const std::size_t capacity =
                std::max(required, std::min(frame_capacity, EXTRA_CHUNK_SIZE));
            // At most one frame allocation's worth of extra memory per slot.
            if (capacity > frame_capacity - (frame.extra_capacity - chunk.capacity)) return false;
            std::unique_ptr<Entry[]> entries{new (std::nothrow) Entry[capacity]};
            if (!entries) return false;
            frame.extra_capacity += capacity - chunk.capacity;
            chunk.entries = std::move(entries);
            chunk.capacity = capacity;
        }
        previous_used += static_cast<std::size_t>(cursor - active_start);
        active_start = cursor = chunk.entries.get();
        active_end = active_start + chunk.capacity;
        ++next_extra;
        return true;
    }

    const std::size_t frame_capacity;
    std::vector<Entry> base;
    std::array<Frame, FramesInFlight> frames;
    std::size_t frame_index{};
    std::size_t next_extra{};
    std::size_t previous_used{};
    Entry* active_start{};
    Entry* active_end{};
    Entry* cursor{};
    Entry* upload_start{};
};

} // namespace Vulkan
