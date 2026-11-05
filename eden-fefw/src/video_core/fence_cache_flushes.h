// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <mutex>

namespace VideoCommon {

struct FenceCacheFlushes {
    bool buffers{true};
    bool textures{true};
    bool needs_fence{};
};

template <class BufferCache, class TextureCache, class QueryCache>
FenceCacheFlushes CommitFenceCacheFlushes(BufferCache& buffers, TextureCache& textures,
                                        QueryCache& queries) {
    FenceCacheFlushes flushes;
    {
        std::scoped_lock lock{buffers.mutex, textures.mutex};
        flushes.buffers = buffers.HasUncommittedFlushes();
        flushes.textures = textures.HasUncommittedFlushes();
        flushes.needs_fence = flushes.buffers || flushes.textures ||
                             queries.HasUncommittedFlushes();
        if (flushes.textures) {
            textures.CommitAsyncFlushes();
        }
        if (flushes.buffers) {
            buffers.CommitAsyncFlushes();
        }
    }
    // Even an empty query batch performs WFI and schedules pending-query retirement.
    queries.CommitAsyncFlushes();
    return flushes;
}

template <class BufferCache, class TextureCache, class QueryCache>
void PopFenceCacheFlushes(const FenceCacheFlushes& flushes, BufferCache& buffers,
                         TextureCache& textures, QueryCache& queries) {
    if (flushes.buffers || flushes.textures) {
        // Nonempty writebacks retain joint exclusion from draws and CPU invalidation.
        std::scoped_lock lock{buffers.mutex, textures.mutex};
        if (flushes.textures) {
            textures.PopAsyncFlushes();
        }
        if (flushes.buffers) {
            buffers.PopAsyncFlushes();
        }
    }
    queries.PopAsyncFlushes();
}

} // namespace VideoCommon
