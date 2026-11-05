// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "video_core/fence_cache_flushes.h"

namespace {

struct CountedMutex {
    std::recursive_mutex inner;
    std::atomic<unsigned> acquisitions{};
    void lock() { inner.lock(); ++acquisitions; }
    bool try_lock() {
        if (!inner.try_lock()) return false;
        ++acquisitions;
        return true;
    }
    void unlock() { inner.unlock(); }
};

struct Cache {
    CountedMutex mutex;
    std::vector<unsigned> pending;
    std::deque<std::vector<unsigned>> committed;
    std::vector<unsigned> completed;
    bool cancel_next{};
    bool HasUncommittedFlushes() const { return !pending.empty(); }
    void CommitAsyncFlushes() {
        if (cancel_next) pending.clear();
        cancel_next = false;
        committed.push_back(std::move(pending));
        pending.clear();
    }
    void PopAsyncFlushes() {
        if (committed.empty()) throw std::logic_error("Unpaired cache completion");
        const auto& batch = committed.front();
        completed.insert(completed.end(), batch.begin(), batch.end());
        committed.pop_front();
    }
};

struct Queries {
    std::mutex mutex;
    std::deque<unsigned> committed;
    unsigned next{};
    bool dirty{};
    std::vector<unsigned> completed;
    bool HasUncommittedFlushes() const { return dirty; }
    void CommitAsyncFlushes() {
        std::scoped_lock lock{mutex};
        committed.push_back(next++);
        dirty = false;
    }
    void PopAsyncFlushes() {
        std::scoped_lock lock{mutex};
        if (committed.empty()) throw std::logic_error("Unpaired query retirement");
        completed.push_back(committed.front());
        committed.pop_front();
    }
};

} // namespace

TEST_CASE("Empty fence completions avoid cache locks but retire queries", "[fence_cache_flushes]") {
    Cache buffers, textures;
    Queries queries;
    const auto batch = VideoCommon::CommitFenceCacheFlushes(buffers, textures, queries);
    REQUIRE_FALSE(batch.buffers);
    REQUIRE_FALSE(batch.textures);
    REQUIRE_FALSE(batch.needs_fence);
    const auto buffer_locks = buffers.mutex.acquisitions.load();
    const auto texture_locks = textures.mutex.acquisitions.load();
    VideoCommon::PopFenceCacheFlushes(batch, buffers, textures, queries);
    REQUIRE(buffers.mutex.acquisitions == buffer_locks);
    REQUIRE(textures.mutex.acquisitions == texture_locks);
    REQUIRE(queries.completed == std::vector<unsigned>{0});
}

TEST_CASE("Fence work masks cover every cache and retain canceled batch positions", "[fence_cache_flushes]") {
    Cache buffers, textures;
    Queries queries;
    std::vector<VideoCommon::FenceCacheFlushes> batches;
    for (unsigned mask = 0; mask != 8; ++mask) {
        if (mask & 1) buffers.pending.push_back(mask);
        if (mask & 2) textures.pending.push_back(mask);
        queries.dirty = (mask & 4) != 0;
        buffers.cancel_next = mask == 3;
        const auto batch = VideoCommon::CommitFenceCacheFlushes(buffers, textures, queries);
        REQUIRE(batch.buffers == ((mask & 1) != 0));
        REQUIRE(batch.textures == ((mask & 2) != 0));
        REQUIRE(batch.needs_fence == (mask != 0));
        batches.push_back(batch);
    }
    for (const auto batch : batches) {
        VideoCommon::PopFenceCacheFlushes(batch, buffers, textures, queries);
    }
    REQUIRE(buffers.completed == std::vector<unsigned>{1, 5, 7});
    REQUIRE(textures.completed == std::vector<unsigned>{2, 3, 6, 7});
    REQUIRE(queries.completed == std::vector<unsigned>{0, 1, 2, 3, 4, 5, 6, 7});
    REQUIRE(buffers.committed.empty());
    REQUIRE(textures.committed.empty());
}

TEST_CASE("Queued fences preserve work order while compact mode changes", "[fence_cache_flushes]") {
    Cache buffers, textures;
    Queries queries;
    std::mutex queue_mutex;
    std::condition_variable ready;
    std::deque<VideoCommon::FenceCacheFlushes> queue;
    std::vector<unsigned> expected_buffers, expected_textures, expected_queries;
    bool done{};
    std::exception_ptr worker_error;
    std::jthread consumer{[&] {
        try {
            for (;;) {
                std::unique_lock lock{queue_mutex};
                ready.wait(lock, [&] { return done || !queue.empty(); });
                if (queue.empty()) return;
                const auto batch = queue.front();
                queue.pop_front();
                lock.unlock();
                VideoCommon::PopFenceCacheFlushes(batch, buffers, textures, queries);
            }
        } catch (...) {
            worker_error = std::current_exception();
        }
    }};
    std::mt19937 random{0xF3'30};
    for (unsigned frame = 0; frame != 20000; ++frame) {
        const auto bits = random();
        VideoCommon::FenceCacheFlushes batch;
        {
            std::scoped_lock lock{buffers.mutex, textures.mutex};
            if (bits & 1) { buffers.pending.push_back(frame); expected_buffers.push_back(frame); }
            if (bits & 2) { textures.pending.push_back(frame); expected_textures.push_back(frame); }
        }
        if (bits & 4) {
            batch = VideoCommon::CommitFenceCacheFlushes(buffers, textures, queries);
        } else {
            {
                std::scoped_lock lock{buffers.mutex, textures.mutex};
                textures.CommitAsyncFlushes();
                buffers.CommitAsyncFlushes();
            }
            queries.CommitAsyncFlushes();
        }
        expected_queries.push_back(frame);
        {
            std::scoped_lock lock{queue_mutex};
            queue.push_back(batch);
        }
        ready.notify_one();
    }
    {
        std::scoped_lock lock{queue_mutex};
        done = true;
    }
    ready.notify_one();
    consumer.join();
    REQUIRE_FALSE(worker_error);
    REQUIRE(buffers.completed == expected_buffers);
    REQUIRE(textures.completed == expected_textures);
    REQUIRE(queries.completed == expected_queries);
    REQUIRE(buffers.committed.empty());
    REQUIRE(textures.committed.empty());
    REQUIRE(queries.committed.empty());
}
