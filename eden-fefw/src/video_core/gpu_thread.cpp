// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>

#include "common/assert.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"
#include "common/timeline.h"

namespace VideoCommon::GPUThread {

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() { NotifyShutdown(); }

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    std::scoped_lock mapping_lock{mapping_submission_mutex};
    rasterizer = renderer.ReadRasterizer();
    running = true;
    thread = std::jthread([&](std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPU");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        auto& split = VideoCommon::GpuSplit::Get();
        SCOPE_EXIT { split.Stop(); };
        CommandDataContainer next;
        while (!stop_token.stop_requested()) {
            if (scheduler.HasPendingNativeChannels()) {
                scheduler.ResumePending(system.GPU());
                if (!state.queue.TryPop(next)) {
                    split.Publish();
                    std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    continue;
                }
            } else if (!state.queue.TryPop(next)) {
                if (split.Active()) {
                    split.Defer([rasterizer_ptr = rasterizer] { rasterizer_ptr->OnGpuThreadIdle(); });
                    split.Publish();
                } else {
                    rasterizer->OnGpuThreadIdle();
                }
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            Common::Timeline::Mark("gpu_begin", next.data.index(), next.fence);
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (auto* mapping = std::get_if<MappingCommand>(&next.data)) {
                split.Drain();
                mapping->func();
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                if (split.Active()) {
                    split.Defer([gpu = &system.GPU()] { gpu->TickWork(); });
                } else {
                    system.GPU().TickWork();
                }
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                if (split.Active()) {
                    split.Defer([rasterizer_ptr = rasterizer, addr = flush->addr, size = flush->size] {
                        rasterizer_ptr->FlushRegion(addr, size);
                    });
                } else {
                    rasterizer->FlushRegion(flush->addr, flush->size);
                }
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                if (split.Active()) {
                    split.Defer([rasterizer_ptr = rasterizer, addr = invalidate->addr, size = invalidate->size] {
                        rasterizer_ptr->OnCacheInvalidation(addr, size);
                    });
                } else {
                    rasterizer->OnCacheInvalidation(invalidate->addr, invalidate->size);
                }
            } else {
                ASSERT(false);
            }
            // DispatchCalls can start the worker while handling the first native submission.
            const auto complete = [this, fence = next.fence, block = next.block,
                                   command_index = next.data.index()] {
                state.signaled_fence.store(fence);
                Common::Timeline::Mark("gpu_end", command_index, fence);
                if (block) {
                    std::scoped_lock lk{state.write_lock};
                    state.cv.notify_all();
                }
            };
            if (split.Active()) {
                split.Defer(complete);
                split.Publish();
            } else {
                complete();
            }
            if (static const long delay_us = [] { const char* v = std::getenv("EDEN_GPU_THREAD_DELAY_US"); return v ? std::atol(v) : 0L; }(); delay_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds{delay_us});
            }
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::RunMapping(std::function<void()>&& func) {
    std::scoped_lock mapping_lock{mapping_submission_mutex};
    if (running) {
        PushCommand(MappingCommand{std::move(func)}, true, true);
    } else {
        func();
    }
}

void ThreadManager::NotifyShutdown() {
    std::scoped_lock mapping_lock{mapping_submission_mutex};
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    running = false;
}

} // namespace VideoCommon::GPUThread
