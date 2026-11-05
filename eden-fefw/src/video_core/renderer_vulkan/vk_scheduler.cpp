// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/frame_pacing_trace.h"
#include "common/scope_exit.h"
#include <algorithm>
#include <map>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "common/timeline.h"

#include "common/settings.h"
#include "video_core/native_draw_optimizations.h"
#include "common/thread.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/gpu_split.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

/// EDEN_GPU_TIMING: brackets every submitted upload+main command buffer pair with GPU timestamps
/// and appends one row per submission to gpu-timing.csv in the working directory. Rows are read
/// without waiting, once the submission's tick has signalled. cpu_ns uses the same steady clock
/// origin-free count as the frame pacing trace's clock, so offsets can be aligned afterwards.
class GpuTimer {
public:
    static bool Enabled() {
        static const bool enabled = std::getenv("EDEN_GPU_TIMING") != nullptr;
        return enabled;
    }

    explicit GpuTimer(const Device& device_) : device{device_} {
        pool = device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = kSlots * 2,
            .pipelineStatistics = 0,
        });
        period_ns = device.GetPhysical().GetProperties().limits.timestampPeriod;
        file = std::fopen("gpu-timing.csv", "w");
        if (file) {
            std::fputs("cpu_submit_ns,tick,gpu_start_ns,gpu_end_ns,rp_begins,rp_breaks\n", file);
        }
        LOG_ERROR(Render_Vulkan, "GpuTiming enabled period_ns={} file={}", period_ns,
                  file != nullptr);
    }

    ~GpuTimer() {
        if (file) {
            std::fclose(file);
        }
    }

    /// Opens the bracket at the start of a freshly begun upload command buffer.
    void Begin(vk::CommandBuffer upload_cmdbuf) {
        if (pending.size() >= kSlots) {
            current = kNone;
            return;
        }
        current = next_slot;
        next_slot = (next_slot + 1) % kSlots;
        upload_cmdbuf.ResetQueryPool(*pool, current * 2, 2);
        upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *pool, current * 2);
    }

    /// Closes the bracket at the end of the main command buffer, before it is ended.
    void End(vk::CommandBuffer cmdbuf) const {
        if (current != kNone) {
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pool, current * 2 + 1);
        }
    }

    void Submitted(u64 tick, u32 rp_begins, u32 rp_breaks) {
        if (current != kNone) {
            const auto now = std::chrono::steady_clock::now().time_since_epoch();
            pending.push_back({
                .tick = tick,
                .slot = current,
                .cpu_ns = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()),
                .rp_begins = rp_begins,
                .rp_breaks = rp_breaks,
            });
            current = kNone;
        }
    }

    /// Writes every completed submission. Slots are reused only after their row is written.
    void Collect(MasterSemaphore& semaphore) {
        if (pending.empty()) {
            return;
        }
        semaphore.Refresh();
        while (!pending.empty() && semaphore.IsFree(pending.front().tick)) {
            const Pending& entry = pending.front();
            std::array<u64, 2> stamps{};
            const VkResult result = device.GetLogical().GetQueryResults(
                *pool, entry.slot * 2, 2, sizeof(stamps), stamps.data(), sizeof(u64),
                VK_QUERY_RESULT_64_BIT);
            if (result != VK_SUCCESS) {
                break;
            }
            if (file) {
                std::fprintf(file, "%llu,%llu,%.0f,%.0f,%u,%u\n",
                             static_cast<unsigned long long>(entry.cpu_ns),
                             static_cast<unsigned long long>(entry.tick),
                             static_cast<double>(stamps[0]) * period_ns,
                             static_cast<double>(stamps[1]) * period_ns, entry.rp_begins,
                             entry.rp_breaks);
                if (++rows % 256 == 0) {
                    std::fflush(file);
                }
            }
            pending.pop_front();
        }
    }

private:
    struct Pending {
        u64 tick;
        u32 slot;
        u64 cpu_ns;
        u32 rp_begins;
        u32 rp_breaks;
    };
    static constexpr u32 kSlots = 1024;
    static constexpr u32 kNone = ~0u;

    const Device& device;
    vk::QueryPool pool;
    double period_ns{1.0};
    std::FILE* file{};
    std::deque<Pending> pending;
    u64 rows{};
    u32 next_slot{};
    u32 current{kNone};
};

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)} {

    if (thread_check_enabled) {
        LOG_ERROR(Render_Vulkan, "SchedulerThread check enabled");
    }
    if (GpuTimer::Enabled()) {
        gpu_timer = std::make_unique<GpuTimer>(device);
    }
    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() = default;

namespace {
/// EDEN_GPU_PROFILE: counts submits per caller (return address), reported every 4 s. Resolve the
/// printed offsets with `nm -C` / `addr2line` against the binary (offset is relative to NoteSubmit).
void NoteSubmit(const char* kind, void* caller) {
    static const bool enabled = std::getenv("EDEN_GPU_PROFILE") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static std::map<std::pair<const char*, void*>, u64> counts;
    static auto window_start = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++counts[{kind, caller}];
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - window_start).count() < 4.0) {
        return;
    }
    const auto base = reinterpret_cast<intptr_t>(&NoteSubmit);
    for (const auto& [key, count] : counts) {
        LOG_ERROR(Render_Vulkan, "SubmitCaller {} n={} ret_minus_NoteSubmit={}", key.first, count,
                  reinterpret_cast<intptr_t>(key.second) - base);
    }
    counts.clear();
    window_start = now;
}

#ifdef _MSC_VER
extern "C" char __ImageBase;
#endif

/// Caller address as an image-relative offset (MSVC: resolve with llvm-symbolizer
/// --relative-address), elsewhere relative to NoteSubmit.
intptr_t CallerOffset(void* caller) {
#ifdef _MSC_VER
    const auto base = reinterpret_cast<intptr_t>(&__ImageBase);
#else
    const auto base = reinterpret_cast<intptr_t>(&NoteSubmit);
#endif
    return reinterpret_cast<intptr_t>(caller) - base;
}

/// EDEN_GPU_TIMING: counts render pass breaks per caller, reported every 4 s (producer thread).
void NoteRenderPassBreak(void* caller) {
    static std::map<void*, u64> counts;
    static auto window_start = std::chrono::steady_clock::now();
    ++counts[caller];
    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(now - window_start).count();
    if (secs < 4.0) {
        return;
    }
    std::vector<std::pair<u64, void*>> sorted;
    for (const auto& [address, count] : counts) {
        sorted.emplace_back(count, address);
    }
    std::sort(sorted.begin(), sorted.end(), std::greater{});
    for (size_t i = 0; i < sorted.size() && i < 12; ++i) {
        LOG_ERROR(Render_Vulkan, "RenderPassBreak window={:.1f}s n={} caller_rva={:#x}", secs,
                  sorted[i].first, CallerOffset(sorted[i].second));
    }
    counts.clear();
    window_start = now;
}
} // namespace

const bool Scheduler::thread_check_enabled = std::getenv("EDEN_SCHEDULER_THREAD_CHECK") != nullptr;

void Scheduler::CheckRecordThread(const char* kind, void* caller) {
    // EDEN_SCHEDULER_THREAD_CHECK: the scheduler is single-producer. Log whenever the producing
    // thread changes, with the caller of the call that switched threads.
    if (!caller) {
        // Called from the inlined Record template: our return address is in its caller.
        caller = EDEN_RETURN_ADDRESS();
    }
    static std::atomic<u32> lines{0};
    const std::thread::id self = std::this_thread::get_id();
    const std::thread::id previous = last_record_thread.exchange(self, std::memory_order_relaxed);
    if (previous == self || lines.fetch_add(1, std::memory_order_relaxed) >= 400) {
        return;
    }
    LOG_ERROR(Render_Vulkan, "SchedulerThread {} thread={} previous={} caller_rva={:#x}", kind,
              std::hash<std::thread::id>{}(self), std::hash<std::thread::id>{}(previous),
              CallerOffset(caller));
}

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (VideoCommon::GpuSplit::check_owner) {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Flush");
    }
    NoteSubmit("Flush", EDEN_RETURN_ADDRESS());
    if (thread_check_enabled) [[unlikely]] {
        CheckRecordThread("Flush", EDEN_RETURN_ADDRESS());
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);
    AllocateNewContext();
    return signal_value;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (VideoCommon::GpuSplit::check_owner) {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Finish");
    }
    NoteSubmit("Finish", EDEN_RETURN_ADDRESS());
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitExecution(signal_semaphore, wait_semaphore);
    Wait(presubmit_tick);
    AllocateNewContext();
}

void Scheduler::WaitWorker() {
    Common::FramePacing::Mark("worker_wait_b", 0, reinterpret_cast<u64>(EDEN_RETURN_ADDRESS()));
    SCOPE_EXIT { Common::FramePacing::Mark("worker_wait_e"); };
    DispatchWork();

    // Ensure the queue is drained.
    Common::Timeline::Mark("worker_wait_b", 0, reinterpret_cast<u64>(EDEN_RETURN_ADDRESS()));
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
    Common::Timeline::Mark("worker_wait_e");
}

void Scheduler::DispatchWork() {
    if (VideoCommon::GpuSplit::check_owner) {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::DispatchWork");
    }
    if (chunk && !chunk->Empty()) {
        bool wake;
        {
            std::scoped_lock ql{queue_mutex};
            work_queue.push(std::move(chunk));
            wake = worker_waiting;
        }
        if (wake || !VideoCommon::NativeDrawOptimizations::batch_vulkan_work.load(std::memory_order_relaxed)) {
            event_cv.notify_all();
        }
        AcquireNewChunk();
    }
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count) {
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    ++timing_renderpass_begins;
    state.renderpass = renderpass;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    Record([renderpass, framebuffer_handle, render_area, values, clear_value_count](
               vk::CommandBuffer cmdbuf) {
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count);
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    RealizeDeferredClear();
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer) {
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    if (renderpass == state.renderpass && framebuffer_handle == state.framebuffer &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        return;
    }
    // Ends any active pass and realizes a deferred clear
    EndRenderPass();
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestOutsideRenderPassOperationContext() {
    if (gpu_timer && state.renderpass) [[unlikely]] {
        ++timing_renderpass_breaks;
        NoteRenderPassBreak(EDEN_RETURN_ADDRESS());
    }
    EndRenderPass();
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            event_cv.wait(lk, stop_token, [&] {
                worker_waiting = !TryPopQueue(work);
                return !worker_waiting;
            });
            worker_waiting = false;

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            void(std::exchange(lk, std::unique_lock{execution_mutex}));

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf);

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffer();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffer() {
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    if (gpu_timer) {
        gpu_timer->Begin(current_upload_cmdbuf);
    }
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    EndPendingOperations();
    InvalidateState();

    const u64 signal_value = master_semaphore->NextTick();
    const u32 rp_begins = std::exchange(timing_renderpass_begins, 0);
    const u32 rp_breaks = std::exchange(timing_renderpass_breaks, 0);
    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value, rp_begins, rp_breaks,
                            this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.End();
        if (gpu_timer) {
            gpu_timer->End(cmdbuf);
        }
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        std::scoped_lock lock{submit_mutex};
        switch (const VkResult result = master_semaphore->SubmitQueue(
                    cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore, signal_value)) {
        case VK_SUCCESS:
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            if (gpu_timer) {
                gpu_timer->Submitted(signal_value, rp_begins, rp_breaks);
                gpu_timer->Collect(*master_semaphore);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
    });
    chunk->MarkSubmit();
    DispatchWork();
    return signal_value;
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations() {
    query_cache->CounterReset(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass();
}

void Scheduler::EndRenderPass()
    {
        RealizeDeferredClear();
        if (!state.renderpass) {
            return;
        }

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);

        Record([num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback = device.IsExtTransformFeedbackSupported()](
                          vk::CommandBuffer cmdbuf) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        });

        state.renderpass = VkRenderPass{};
        num_renderpass_images = 0;
    }


void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
