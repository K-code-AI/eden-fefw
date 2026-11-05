#include "common/frame_pacing_trace.h"
#include "video_core/native_readback_trace.h"
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/ampere/program_capture.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <string>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/draw_sampler_cache.h"
#include "video_core/renderer_vulkan/texture_handle_span.h"

#include "common/bit_field.h"
#include "common/scope_exit.h"
#include "shader_recompiler/exception.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_notify.h"
#include "video_core/native_vprs.h"
#include "video_core/native_draw_optimizations.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/settings.h"

#if defined(_MSC_VER) && defined(NDEBUG)
#define LAMBDA_FORCEINLINE [[msvc::forceinline]]
#else
#define LAMBDA_FORCEINLINE
#endif

namespace Vulkan {
namespace {
using boost::container::small_vector;
using boost::container::static_vector;
using Shader::ImageBufferDescriptor;
using Shader::Backend::SPIRV::RENDERAREA_LAYOUT_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_DOWN_FACTOR_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_WORDS_OFFSET;
using Tegra::Texture::TexturePair;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;

constexpr size_t NUM_STAGES = Maxwell::MaxShaderStage;

struct DecodedTextureRecord {
    std::pair<u32, u32> handle;
    bool excluded{};
};

struct DecodedTextureArray {
    Shader::BufferTextureSource source{};
    u32 count{};
    bool via_header_index{}, valid{};
    std::vector<u8> bytes;
    std::vector<DecodedTextureRecord> records;
    u32 max_image{}, max_sampler{};
    u32 misses{}, cooldown{};
};

class TextureRecordCache {
public:
    const DecodedTextureArray* Resolve(const Shader::TextureDescriptor& desc,
                                      std::span<const u8> bytes, bool via_header_index) {
        if (desc.count < 64 || desc.count > 4096 || bytes.size() > 256 * 1024) return nullptr;
        auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
            return entry.valid && entry.source == desc.buffer_source && entry.count == desc.count &&
                   entry.via_header_index == via_header_index;
        });
        if (found == entries.end()) {
            found = entries.begin() + next_entry;
            next_entry = (next_entry + 1) % entries.size();
            found->misses = found->cooldown = 0;
        } else if (found->cooldown) {
            --found->cooldown;
            return nullptr;
        } else if (found->bytes.size() == bytes.size() &&
                   std::equal(bytes.begin(), bytes.end(), found->bytes.begin())) {
            found->misses = 0;
            return &*found;
        } else if (++found->misses >= 4) {
            // Frequently changing records use the direct decoder without copying a cache snapshot.
            found->misses = 0;
            found->cooldown = 64;
            return nullptr;
        }
        auto& entry = *found;
        entry.valid = false;
        entry.source = desc.buffer_source;
        entry.count = desc.count;
        entry.via_header_index = via_header_index;
        entry.max_image = entry.max_sampler = 0;
        entry.records.resize(desc.count);
        const auto word = [&](u64 offset) {
            if (offset > bytes.size() || bytes.size() - offset < sizeof(u32))
                throw Shader::RuntimeError("Texture-handle record outside captured buffer");
            u32 value;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        };
        for (u32 index = 0; index < desc.count; ++index) {
            auto& record = entry.records[index];
            record.handle = TexturePair(word(u64{index} * entry.source.stride + entry.source.offset),
                                        via_header_index);
            entry.max_image = (std::max)(entry.max_image, record.handle.first);
            entry.max_sampler = (std::max)(entry.max_sampler, record.handle.second);
            record.excluded = false;
            for (const auto& exclusion : entry.source.exclusions) {
                if (exclusion.mask && exclusion.Matches(word(u64{index} * entry.source.stride + exclusion.offset))) {
                    record.excluded = true;
                    break;
                }
            }
        }
        entry.bytes.assign(bytes.begin(), bytes.end());
        entry.valid = true;
        return &entry;
    }
private:
    std::array<DecodedTextureArray, 4> entries;
    size_t next_entry{};
};

struct DrawBindingScratch {
    small_vector<VideoCommon::ImageViewInOut, 64> views;
    small_vector<size_t, 64> excluded_views;
    small_vector<VideoCommon::SamplerId, 64> samplers;
    std::vector<u8> texture_bytes;
    TextureRecordCache texture_records;
    DrawSamplerCache<VideoCommon::SamplerId> sampler_ids;
    bool in_use{};
};

// Temporary local profiling: enabled only while the request file exists.
class DrawTiming {
    using Clock = std::chrono::steady_clock;
    struct Totals {
        std::array<u64, 8> ns{};
        u64 draws{}, probes{};
        bool enabled{};
        bool direct_handles{true};
        bool batch_image_views{};
        bool fast_invalid_views{};
        bool batch_invalid_views{};
        bool fast_invalid_samplers{};
        Clock::time_point started{};
        Clock::time_point next_control_poll{};
        Clock::time_point next_mode_poll{};
        bool legacy_control_poll{};
    };
    static Totals& Shared() {
        static thread_local Totals totals;
        return totals;
    }
public:
    explicit DrawTiming(bool native) : totals{Shared()} {
        if ((++totals.probes & 4095) == 0) {
            const auto now = Clock::now();
            if (now >= totals.next_mode_poll) {
                std::error_code error;
                totals.legacy_control_poll =
                    std::filesystem::exists("native-control-poll-legacy.request", error);
                totals.next_mode_poll = now + std::chrono::seconds{1};
            }
            if (totals.legacy_control_poll || now >= totals.next_control_poll) {
                totals.next_control_poll = now + std::chrono::seconds{1};
                Common::FramePacing::Mark("control_poll_b");
                std::error_code error;
                const bool requested = std::filesystem::exists("native-render-timing.request", error);
                if (requested && !totals.enabled) {
                    totals.ns = {};
                    totals.draws = 0;
                    totals.started = Clock::now();
                }
                totals.enabled = requested;
                totals.direct_handles = !std::filesystem::exists("native-handle-span-disable.request", error);
                totals.batch_image_views = native &&
                    !std::filesystem::exists("native-view-runs-disable.request", error);
                totals.fast_invalid_views = native &&
                    !std::filesystem::exists("native-invalid-view-disable.request", error);
                totals.batch_invalid_views = native &&
                    !std::filesystem::exists("native-invalid-runs-disable.request", error);
                totals.fast_invalid_samplers = native &&
                    !std::filesystem::exists("native-sampler-bounds-disable.request", error);
                VideoCommon::NativeDrawOptimizations::skip_unchanged_state.store(
                    !std::filesystem::exists("native-state-write-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::reuse_usage_words.store(
                    !std::filesystem::exists("native-usage-reuse-disable.request", error), std::memory_order_relaxed);
                u32 spins = native ? 1024 : 0;
                if (native) {
                    std::ifstream request("native-cache-spin.request");
                    u32 requested_spins;
                    if (request >> requested_spins) spins = std::min(requested_spins, 4096U);
                }
                VideoCommon::NativeDrawOptimizations::cache_lock_spins.store(spins, std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::compact_fence_flushes.store(
                    native && !std::filesystem::exists("native-fence-flush-disable.request", error),
                    std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::reuse_sparse_uniforms.store(
                    !std::filesystem::exists("native-sparse-uniform-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::batch_texture_handle_runs.store(
                    !std::filesystem::exists("native-handle-runs-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::vector_texture_handle_scan.store(
                    native && !std::filesystem::exists("native-vector-handles-disable.request", error),
                    std::memory_order_relaxed);
                {
                    namespace Optimizations = VideoCommon::NativeDrawOptimizations;
                    Optimizations::tight_texture_loops.store(
                        !std::filesystem::exists("native-texture-loops-disable.request", error),
                        std::memory_order_relaxed);
                    const bool check = std::filesystem::exists("native-texture-loops-check.request", error);
                    Optimizations::check_texture_loops.store(check, std::memory_order_relaxed);
                    if (check) {
                        std::ofstream file("native-texture-loops-check.csv", std::ios::app);
                        file << std::chrono::duration<double>(now.time_since_epoch()).count() << ','
                             << Optimizations::texture_array_checks.load(std::memory_order_relaxed) << ','
                             << Optimizations::texture_array_mismatches.load(std::memory_order_relaxed) << ','
                             << Optimizations::texture_array_unstable.load(std::memory_order_relaxed)
                             << '\n';
                    }
                    Optimizations::thread_contiguous_runs.store(
                        !std::filesystem::exists("native-thread-runs-disable.request", error),
                        std::memory_order_relaxed);
                    Optimizations::skip_empty_rejections.store(
                        !std::filesystem::exists("native-rejection-flag-disable.request", error),
                        std::memory_order_relaxed);
                    const bool check_locks =
                        std::filesystem::exists("native-uniform-locks-check.request", error);
                    Optimizations::check_uniform_locks.store(check_locks, std::memory_order_relaxed);
                    if (check_locks) {
                        std::ofstream file("native-uniform-locks-check.csv", std::ios::app);
                        file << std::chrono::duration<double>(now.time_since_epoch()).count() << ','
                             << Optimizations::contiguous_checks.load(std::memory_order_relaxed) << ','
                             << Optimizations::contiguous_mismatches.load(std::memory_order_relaxed) << ','
                             << Optimizations::contiguous_unstable.load(std::memory_order_relaxed) << ','
                             << Optimizations::rejection_checks.load(std::memory_order_relaxed) << ','
                             << Optimizations::rejection_mismatches.load(std::memory_order_relaxed)
                             << '\n';
                    }
                }
                VideoCommon::NativeDrawOptimizations::batch_native_registers.store(
                    !std::filesystem::exists("native-split-control-sync.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::defer_front_verification.store(
                    !std::filesystem::exists("native-front-sync-verify.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::front_verify_interval.store(
                    std::filesystem::exists("native-front-verify.request", error) ? 1U : 4096U, std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::park_front_waits.store(
                    !std::filesystem::exists("native-front-wait-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::gpu_split_enabled.store(
                    !std::filesystem::exists("native-gpu-split-disable.request", error), std::memory_order_relaxed);
                const bool reduce_cpu_work = !std::filesystem::exists("native-cpu-overhead-disable.request", error);
                VideoCommon::NativeDrawOptimizations::fast_gpu_range_queries.store(
                    reduce_cpu_work && !std::filesystem::exists("native-range-query-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::short_macro_parameters.store(
                    reduce_cpu_work && !std::filesystem::exists("native-short-macro-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::guard_macro_debug.store(
                    reduce_cpu_work && !std::filesystem::exists("native-macro-debug-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::batch_vulkan_work.store(
                    reduce_cpu_work && !std::filesystem::exists("native-worker-batch-disable.request", error), std::memory_order_relaxed);
                VideoCommon::NativeDrawOptimizations::optimize_macro_tail.store(
                    !std::filesystem::exists("native-macro-tail-disable.request", error), std::memory_order_relaxed);
                const bool batch_shaders =
                    !std::filesystem::exists("native-shader-batch-disable.request", error);
                if (VideoCommon::NativeDrawOptimizations::batch_shader_invalidation.exchange(
                        batch_shaders, std::memory_order_relaxed) != batch_shaders) {
                    LOG_INFO(Render_Vulkan, "Shader invalidation batching: {}", batch_shaders);
                }
                const bool flat_samplers =
                    !std::filesystem::exists("native-flat-sampler-disable.request", error);
                if (VideoCommon::NativeDrawOptimizations::flat_sampler_lookup.exchange(
                        flat_samplers, std::memory_order_relaxed) != flat_samplers) {
                    LOG_INFO(Render_Vulkan, "Flat sampler lookup: {}", flat_samplers);
                }
                Common::FramePacing::Mark("control_poll_e");
            }
        }
        enabled = totals.enabled;
        if (enabled) last = Clock::now();
    }
    bool DirectHandles() const { return totals.direct_handles; }
    bool BatchImageViews() const { return totals.batch_image_views; }
    bool FastInvalidViews() const { return totals.fast_invalid_views; }
    bool BatchInvalidViews() const { return totals.batch_invalid_views; }
    bool FastInvalidSamplers() const { return totals.fast_invalid_samplers; }
    void Mark(size_t part) {
        if (!enabled) return;
        const auto now = Clock::now();
        totals.ns[part] += std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count();
        last = now;
    }
    ~DrawTiming() {
        if (!enabled) return;
        ++totals.draws;
        const auto now = Clock::now();
        if (now - totals.started < std::chrono::seconds{1}) return;
        std::ofstream file("native-render-timing.csv", std::ios::app);
        file << std::chrono::duration<double>(now.time_since_epoch()).count() << ','
             << std::chrono::duration<double>(now - totals.started).count() << ',' << totals.draws;
        for (u64 elapsed : totals.ns) file << ',' << elapsed;
        file << '\n';
        totals.ns = {};
        totals.draws = 0;
        totals.started = now;
    }
private:
    Totals& totals;
    bool enabled{};
    Clock::time_point last{};
};

u64 TextureBufferReadSize(const Shader::TextureDescriptor& desc) {
    const auto& source = desc.buffer_source;
    u64 record_end = u64{source.offset} + sizeof(u32);
    for (const auto& exclusion : source.exclusions) {
        if (!exclusion.mask) continue;
        if (exclusion.offset > source.stride || source.stride - exclusion.offset < sizeof(u32)) {
            throw Shader::RuntimeError("Invalid buffer texture exclusion field");
        }
        record_end = (std::max)(record_end, u64{exclusion.offset} + sizeof(u32));
    }
    const u64 size = desc.count ? u64{desc.count - 1} * source.stride + record_end : 0;
    if (size > source.size) {
        throw Shader::RuntimeError("Texture-handle records outside buffer");
    }
    return size;
}

// Same views, samplers and sampler lookups as filling each run of equal handles, in one pass.
// Call sampler_ids.BeginArray() first.
template <typename Read, typename Resolve>
void FillDirectTextures(u32 count, bool via_header_index, Read read,
                        DrawSamplerCache<VideoCommon::SamplerId>& sampler_ids, Resolve& resolve,
                        VideoCommon::ImageViewInOut* views, VideoCommon::SamplerId* samplers) {
    const u32 skip_above = sampler_ids.SkipAbove();
    const VideoCommon::SamplerId invalid_sampler = sampler_ids.InvalidValue();
    u32 last_sampler_index = ~u32{0};
    VideoCommon::SamplerId last_sampler_id = VideoCommon::NULL_SAMPLER_ID;
    for (u32 index = 0; index < count; ++index) {
        const auto handle = TexturePair(read(index), via_header_index);
        auto& view = views[index];
        view.index = handle.first;
        view.blacklist = false;
        view.id = {};
        if (handle.second != last_sampler_index) {
            last_sampler_index = handle.second;
            last_sampler_id = handle.second > skip_above ? invalid_sampler
                                                         : sampler_ids.Get(handle.second, resolve);
        }
        samplers[index] = last_sampler_id;
    }
}

template <typename Resolve>
void FillDirectTextures(const TextureHandleSpan& handles, u32 count, bool via_header_index,
                        DrawSamplerCache<VideoCommon::SamplerId>& sampler_ids, Resolve& resolve,
                        VideoCommon::ImageViewInOut* views, VideoCommon::SamplerId* samplers) {
    if (const u8* const words = handles.PackedWords()) {
        FillDirectTextures(count, via_header_index, [words](u32 index) LAMBDA_FORCEINLINE {
            u32 raw;
            std::memcpy(&raw, words + size_t{index} * sizeof(raw), sizeof(raw));
            return raw;
        }, sampler_ids, resolve, views, samplers);
    } else {
        FillDirectTextures(count, via_header_index, [&handles](u32 index) LAMBDA_FORCEINLINE {
            return handles.Read(index);
        }, sampler_ids, resolve, views, samplers);
    }
}

// Fills the array as above, then recomputes it with the run-filling loop and counts differences.
// Handles and sampler entries are live guest memory: the second pass reuses the first pass's
// sampler ids, and a handle change between the passes is counted as unstable, not as a difference.
template <typename Resolve>
void FillDirectTexturesChecked(const TextureHandleSpan& handles, u32 count, bool via_header_index,
                               DrawSamplerCache<VideoCommon::SamplerId>& sampler_ids,
                               Resolve& resolve, VideoCommon::ImageViewInOut* views,
                               VideoCommon::SamplerId* samplers, bool vector_scan) {
    namespace Optimizations = VideoCommon::NativeDrawOptimizations;
    static thread_local std::array<std::vector<u32>, 2> lookups;
    static thread_local std::vector<VideoCommon::SamplerId> resolved;
    static thread_local std::array<std::vector<u32>, 2> words;
    static thread_local std::vector<VideoCommon::ImageViewInOut> expected_views;
    static thread_local std::vector<VideoCommon::SamplerId> expected_samplers;
    const auto snapshot = [&](std::vector<u32>& out) {
        out.resize(count);
        for (u32 index = 0; index < count; ++index) out[index] = handles.Read(index);
    };
    lookups[0].clear();
    lookups[1].clear();
    resolved.clear();
    snapshot(words[0]);
    auto record_resolve = [&](u32 sampler) {
        lookups[0].push_back(sampler);
        return resolved.emplace_back(resolve(sampler));
    };
    FillDirectTextures(handles, count, via_header_index, sampler_ids, record_resolve, views, samplers);

    expected_views.assign(count, {});
    expected_samplers.assign(count, {});
    sampler_ids.BeginArray();
    bool sampler_changed = false;
    u32 last_sampler_index = ~u32{0};
    VideoCommon::SamplerId last_sampler_id = VideoCommon::NULL_SAMPLER_ID;
    handles.ForEachRun(count, [&](u32 start, u32 length, u32 raw) {
        const auto handle = TexturePair(raw, via_header_index);
        if (handle.second != last_sampler_index) {
            last_sampler_index = handle.second;
            last_sampler_id = sampler_ids.Get(handle.second, [&](u32 sampler) {
                const size_t lookup = lookups[1].size();
                lookups[1].push_back(sampler);
                VideoCommon::SamplerId id = resolve(sampler);
                if (lookup < resolved.size() && lookups[0][lookup] == sampler) {
                    sampler_changed |= id != resolved[lookup];
                    id = resolved[lookup];
                }
                return id;
            });
        }
        std::fill_n(expected_views.data() + start, length, VideoCommon::ImageViewInOut{handle.first});
        std::fill_n(expected_samplers.data() + start, length, last_sampler_id);
    }, vector_scan);
    snapshot(words[1]);

    u32 difference = count;
    for (u32 index = 0; index < count; ++index) {
        if (views[index].index != expected_views[index].index ||
            views[index].blacklist != expected_views[index].blacklist ||
            views[index].id != expected_views[index].id || samplers[index] != expected_samplers[index]) {
            difference = index;
            break;
        }
    }
    Optimizations::texture_array_checks.fetch_add(1, std::memory_order_relaxed);
    const bool differs = lookups[0] != lookups[1] || difference != count;
    if (!differs && !sampler_changed) return;
    if (differs && words[0] == words[1]) {
        if (Optimizations::texture_array_mismatches.fetch_add(1, std::memory_order_relaxed) < 16) {
            const u32 at = (std::min)(difference, count - 1);
            LOG_ERROR(Render_Vulkan,
                      "Texture array loop mismatch: {} handles, {} vs {} sampler lookups, element {} "
                      "raw {:#x}, view {} vs {}, sampler {} vs {}",
                      count, lookups[0].size(), lookups[1].size(), difference, words[0][at],
                      views[at].index, expected_views[at].index, samplers[at].index,
                      expected_samplers[at].index);
        }
    } else if (Optimizations::texture_array_unstable.fetch_add(1, std::memory_order_relaxed) < 16) {
        u32 changed = 0;
        while (changed < count && words[0][changed] == words[1][changed]) ++changed;
        LOG_WARNING(Render_Vulkan,
                    "Texture array check unstable: {} handles, first changed handle {}, sampler "
                    "changed {}, loops differ {}",
                    count, changed, sampler_changed, differs);
    }
}

DescriptorLayoutBuilder MakeBuilder(const Device& device, std::span<const Shader::Info> infos) {
    DescriptorLayoutBuilder builder{device};
    for (size_t index = 0; index < infos.size(); ++index) {
        static constexpr std::array stages{
            VK_SHADER_STAGE_VERTEX_BIT,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            VK_SHADER_STAGE_GEOMETRY_BIT,
            VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        builder.Add(infos[index], stages.at(index));
    }
    return builder;
}

template <class StencilFace>
VkStencilOpState GetStencilFaceState(const StencilFace& face) {
    return {
        .failOp = MaxwellToVK::StencilOp(face.ActionStencilFail()),
        .passOp = MaxwellToVK::StencilOp(face.ActionDepthPass()),
        .depthFailOp = MaxwellToVK::StencilOp(face.ActionDepthFail()),
        .compareOp = MaxwellToVK::ComparisonOp(face.TestFunc()),
        .compareMask = 0,
        .writeMask = 0,
        .reference = 0,
    };
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    static constexpr std::array unsupported_topologies{
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
        // VK_PRIMITIVE_TOPOLOGY_QUAD_LIST_EXT,
    };
    return std::ranges::find(unsupported_topologies, topology) == unsupported_topologies.end();
}

bool IsLine(VkPrimitiveTopology topology) {
    static constexpr std::array line_topologies{
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
        // VK_PRIMITIVE_TOPOLOGY_LINE_LOOP_EXT,
    };
    return std::ranges::find(line_topologies, topology) != line_topologies.end();
}

VkViewportSwizzleNV UnpackViewportSwizzle(u16 swizzle) {
    union Swizzle {
        u32 raw;
        BitField<0, 3, Maxwell::ViewportSwizzle> x;
        BitField<4, 3, Maxwell::ViewportSwizzle> y;
        BitField<8, 3, Maxwell::ViewportSwizzle> z;
        BitField<12, 3, Maxwell::ViewportSwizzle> w;
    };
    const Swizzle unpacked{swizzle};
    return VkViewportSwizzleNV{
        .x = MaxwellToVK::ViewportSwizzle(unpacked.x),
        .y = MaxwellToVK::ViewportSwizzle(unpacked.y),
        .z = MaxwellToVK::ViewportSwizzle(unpacked.z),
        .w = MaxwellToVK::ViewportSwizzle(unpacked.w),
    };
}

PixelFormat DecodeFormat(u8 encoded_format) {
    const auto format{static_cast<Tegra::RenderTargetFormat>(encoded_format)};
    if (format == Tegra::RenderTargetFormat::NONE) {
        return PixelFormat::Invalid;
    }
    return PixelFormatFromRenderTargetFormat(format);
}

RenderPassKey MakeRenderPassKey(const FixedPipelineState& state, const Device& device) {
    RenderPassKey key{};
    std::ranges::transform(state.color_formats, key.color_formats.begin(), DecodeFormat);
    if (state.depth_enabled != 0) {
        const auto depth_format{static_cast<Tegra::DepthFormat>(state.depth_format.Value())};
        key.depth_format = PixelFormatFromDepthFormat(depth_format);
    } else {
        key.depth_format = PixelFormat::Invalid;
    }
    key.samples = MaxwellToVK::MsaaMode(state.msaa_mode);
    const bool has_color = std::ranges::any_of(key.color_formats, [](PixelFormat format) {
        return format != PixelFormat::Invalid;
    });
    key.resolve_color =
        key.samples != VK_SAMPLE_COUNT_1_BIT && has_color && device.IsTiler();
    return key;
}

size_t NumAttachments(const FixedPipelineState& state) {
    size_t num{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; ++index) {
        const auto format{static_cast<Tegra::RenderTargetFormat>(state.color_formats[index])};
        if (format != Tegra::RenderTargetFormat::NONE) {
            num = index + 1;
        }
    }
    return num;
}

template <typename Spec>
bool Passes(const std::array<vk::ShaderModule, NUM_STAGES>& modules,
            const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!Spec::enabled_stages[stage] && modules[stage]) {
            return false;
        }
        const auto& info{stage_infos[stage]};
        if constexpr (!Spec::has_storage_buffers) {
            if (!info.storage_buffers_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_texture_buffers) {
            if (!info.texture_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_image_buffers) {
            if (!info.image_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_images) {
            if (!info.image_descriptors.empty()) {
                return false;
            }
        }
    }
    return true;
}

using ConfigureFuncPtr = bool (*)(GraphicsPipeline*, bool);

template <typename Spec, typename... Specs>
ConfigureFuncPtr FindSpec(const std::array<vk::ShaderModule, NUM_STAGES>& modules,
                          const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    if constexpr (sizeof...(Specs) > 0) {
        if (!Passes<Spec>(modules, stage_infos)) {
            return FindSpec<Specs...>(modules, stage_infos);
        }
    }
    return GraphicsPipeline::MakeConfigureSpecFunc<Spec>();
}

struct SimpleVertexFragmentSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleVertexSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, false};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleStorageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleImageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = true;
};

struct DefaultSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, true, true, true, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = true;
    static constexpr bool has_image_buffers = true;
    static constexpr bool has_images = true;
};

ConfigureFuncPtr ConfigureFunc(const std::array<vk::ShaderModule, NUM_STAGES>& modules,
                               const std::array<Shader::Info, NUM_STAGES>& infos) {
    return FindSpec<SimpleVertexSpec, SimpleVertexFragmentSpec, SimpleStorageSpec, SimpleImageSpec,
                    DefaultSpec>(modules, infos);
}

// native-draw-ps.request = <pixel shader address> [rt0 address] (hex): true for the next such draw,
// whose state ConfigureImpl then dumps. Shader addresses move between runs, so a value below
// 0x10000 is the pixel shader's offset from the vertex shader instead. Shared by every
// ConfigureImpl instantiation.
bool WantDrawCapture(const Tegra::Engines::Maxwell3D::Regs& regs) {
    static u64 wanted_ps{};
    static u64 wanted_rt0{};
    std::ifstream request;
    if (VideoCommon::NativeRequestPending()) request.open("native-draw-ps.request");
    if (request.is_open()) {
        wanted_rt0 = 0;
        request >> std::hex >> wanted_ps >> wanted_rt0;
        request.close();
        std::error_code error;
        std::filesystem::remove("native-draw-ps.request", error);
    }
    const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    const bool ps_match = wanted_ps < 0x10000 ? ps - vs == wanted_ps : ps == wanted_ps;
    if (wanted_ps == 0 || !ps_match || (wanted_rt0 != 0 && regs.rt[0].Address() != wanted_rt0)) {
        return false;
    }
    wanted_ps = 0;
    LOG_ERROR(Render_Vulkan, "Draw capture for ps={:x} rt0={:x}", ps, regs.rt[0].Address());
    return true;
}
} // Anonymous namespace

// TODO(crueter): This is the worst-formatted code I have EVER seen
GraphicsPipeline::GraphicsPipeline(
    Scheduler& scheduler_, BufferCache& buffer_cache_, TextureCache& texture_cache_,
    vk::PipelineCache& pipeline_cache_, VideoCore::ShaderNotify* shader_notify,
    const Device& device_, DescriptorPool& descriptor_pool,
    GuestDescriptorQueue& guest_descriptor_queue_, DescriptorBufferRing& descriptor_buffer_ring_,
    Common::ThreadWorker* worker_thread,
    PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
    const GraphicsPipelineCacheKey& key_, std::array<vk::ShaderModule, NUM_STAGES> stages,
    const std::array<const Shader::Info*, NUM_STAGES>& infos)
    : key{key_}, device{device_}, texture_cache{texture_cache_}, buffer_cache{buffer_cache_},
      pipeline_cache(pipeline_cache_), scheduler{scheduler_},
      guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, spv_modules{std::move(stages)} {
    if (shader_notify) {
        shader_notify->MarkShaderBuilding();
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info* const info{infos[stage]};
        if (!info) {
            continue;
        }
        stage_infos[stage] = *info;
        enabled_uniform_buffer_masks[stage] = info->constant_buffer_mask;
        std::ranges::copy(info->constant_buffer_used_sizes, uniform_buffer_sizes[stage].begin());
        num_image_elements += Shader::NumDescriptors(info->texture_buffer_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_buffer_descriptors);
        num_textures += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_descriptors);
        num_descriptor_entries += NumDescriptorEntries(*info);
    }
    fragment_has_color0_output = stage_infos[NUM_STAGES - 1].stores_frag_color[0];
    if (infos[NUM_STAGES - 1]) {
        const auto& frag{stage_infos[NUM_STAGES - 1]};
        fragment_store_mask = 0;
        for (size_t rt = 0; rt < frag.stores_frag_color.size(); ++rt) {
            fragment_store_mask |= frag.stores_frag_color[rt] ? 1u << rt : 0u;
        }
        if (VideoCommon::NativeDrawDiagnosticsEnabled()) {
            LOG_ERROR(Render_Vulkan,
                      "FragStores pipeline={:016X} rt_store_mask={:02X} depth={} vs={:016x} fs={:016x}",
                      key.Hash(), fragment_store_mask, frag.stores_frag_depth, key.unique_hashes[1],
                      key.unique_hashes[5]);
        }
    }

    DescriptorLayoutBuilder builder{MakeBuilder(device, stage_infos)};
    uses_push_descriptor = builder.CanUsePushDescriptor();
    uses_descriptor_buffer = builder.CanUseDescriptorBuffer() && descriptor_buffer_ring.IsValid();
    descriptor_set_layout =
        builder.CreateDescriptorSetLayout(uses_push_descriptor, uses_descriptor_buffer);
    if (uses_descriptor_buffer) {
        descriptor_buffer_layout = builder.MakeDescriptorBufferLayout(*descriptor_set_layout);
        if (!descriptor_buffer_ring.CanAllocate(descriptor_buffer_layout.size)) {
            LOG_WARNING(Render_Vulkan,
                        "Graphics pipeline {:016X} needs {} descriptor bytes per draw, falling back "
                        "to sets",
                        key.Hash(), descriptor_buffer_layout.size);
            uses_descriptor_buffer = false;
            descriptor_buffer_layout = {};
            descriptor_set_layout = builder.CreateDescriptorSetLayout(uses_push_descriptor);
        }
    }

    const VkDescriptorSetLayout set_layout{*descriptor_set_layout};
    pipeline_layout = builder.CreatePipelineLayout(set_layout);
    if (!uses_descriptor_buffer) {
        descriptor_update_template =
            builder.CreateTemplate(set_layout, *pipeline_layout, uses_push_descriptor);
        if (!uses_push_descriptor) {
            descriptor_allocator =
                descriptor_pool.Allocator(device, scheduler, set_layout, stage_infos);
        }
    }

    auto func{[this, shader_notify, &render_pass_cache, pipeline_statistics] {
        const VkRenderPass render_pass{render_pass_cache.Get(MakeRenderPassKey(key.state, device))};
        Validate();
        try {
            MakePipeline(render_pass);
        } catch (const vk::Exception& exception) {
            LOG_CRITICAL(Render_Vulkan, "Graphics pipeline build failed: {}", exception.what());
            std::scoped_lock lock{build_mutex};
            is_built = true;
            build_condvar.notify_one();
            if (shader_notify) {
                shader_notify->MarkShaderComplete();
            }
            return;
        }
        if (pipeline_statistics) {
            pipeline_statistics->Collect(device, *pipeline);
        }

        std::scoped_lock lock{build_mutex};
        is_built = true;
        build_condvar.notify_one();
        if (shader_notify) {
            shader_notify->MarkShaderComplete();
        }
    }};
    if (worker_thread) {
        worker_thread->QueueWork(std::move(func));
    } else {
        func();
    }
    configure_func = ConfigureFunc(spv_modules, stage_infos);
}

void GraphicsPipeline::AddTransition(GraphicsPipeline* transition) {
    transition_keys.push_back(transition->key);
    transitions.push_back(transition);
}

template <typename Spec>
bool GraphicsPipeline::ConfigureImpl(bool is_indexed) {
    DrawTiming draw_timing{maxwell3d->GetNativeMacroEngine().native_turing};
    if (std::any_of(stage_infos.begin(), stage_infos.end(), [](const auto& info) {
            return info.stores[Shader::IR::Attribute::VprsTableIndex];
        })) {
        if (!maxwell3d->GetNativeMacroEngine().native_turing)
            throw Shader::RuntimeError("VPRS table-index output requires native graphics state");
        std::array<u32, 16> controls{}, rates_a{}, rates_b{};
        for (u32 table = 0; table < controls.size(); ++table) {
            const u32 base = 0x2a00 / 4 + table * 4;
            controls[table] = maxwell3d->regs.reg_array[base];
            rates_a[table] = maxwell3d->regs.reg_array[base + 1];
            rates_b[table] = maxwell3d->regs.reg_array[base + 2];
        }
        const auto state = Tegra::ClassifyNativeVprs(controls, rates_a, rates_b);
        const auto& fragment = stage_infos.back();
        const auto writable = [](const auto& descriptors) {
            return std::any_of(descriptors.begin(), descriptors.end(), [](const auto& desc) { return desc.is_written; });
        };
        const bool writes = fragment.stores_global_memory || writable(fragment.storage_buffers_descriptors) ||
                            writable(fragment.image_descriptors) || writable(fragment.image_buffer_descriptors);
        if (!state.AllowsFullRate(writes))
            throw Shader::RuntimeError("Native VPRS full-rate fallback is unsafe: enabled={:#x}, coarse={:#x}, unsupported={:#x}, fragment_writes={}",
                                       state.enabled, state.coarse, state.unsupported, writes);
        static thread_local u32 traces{};
        if (VideoCommon::NativeDrawDiagnosticsEnabled() && traces++ < 8)
            LOG_INFO(Render_Vulkan, "Native VPRS uses full pixel rate: enabled_tables={:#x}, coarse_tables={:#x}, select={:#x}",
                     state.enabled, state.coarse, maxwell3d->regs.reg_array[0x2cc / 4]);
    }

    // Retain allocation capacity, but rebuild every binding on every draw. A
    // nested preparation uses its own storage until the outer draw completes.
    static thread_local DrawBindingScratch shared_scratch;
    std::optional<DrawBindingScratch> nested_scratch;
    auto& scratch = shared_scratch.in_use ? nested_scratch.emplace() : shared_scratch;
    scratch.in_use = true;
    SCOPE_EXIT { scratch.in_use = false; };
    auto& views = scratch.views;
    auto& excluded_views = scratch.excluded_views;
    auto& samplers = scratch.samplers;
    excluded_views.clear();
    // The immutable pipeline metadata already gives the exact array sizes.
    // Fill slots directly instead of growing two vectors for every binding.
    views.resize(num_image_elements);
    samplers.resize(num_textures);
    size_t view_index{};
    size_t sampler_index{};

    texture_cache.SynchronizeDescriptors(false);

    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);
    draw_timing.Mark(0);

    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};
    const u32 sampler_limit = via_header_index ? regs.tex_header.limit : regs.tex_sampler.limit;
    scratch.sampler_ids.SetInvalidIndexLimit(
        sampler_limit, VideoCommon::NULL_SAMPLER_ID,
        maxwell3d->GetNativeMacroEngine().native_turing && draw_timing.FastInvalidSamplers() &&
        !Common::Log::IsMessageEnabled(Common::Log::Class::HW_GPU, Common::Log::Level::Debug));
    // Buffer-backed texture arrays: the root, size and mapping are validated once per descriptor
    // (index 0) and the handle records are read with one block read instead of per element.
    const Shader::TextureDescriptor* buffer_texture_desc{};
    const DecodedTextureArray* buffer_texture_records{};
    GPUVAddr buffer_texture_base{};
    auto& buffer_texture_bytes = scratch.texture_bytes;
    const auto read_buffer_texture_word{[&](u64 offset) {
        if (offset + sizeof(u32) <= buffer_texture_bytes.size()) {
            u32 word;
            std::memcpy(&word, buffer_texture_bytes.data() + offset, sizeof(word));
            return word;
        }
        return gpu_memory->Read<u32>(buffer_texture_base + offset);
    }};
    // Texture handles of a stage usually sit in one constant buffer page; reuse its host pointer
    // instead of translating every handle address.
    GPUVAddr handle_page{~GPUVAddr{0}};
    const u8* handle_page_pointer{};
    const auto read_handle_word{[&](GPUVAddr addr) -> u32 {
        constexpr GPUVAddr page_mask = Core::DEVICE_PAGESIZE - 1;
        if ((addr & page_mask) > Core::DEVICE_PAGESIZE - sizeof(u32)) {
            return gpu_memory->Read<u32>(addr);
        }
        if ((addr & ~page_mask) != handle_page) {
            handle_page = addr & ~page_mask;
            handle_page_pointer = gpu_memory->GetPointer(handle_page);
        }
        if (!handle_page_pointer) {
            return gpu_memory->Read<u32>(addr);
        }
        u32 word;
        std::memcpy(&word, handle_page_pointer + (addr & page_mask), sizeof(word));
        return word;
    }};
    const auto config_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        const Shader::Info& info{stage_infos[stage]};
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        if constexpr (Spec::has_storage_buffers) {
            size_t ssbo_index{};
            for (const auto& desc : info.storage_buffers_descriptors) {
                ASSERT(desc.count == 1);
                buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                       desc.cbuf_offset, desc.is_written);
                ++ssbo_index;
            }
        }
        const auto& cbufs{maxwell3d->state.shader_stages[stage].const_buffers};
        const auto read_handle{[&](const auto& desc, u32 index) LAMBDA_FORCEINLINE {
            ASSERT(cbufs[desc.cbuf_index].enabled);
            if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&>) {
                if (desc.buffer_source.enabled) {
                    const auto& source = desc.buffer_source;
                    if (index == 0 || buffer_texture_desc != &desc) {
                        const u64 span = TextureBufferReadSize(desc);
                        const GPUVAddr root = cbufs[desc.cbuf_index].address + desc.cbuf_offset;
                        const GPUVAddr base = gpu_memory->Read<u64>(root);
                        if (gpu_memory->Read<u64>(root + 8) != source.size || !base ||
                            base + source.size < base || !gpu_memory->IsFullyMappedRange(base, span))
                            throw Shader::RuntimeError("Buffer-backed texture array changed or is unmapped");
                        if (index == 0) {
                            VideoCommon::ReadbackTrace::ReaderScope readback_scope{base, source.size,
                                source.offset, source.stride, desc.count, key.unique_hashes[stage == 4 ? 5 : stage + 1]};
                            GPUVAddr expected = base;
                            for (const auto& [range_address, range_size] : gpu_memory->GetSubmappedRange(base, span)) {
                                const auto device_address = gpu_memory->GpuToCpuAddress(range_address);
                                if (range_address != expected || !range_size || !device_address)
                                    throw Shader::RuntimeError("Unmapped texture-handle buffer segment");
                                VideoCommon::ReadbackTrace::Segment(*device_address, range_address, range_size);
                                buffer_cache.DownloadMemory(*device_address, range_size);
                                expected += range_size;
                            }
                            if (expected != base + span)
                                throw Shader::RuntimeError("Incomplete texture-handle buffer mapping");
                            static const bool dump_buffer_textures = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
                            if (dump_buffer_textures) {
                                // Written once per name: rewriting it on every draw slowed
                                // diagnostic runs to about 1 fps.
                                static std::set<std::string> dumped;
                                std::string name = fmt::format("native-buffer-textures-{}-{}-{:x}-{:x}.bin", stage,
                                    desc.cbuf_index, desc.cbuf_offset, source.offset);
                                if (dumped.insert(name).second) {
                                    std::vector<u8> bytes(span);
                                    gpu_memory->ReadBlock(base, bytes.data(), bytes.size());
                                    std::ofstream dump(name, std::ios::binary);
                                    dump.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                                }
                            }
                        }
                        buffer_texture_desc = &desc;
                        buffer_texture_base = base;
                        buffer_texture_bytes.resize(span);
                        gpu_memory->ReadBlockUnsafe(base, buffer_texture_bytes.data(), span);
                        // Reuse only decoding of identical freshly read bytes. Guest mappings,
                        // GPU readbacks, image views and sampler resources remain current.
                        buffer_texture_records = scratch.texture_records.Resolve(
                            desc, buffer_texture_bytes, via_header_index);
                        if (buffer_texture_records &&
                            (buffer_texture_records->max_image > regs.tex_header.limit ||
                             buffer_texture_records->max_sampler > regs.tex_sampler.limit))
                            throw Shader::RuntimeError("Buffer-backed texture handle outside descriptor tables");
                    }
                    if (buffer_texture_records) return buffer_texture_records->records[index].handle;
                    const u64 offset = source.offset + u64{index} * source.stride;
                    if (offset + sizeof(u32) > source.size)
                        throw Shader::RuntimeError("Texture-handle record outside buffer");
                    const auto pair = TexturePair(read_buffer_texture_word(offset), via_header_index);
                    if (pair.first > regs.tex_header.limit || pair.second > regs.tex_sampler.limit)
                        throw Shader::RuntimeError("Buffer-backed texture handle outside descriptor tables");
                    return pair;
                }
            }
            const u32 index_offset{index << desc.size_shift};
            const u32 offset{desc.cbuf_offset + index_offset};
            const GPUVAddr addr{cbufs[desc.cbuf_index].address + offset};
            if constexpr (std::is_same_v<decltype(desc), const Shader::TextureDescriptor&> ||
                          std::is_same_v<decltype(desc), const Shader::TextureBufferDescriptor&>) {
                if (desc.has_secondary) {
                    ASSERT(cbufs[desc.secondary_cbuf_index].enabled);
                    const u32 second_offset{desc.secondary_cbuf_offset + index_offset};
                    const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].address +
                                                 second_offset};
                    const u32 lhs_raw{read_handle_word(addr) << desc.shift_left};
                    const u32 rhs_raw{read_handle_word(separate_addr)
                                      << desc.secondary_shift_left};
                    const u32 raw{lhs_raw | rhs_raw};
                    return TexturePair(raw, via_header_index);
                }
            }
            return TexturePair(read_handle_word(addr), via_header_index);
        }};
        const auto add_image{[&](const auto& desc, bool blacklist) LAMBDA_FORCEINLINE {
            ASSERT(desc.count <= views.size() - view_index);
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle{read_handle(desc, index)};
                views[view_index++] = {
                    .index = handle.first,
                    .blacklist = blacklist,
                    .id = {}
                };
            }
        }};
        if constexpr (Spec::has_texture_buffers) {
            for (const auto& desc : info.texture_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        if constexpr (Spec::has_image_buffers) {
            for (const auto& desc : info.image_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        for (const auto& desc : info.texture_descriptors) {
            ASSERT(desc.count <= views.size() - view_index);
            ASSERT(desc.count <= samplers.size() - sampler_index);
            if (desc.count >= 64 && !desc.buffer_source.enabled && draw_timing.DirectHandles()) {
                ASSERT(cbufs[desc.cbuf_index].enabled);
                if (desc.has_secondary) ASSERT(cbufs[desc.secondary_cbuf_index].enabled);
                TextureHandleSpan handles;
                if (handles.Prepare(*gpu_memory, cbufs[desc.cbuf_index].address, desc.cbuf_offset,
                                    desc.has_secondary ? cbufs[desc.secondary_cbuf_index].address : 0,
                                    desc.secondary_cbuf_offset, desc.count, desc.size_shift,
                                    desc.has_secondary, desc.shift_left, desc.secondary_shift_left)) {
                    scratch.sampler_ids.BeginArray();
                    u32 last_sampler_index = ~u32{0};
                    VideoCommon::SamplerId last_sampler_id = VideoCommon::NULL_SAMPLER_ID;
                    auto* array_views = views.data() + view_index;
                    auto* array_samplers = samplers.data() + sampler_index;
                    namespace Optimizations = VideoCommon::NativeDrawOptimizations;
                    if (Optimizations::tight_texture_loops.load(std::memory_order_relaxed)) {
                        auto resolve = [&](u32 sampler) { return texture_cache.GetSamplerId(sampler, false); };
                        if (Optimizations::check_texture_loops.load(std::memory_order_relaxed)) {
                            FillDirectTexturesChecked(handles, desc.count, via_header_index,
                                scratch.sampler_ids, resolve, array_views, array_samplers,
                                Optimizations::vector_texture_handle_scan.load(std::memory_order_relaxed));
                        } else {
                            FillDirectTextures(handles, desc.count, via_header_index, scratch.sampler_ids,
                                               resolve, array_views, array_samplers);
                        }
                    } else if (VideoCommon::NativeDrawOptimizations::batch_texture_handle_runs.load(
                            std::memory_order_relaxed)) {
                        handles.ForEachRun(desc.count, [&](u32 start, u32 count, u32 raw) {
                            const auto handle = TexturePair(raw, via_header_index);
                            if (handle.second != last_sampler_index) {
                                last_sampler_index = handle.second;
                                last_sampler_id = scratch.sampler_ids.Get(handle.second, [&](u32 sampler) {
                                    return texture_cache.GetSamplerId(sampler, false);
                                });
                            }
                            std::fill_n(array_views + start, count, VideoCommon::ImageViewInOut{handle.first});
                            std::fill_n(array_samplers + start, count, last_sampler_id);
                        }, VideoCommon::NativeDrawOptimizations::vector_texture_handle_scan.load(
                               std::memory_order_relaxed));
                    } else {
                        for (u32 index = 0; index < desc.count; ++index) {
                            const auto handle = TexturePair(handles.Read(index), via_header_index);
                            array_views[index] = {handle.first};
                            if (handle.second != last_sampler_index) {
                                last_sampler_index = handle.second;
                                last_sampler_id = scratch.sampler_ids.Get(handle.second, [&](u32 sampler) {
                                    return texture_cache.GetSamplerId(sampler, false);
                                });
                            }
                            array_samplers[index] = last_sampler_id;
                        }
                    }
                    view_index += desc.count;
                    sampler_index += desc.count;
                    continue;
                }
            }
            const bool cache_sampler_ids = desc.count >= 64;
            if (cache_sampler_ids) scratch.sampler_ids.BeginArray();
            u32 last_sampler_index = ~u32{0};
            VideoCommon::SamplerId last_sampler_id = VideoCommon::NULL_SAMPLER_ID;
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle{read_handle(desc, index)};
                bool excluded = false;
                if (desc.buffer_source.enabled && buffer_texture_records) {
                    excluded = buffer_texture_records->records[index].excluded;
                    if (excluded) excluded_views.push_back(view_index);
                } else if (desc.buffer_source.enabled) {
                    const auto& source = desc.buffer_source;
                    for (const auto& exclusion : source.exclusions) {
                        if (!exclusion.mask) continue;
                        if (exclusion.offset > source.stride || source.stride - exclusion.offset < sizeof(u32))
                            throw Shader::RuntimeError("Invalid buffer texture exclusion field");
                        const u32 word = read_buffer_texture_word(u64{index} * source.stride + exclusion.offset);
                        if (exclusion.Matches(word)) {
                            excluded_views.push_back(view_index);
                            excluded = true;
                            break;
                        }
                    }
                }
                views[view_index++] = {handle.first};

                if (excluded) {
                    samplers[sampler_index++] = VideoCommon::NULL_SAMPLER_ID;
                } else {
                    // Array elements often share a sampler. The descriptor-array setup
                    // does not write the sampler table between these reads.
                    if (handle.second != last_sampler_index) {
                        last_sampler_index = handle.second;
                        const auto resolve = [&](u32 sampler) {
                            return texture_cache.GetSamplerId(sampler, false);
                        };
                        last_sampler_id = cache_sampler_ids
                            ? scratch.sampler_ids.Get(handle.second, resolve)
                            : resolve(handle.second);
                    }
                    samplers[sampler_index++] = last_sampler_id;
                }
            }
        }
        if constexpr (Spec::has_images) {
            for (const auto& desc : info.image_descriptors) {
                add_image(desc, desc.is_written);
            }
        }

        return true;
    }};
    if constexpr (Spec::enabled_stages[0]) {
        config_stage(0);
    }
    if constexpr (Spec::enabled_stages[1]) {
        config_stage(1);
    }
    if constexpr (Spec::enabled_stages[2]) {
        config_stage(2);
    }
    if constexpr (Spec::enabled_stages[3]) {
        config_stage(3);
    }
    if constexpr (Spec::enabled_stages[4]) {
        config_stage(4);
    }
    ASSERT(view_index == views.size());
    ASSERT(sampler_index == samplers.size());
    draw_timing.Mark(1);
    texture_cache.FillImageViews(std::span(views.data(), views.size()), false, Spec::has_images,
                                std::span(excluded_views.data(), excluded_views.size()),
                                draw_timing.BatchImageViews(), draw_timing.FastInvalidViews(),
                                draw_timing.BatchInvalidViews());
    draw_timing.Mark(2);
    // Armed by native-dispatch-trace.request: give each traced draw's targets and shaders, so the
    // DispatchTrace line that follows (its texture bindings) reads as one frame-order record.
    if (auto& budget = VideoCommon::NativeDispatchDimsBudget(); budget > 0) {
        --budget;
        std::string targets;
        for (u32 rt = 0; rt < regs.rt_control.count; ++rt) {
            const u32 slot = regs.rt_control.Map(rt);
            targets += fmt::format(" rt{}={:x}:{:#x}:{}x{}", slot, regs.rt[slot].Address(),
                                   static_cast<u32>(regs.rt[slot].format), regs.rt[slot].width,
                                   regs.rt[slot].height);
        }
        LOG_ERROR(Render_Vulkan, "DrawTarget{} zeta={:x} blend0={} vs={:x} ps={:x} count={}", targets,
                  regs.zeta_enable ? regs.zeta.Address() : 0, regs.blend.enable[0],
                  (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1],
                  (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1],
                  maxwell3d->draw_manager.draw_state.vertex_buffer.count);
    }
    const bool capture_draw = VideoCommon::NativeTextureRequestsEnabled() && WantDrawCapture(regs);
    if (texture_cache.CaptureNativeGpuTexture(std::span(views.data(), views.size())) || capture_draw) {
        std::ofstream capture("native-gpu-texture-shaders.txt");
        std::ofstream cbuf_layout("native-gpu-texture-cbuf-layout.txt");
        const auto& world_buffer = maxwell3d->state.shader_stages[0].const_buffers[10];
        if (world_buffer.enabled && world_buffer.size >= 256 &&
            gpu_memory->IsFullyMappedRange(world_buffer.address, 256)) {
            std::array<u8, 256> before{};
            gpu_memory->ReadBlockUnsafe(world_buffer.address, before.data(), before.size());
            VideoCommon::NativeTextureCapture::Write("native-gpu-texture-world-before.bin", before.data(), before.size());
            const auto device_address = gpu_memory->GpuToCpuAddress(world_buffer.address);
            std::ofstream world_state("native-gpu-texture-world-state.txt");
            world_state << "gpu " << std::hex << world_buffer.address << "\ndevice " << device_address.value_or(0)
                        << "\ngpu_modified " << (device_address && buffer_cache.IsRegionGpuModified(*device_address, before.size())) << '\n';
        }
        for (size_t stage = 0; stage < key.unique_hashes.size(); ++stage)
            capture << stage << ' ' << std::hex << key.unique_hashes[stage] << '\n';
        VideoCommon::NativeTextureCapture::Write("native-gpu-texture-registers.bin", &regs, sizeof(regs));
        const auto& draw = maxwell3d->draw_manager.draw_state;
        capture << "draw " << std::dec << draw.vertex_buffer.first << ' ' << draw.vertex_buffer.count
                << ' ' << draw.base_index << ' ' << draw.index_buffer.first << ' ' << draw.index_buffer.count << '\n';
        for (u32 stream = 0; stream < 4; ++stream) {
            const auto& config = regs.vertex_streams[stream];
            if (!config.IsEnabled()) continue;
            const GPUVAddr address = config.Address();
            capture << "vertex " << stream << ' ' << std::hex << address << '\n';
            if (!gpu_memory->IsFullyMappedRange(address, 1024)) continue;
            std::array<u8, 1024> data{};
            gpu_memory->ReadBlock(address, data.data(), data.size());
            VideoCommon::NativeTextureCapture::Write("native-gpu-texture-vertex-" + std::to_string(stream) + ".bin", data.data(), data.size());
        }
        for (u32 stage : {1U, 5U}) {
            const auto& config = regs.pipelines[stage];
            const GPUVAddr address = (u64{config.reserved[0] & 0xff} << 32) | config.reserved[1];
            if (!gpu_memory->IsFullyMappedRange(address, 128)) continue;
            capture << "program " << stage << ' ' << std::hex << address << '\n';
            std::vector<u8> source(128);
            gpu_memory->ReadBlock(address, source.data(), source.size());
            Shader::Ampere::ProgramCaptureState state;
            for (u32 index = 0; index < 4096; ++index) {
                Shader::Ampere::Instruction instruction;
                const GPUVAddr pc = address + 128 + index * sizeof(instruction);
                if (!gpu_memory->IsFullyMappedRange(pc, sizeof(instruction))) break;
                gpu_memory->ReadBlock(pc, &instruction, sizeof(instruction));
                const auto* bytes = reinterpret_cast<const u8*>(&instruction);
                source.insert(source.end(), bytes, bytes + sizeof(instruction));
                const auto status = state.Observe(instruction, index);
                if (status == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) break;
                if (status == Shader::Ampere::ProgramCaptureState::Result::Complete) {
                    VideoCommon::NativeTextureCapture::Write("native-gpu-texture-shader-" + std::to_string(stage) + ".bin", source.data(), source.size());
                    break;
                }
            }
            const auto& constants = maxwell3d->state.shader_stages[stage == 1 ? 0 : 4].const_buffers;
            for (u32 bank = 0; bank < constants.size(); ++bank) {
                if (Tegra::NativeRootTables::IsBank(bank)) {
                    const auto& words = maxwell3d->native_root_tables.Data(bank);
                    cbuf_layout << stage << ' ' << bank << " 1 root 100\n";
                    VideoCommon::NativeTextureCapture::Write("native-gpu-texture-cbuf-" + std::to_string(stage) + "-" + std::to_string(bank) + ".bin", words.data(), sizeof(words));
                    continue;
                }
                const auto& buffer = constants[bank];
                cbuf_layout << stage << ' ' << bank << ' ' << buffer.enabled << ' '
                            << std::hex << buffer.address << ' ' << buffer.size << std::dec << '\n';
                const size_t capture_size = std::min<size_t>(buffer.size, 65536);
                if (!buffer.enabled || !capture_size ||
                    !gpu_memory->IsFullyMappedRange(buffer.address, capture_size)) continue;
                std::vector<u8> data(capture_size);
                gpu_memory->ReadBlock(buffer.address, data.data(), data.size());
                VideoCommon::NativeTextureCapture::Write("native-gpu-texture-cbuf-" + std::to_string(stage) + "-" + std::to_string(bank) + ".bin", data.data(), data.size());
            }
        }
    }

    VideoCommon::ImageViewInOut* texture_buffer_it{views.data()};
    const auto bind_stage_info{[&](size_t stage) LAMBDA_FORCEINLINE {
        size_t index{};
        const auto add_buffer{[&](const auto& desc) {
            constexpr bool is_image = std::is_same_v<decltype(desc), const ImageBufferDescriptor&>;
            for (u32 i = 0; i < desc.count; ++i) {
                bool is_written{false};
                if constexpr (is_image) {
                    is_written = desc.is_written;
                }
                ImageView& image_view{texture_cache.GetImageView(texture_buffer_it->id)};
                PixelFormat format{image_view.format};
                if constexpr (is_image) {
                    if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                        format = *explicit_format;
                    }
                }
                buffer_cache.BindGraphicsTextureBuffer(stage, index, image_view.GpuAddr(),
                                                       image_view.BufferSize(), format,
                                                       is_written, is_image);
                ++index;
                ++texture_buffer_it;
            }
        }};
        buffer_cache.UnbindGraphicsTextureBuffers(stage);

        const Shader::Info& info{stage_infos[stage]};
        if constexpr (Spec::has_texture_buffers) {
            for (const auto& desc : info.texture_buffer_descriptors) {
                add_buffer(desc);
            }
        }
        if constexpr (Spec::has_image_buffers) {
            for (const auto& desc : info.image_buffer_descriptors) {
                add_buffer(desc);
            }
        }
        texture_buffer_it += Shader::NumDescriptors(info.texture_descriptors);
        if constexpr (Spec::has_images) {
            texture_buffer_it += Shader::NumDescriptors(info.image_descriptors);
        }
    }};
    if constexpr (Spec::enabled_stages[0]) {
        bind_stage_info(0);
    }
    if constexpr (Spec::enabled_stages[1]) {
        bind_stage_info(1);
    }
    if constexpr (Spec::enabled_stages[2]) {
        bind_stage_info(2);
    }
    if constexpr (Spec::enabled_stages[3]) {
        bind_stage_info(3);
    }
    if constexpr (Spec::enabled_stages[4]) {
        bind_stage_info(4);
    }

    if (regs.transform_feedback_enabled != 0) {
        scheduler.RequestOutsideRenderPassOperationContext();
    }

    buffer_cache.UpdateGraphicsBuffers(is_indexed);
    buffer_cache.BindHostGeometryBuffers(is_indexed);

    guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);
    draw_timing.Mark(3);

    RescalingPushConstant rescaling;
    RenderAreaPushConstant render_area;
    const VideoCommon::SamplerId* samplers_it{samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{views.data()};
    const auto prepare_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        const auto& shader_config = regs.pipelines[stage == 4 ? 5 : stage + 1];
        VideoCommon::ReadbackTrace::WriterScope write_scope{"graphics", key.unique_hashes[stage == 4 ? 5 : stage + 1],
            (GPUVAddr{shader_config.reserved[0] & 0xff} << 32) | shader_config.reserved[1]};
        buffer_cache.BindHostStageBuffers(stage);
        draw_timing.Mark(4);
        const auto* stage_views = views_it;
        try {
            PushImageDescriptors(texture_cache, guest_descriptor_queue, stage_infos[stage], rescaling,
                                 samplers_it, views_it);
            draw_timing.Mark(5);
        } catch (const Shader::Exception&) {
            static u32 diagnostic_count{};
            if (diagnostic_count++ < 8) {
                if (std::getenv("EDEN_NATIVE_DRAW_ON_DEMAND")) {
                    const auto& config = regs.pipelines[stage == 4 ? 5 : 1];
                    const GPUVAddr address = (GPUVAddr{config.reserved[0] & 0xff} << 32) | config.reserved[1];
                    size_t length{};
                    while (length < 0x20000 && gpu_memory->IsFullyMappedRange(address + length, 0x1000))
                        length += 0x1000;
                    std::vector<u8> bytes(length);
                    gpu_memory->ReadBlock(address, bytes.data(), bytes.size());
                    std::ofstream shader(fmt::format("native-binding-error-shader-{}-{:x}.bin", stage, address), std::ios::binary);
                    shader.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    for (u32 bank = 0; bank < 18; ++bank) {
                        const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                        if (!cb.enabled || !cb.size || cb.size > 65536 ||
                            !gpu_memory->IsFullyMappedRange(cb.address, cb.size)) continue;
                        bytes.resize(cb.size);
                        gpu_memory->ReadBlock(cb.address, bytes.data(), bytes.size());
                        std::ofstream file(fmt::format("native-binding-error-cbuf-{}-{}.bin", stage, bank), std::ios::binary);
                        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    }
                }
                const auto& info = stage_infos[stage];
                const auto* texture_views = stage_views +
                    Shader::NumDescriptors(info.texture_buffer_descriptors) +
                    Shader::NumDescriptors(info.image_buffer_descriptors);
                for (const auto& desc : info.texture_descriptors) {
                    for (u32 index = 0; index < desc.count; ++index) {
                        const auto& view = *texture_views++;
                        if (!desc.buffer_source.enabled || view.id == VideoCommon::NULL_IMAGE_VIEW_ID)
                            continue;
                        auto& cached = texture_cache.GetImageView(view.id);
                        if (!VideoCore::Surface::IsPixelFormatInteger(cached.format) &&
                            cached.Handle(desc.type) != VK_NULL_HANDLE)
                            continue;
                        const auto tic_address = regs.tex_header.Address() +
                            u64{view.index} * sizeof(Tegra::Texture::TICEntry);
                        Tegra::Texture::TICEntry tic{};
                        gpu_memory->ReadBlock(tic_address, &tic, sizeof(tic));
                        const auto& cbuf = maxwell3d->state.shader_stages[stage].const_buffers[desc.cbuf_index];
                        const auto base = gpu_memory->Read<u64>(cbuf.address + desc.cbuf_offset);
                        const auto size = gpu_memory->Read<u64>(cbuf.address + desc.cbuf_offset + 8);
                        LOG_ERROR(Render_Vulkan,
                            "Texture binding evidence: stage {} entry {} TIC {} at {:#x} raw {:016x} {:016x} {:016x} {:016x}; cached type {} format {}; CB{}+{:#x} base {:#x} size {:#x} stride {:#x} field {:#x}",
                            stage, index, view.index, tic_address, tic.raw[0], tic.raw[1], tic.raw[2], tic.raw[3],
                            static_cast<u32>(cached.type), static_cast<u32>(cached.format),
                            desc.cbuf_index, desc.cbuf_offset, base, size,
                            desc.buffer_source.stride, desc.buffer_source.offset);
                        if (size == desc.buffer_source.size && size <= 16 * 1024 * 1024 &&
                            base && base + size >= base && gpu_memory->IsFullyMappedRange(base, size)) {
                            std::vector<u8> bytes(size);
                            gpu_memory->ReadBlock(base, bytes.data(), bytes.size());
                            std::ofstream dump(fmt::format("native-binding-error-{}-{}-{:x}-{:x}-{:x}.bin",
                                stage, desc.cbuf_index, desc.cbuf_offset, base, desc.buffer_source.offset),
                                std::ios::binary);
                            dump.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                        }
                    }
                }
            }
            throw;
        }
        const auto& info{stage_infos[stage]};
        if (info.uses_render_area) {
            render_area.uses_render_area = true;
            render_area.words = {static_cast<float>(regs.surface_clip.width),
                                 static_cast<float>(regs.surface_clip.height)};
        }
    }};
    if constexpr (Spec::enabled_stages[0]) {
        prepare_stage(0);
    }
    if constexpr (Spec::enabled_stages[1]) {
        prepare_stage(1);
    }
    if constexpr (Spec::enabled_stages[2]) {
        prepare_stage(2);
    }
    if constexpr (Spec::enabled_stages[3]) {
        prepare_stage(3);
    }
    if constexpr (Spec::enabled_stages[4]) {
        prepare_stage(4);
    }
    draw_timing.Mark(4);
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }
    texture_cache.UpdateRenderTargets(false);
    texture_cache.CheckFeedbackLoop(std::span<const VideoCommon::ImageViewInOut>{views.data(),
                                                                                 views.size()});
    if (IsBuilt() && !pipeline) {
        return false;
    }
    draw_timing.Mark(6);
    const bool configured = ConfigureDraw(rescaling, render_area);
    draw_timing.Mark(7);
    return configured;
}

bool GraphicsPipeline::ConfigureDraw(const RescalingPushConstant& rescaling,
                                     const RenderAreaPushConstant& render_area) {
    const void* const descriptor_data{guest_descriptor_queue.UpdateData()};

    VkDeviceSize descriptor_buffer_offset{};
    u32 descriptor_buffer_chunk{};
    if (descriptor_set_layout && uses_descriptor_buffer) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        const bool reuse_allocation =
            last_descriptor_buffer_generation == descriptor_buffer_ring.CurrentGeneration() &&
            last_descriptor_payload.size() == num_descriptor_entries &&
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) == 0;
        if (reuse_allocation) {
            descriptor_buffer_offset = last_descriptor_buffer_offset;
            descriptor_buffer_chunk = last_descriptor_buffer_chunk;
            descriptor_buffer_ring.TouchFrame(scheduler);
        } else {
            const DescriptorBufferRing::Allocation alloc{
                descriptor_buffer_ring.Allocate(scheduler, descriptor_buffer_layout.size)};
            if (!alloc.host) {
                LOG_DEBUG(Render_Vulkan, "Failed to reserve descriptor memory, skipping draw");
                return false;
            }
            WriteDescriptorBuffer(device, descriptor_buffer_layout, entries, alloc.host);
            descriptor_buffer_offset = alloc.offset;
            descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_offset = alloc.offset;
            last_descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_generation = alloc.generation;
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }

    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    if (!is_built.load(std::memory_order::relaxed)) {
        // Wait for the pipeline to be built
        scheduler.Record([this](vk::CommandBuffer) {
            std::unique_lock lock{build_mutex};
            build_condvar.wait(lock, [this] { return is_built.load(std::memory_order::relaxed); });
        });
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const bool update_rescaling{scheduler.UpdateRescaling(is_rescaling)};
    const bool bind_pipeline{scheduler.UpdateGraphicsPipeline(this)};
    const bool bind_descriptor_buffer{
        descriptor_set_layout && uses_descriptor_buffer &&
        scheduler.UpdateDescriptorBufferChunk(descriptor_buffer_chunk)};

    // Log graphics pipeline binding
    if (bind_pipeline && GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string pipeline_info = fmt::format("hash={:#016x}", key.Hash());
        GPU::Logging::GPULogger::GetInstance().LogPipelineBind(false, pipeline_info);
    }

    bool update_descriptors = true;
    if (descriptor_set_layout && !uses_push_descriptor && !uses_descriptor_buffer) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        update_descriptors =
            bind_pipeline || last_descriptor_payload.size() != num_descriptor_entries ||
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) != 0;
        if (update_descriptors) {
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }
    if (uses_descriptor_buffer || (!uses_push_descriptor && !update_descriptors)) {
        guest_descriptor_queue.Discard();
    }
    scheduler.Record([this, descriptor_data, bind_pipeline, update_descriptors,
                      descriptor_buffer_offset, descriptor_buffer_chunk, bind_descriptor_buffer,
                      rescaling_data = rescaling.Data(), is_rescaling, update_rescaling,
                      uses_render_area = render_area.uses_render_area,
                      render_area_data = render_area.words](vk::CommandBuffer cmdbuf) {
        if (bind_descriptor_buffer) {
            const VkDescriptorBufferBindingInfoEXT binding_info{
                descriptor_buffer_ring.BindingInfo(descriptor_buffer_chunk)};
            cmdbuf.BindDescriptorBuffersEXT(binding_info);
        }
        if (bind_pipeline) {
            if (!pipeline) {
                return;
            }
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
        }
        cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                             RESCALING_LAYOUT_WORDS_OFFSET, sizeof(rescaling_data),
                             rescaling_data.data());
        if (update_rescaling) {
            const f32 config_down_factor{Settings::values.resolution_info.down_factor};
            const f32 scale_down_factor{is_rescaling ? config_down_factor : 1.0f};
            cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                                 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, sizeof(scale_down_factor),
                                 &scale_down_factor);
        }
        if (uses_render_area) {
            cmdbuf.PushConstants(*pipeline_layout, VK_SHADER_STAGE_ALL_GRAPHICS,
                                 RENDERAREA_LAYOUT_OFFSET, sizeof(render_area_data),
                                 &render_area_data);
        }
        if (!descriptor_set_layout) {
            return;
        }
        if (uses_descriptor_buffer) {
            const u32 buffer_index{};
            cmdbuf.SetDescriptorBufferOffsetsEXT(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout,
                                                 0, buffer_index, descriptor_buffer_offset);
        } else if (uses_push_descriptor) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(*descriptor_update_template, *pipeline_layout,
                                                    0, descriptor_data);
        } else if (update_descriptors) {
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            const vk::Device& dev{device.GetLogical()};
            dev.UpdateDescriptorSet(descriptor_set, *descriptor_update_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                                      descriptor_set, nullptr);
        }
    });
    return true;
}

void GraphicsPipeline::MakePipeline(VkRenderPass render_pass) {
    FixedPipelineState::DynamicState dynamic{};
    if (!key.state.extended_dynamic_state) {
        dynamic = key.state.dynamic_state;
    } else {
        dynamic.raw1 = key.state.dynamic_state.raw1;
    }
    static_vector<VkVertexInputBindingDescription, 32> vertex_bindings;
    static_vector<VkVertexInputBindingDivisorDescriptionEXT, 32> vertex_binding_divisors;
    static_vector<VkVertexInputAttributeDescription, 32> vertex_attributes;
    if (!key.state.dynamic_vertex_input) {
        const size_t num_vertex_arrays = (std::min)(
            Maxwell::NumVertexArrays, static_cast<size_t>(device.GetMaxVertexInputBindings()));
        for (size_t index = 0; index < num_vertex_arrays; ++index) {
            const bool instanced = key.state.binding_divisors[index] != 0;
            const auto rate =
                instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
            vertex_bindings.push_back({
                .binding = static_cast<u32>(index),
                .stride = key.state.vertex_strides[index],
                .inputRate = rate,
            });
            if (instanced) {
                vertex_binding_divisors.push_back({
                    .binding = static_cast<u32>(index),
                    .divisor = key.state.binding_divisors[index],
                });
            }
        }
        for (size_t index = 0; index < key.state.attributes.size(); ++index) {
            const auto& attribute = key.state.attributes[index];
            if (!attribute.enabled || !stage_infos[0].loads.Generic(index)) {
                continue;
            }
            vertex_attributes.push_back({
                .location = static_cast<u32>(index),
                .binding = attribute.buffer,
                .format = MaxwellToVK::VertexFormat(device, attribute.Type(), attribute.Size()),
                .offset = attribute.offset,
            });
        }
    }
    ASSERT(vertex_attributes.size() <= device.GetMaxVertexInputAttributes());

    VkPipelineVertexInputStateCreateInfo vertex_input_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<u32>(vertex_bindings.size()),
        .pVertexBindingDescriptions = vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(vertex_attributes.size()),
        .pVertexAttributeDescriptions = vertex_attributes.data(),
    };
    const VkPipelineVertexInputDivisorStateCreateInfoEXT input_divisor_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .vertexBindingDivisorCount = static_cast<u32>(vertex_binding_divisors.size()),
        .pVertexBindingDivisors = vertex_binding_divisors.data(),
    };
    if (!vertex_binding_divisors.empty()) {
        vertex_input_ci.pNext = &input_divisor_ci;
    }
    const bool has_tess_stages = spv_modules[1] || spv_modules[2];
    auto input_assembly_topology = MaxwellToVK::PrimitiveTopology(device, key.state.topology);
    if (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
        if (!has_tess_stages) {
            LOG_WARNING(Render_Vulkan, "Patch topology used without tessellation, using points");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        }
    } else {
        if (has_tess_stages) {
            // The Vulkan spec requires patch list IA topology be used with tessellation
            // shader stages. Forcing it fixes a crash on some drivers
            LOG_WARNING(Render_Vulkan,
                        "Patch topology not used with tessellation, using patch list");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        }
    }
    const VkPipelineInputAssemblyStateCreateInfo input_assembly_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = input_assembly_topology,
        .primitiveRestartEnable =
        // MoltenVK/Metal always has primitive restart enabled and cannot disable it
        device.IsMoltenVK() ? VK_TRUE :
        (dynamic.primitive_restart_enable != 0 &&
                ((input_assembly_topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsTopologyListPrimitiveRestartSupported()) ||
                 SupportsPrimitiveRestart(input_assembly_topology) ||
                 (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsPatchListPrimitiveRestartSupported()))
            ? VK_TRUE
            : VK_FALSE),
    };
    const VkPipelineTessellationStateCreateInfo tessellation_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .patchControlPoints = key.state.patch_control_points_minus_one.Value() + 1,
    };
    std::array<VkViewportSwizzleNV, Maxwell::NumViewports> swizzles;
    std::ranges::transform(key.state.viewport_swizzles, swizzles.begin(), UnpackViewportSwizzle);
    VkPipelineViewportSwizzleStateCreateInfoNV swizzle_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = Maxwell::NumViewports,
        .pViewportSwizzles = swizzles.data(),
    };
    VkPipelineViewportDepthClipControlCreateInfoEXT ndc_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .pNext = nullptr,
        .negativeOneToOne = key.state.ndc_minus_one_to_one.Value() != 0 ? VK_TRUE : VK_FALSE,
    };
    const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
    VkPipelineViewportStateCreateInfo viewport_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = num_viewports,
        .pViewports = nullptr,
        .scissorCount = num_viewports,
        .pScissors = nullptr,
    };
    if (device.IsNvViewportSwizzleSupported()) {
        swizzle_ci.pNext = std::exchange(viewport_ci.pNext, &swizzle_ci);
    }
    if (device.IsExtDepthClipControlSupported()) {
        ndc_info.pNext = std::exchange(viewport_ci.pNext, &ndc_info);
    }
    VkPipelineRasterizationStateCreateInfo rasterization_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable =
        static_cast<VkBool32>(dynamic.depth_clamp_disabled == 0 ? VK_TRUE : VK_FALSE),
        .rasterizerDiscardEnable =
        static_cast<VkBool32>(dynamic.rasterize_enable == 0 ? VK_TRUE : VK_FALSE),
        .polygonMode =
        MaxwellToVK::PolygonMode(FixedPipelineState::UnpackPolygonMode(key.state.polygon_mode)),
        .cullMode = static_cast<VkCullModeFlags>(
            dynamic.cull_enable ? MaxwellToVK::CullFace(dynamic.CullFace()) : VK_CULL_MODE_NONE),
        .frontFace = MaxwellToVK::FrontFace(dynamic.FrontFace()),
        .depthBiasEnable = (dynamic.depth_bias_enable != 0 ? VK_TRUE : VK_FALSE),
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
        // TODO(alekpop): Transfer from regs
    };
    const bool smooth_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsSmoothLines();
    const bool stippled_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsStippledRectangularLines();
    VkPipelineRasterizationLineStateCreateInfoEXT line_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .lineRasterizationMode = key.state.smooth_lines != 0 && smooth_lines_supported
                                     ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
                                     : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
        .stippledLineEnable =
            (dynamic.line_stipple_enable && stippled_lines_supported) ? VK_TRUE : VK_FALSE,
        .lineStippleFactor = key.state.line_stipple_factor,
        .lineStipplePattern = static_cast<uint16_t>(key.state.line_stipple_pattern),
    };
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .conservativeRasterizationMode = key.state.conservative_raster_enable != 0
                                             ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                                             : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT,
        .extraPrimitiveOverestimationSize = 0.0f,
    };
    const bool supports_provoking_first_mode = device.SupportsProvokingVertexFirstMode();
    const bool supports_provoking_last_mode = device.SupportsProvokingVertexLastMode();
    const bool requested_provoking_last_mode = key.state.provoking_vertex_last != 0;
    const VkProvokingVertexModeEXT provoking_vertex_mode =
        requested_provoking_last_mode
            ? (supports_provoking_last_mode ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                            : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT)
            : (supports_provoking_first_mode ? VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
                                             : VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT);
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .provokingVertexMode = provoking_vertex_mode,
    };

    if (IsLine(input_assembly_topology) && device.IsExtLineRasterizationSupported()) {
        line_state.pNext = std::exchange(rasterization_ci.pNext, &line_state);
    }
    if (device.IsExtConservativeRasterizationSupported()) {
        conservative_raster.pNext = std::exchange(rasterization_ci.pNext, &conservative_raster);
    }
    if (device.IsExtProvokingVertexSupported()) {
        provoking_vertex.pNext = std::exchange(rasterization_ci.pNext, &provoking_vertex);
    }

    const bool supports_alpha_output = fragment_has_color0_output;
    const bool alpha_to_one_supported = device.SupportsAlphaToOne();
    const VkPipelineMultisampleStateCreateInfo multisample_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = MaxwellToVK::MsaaMode(key.state.msaa_mode),
        .sampleShadingEnable = Settings::values.sample_shading.GetValue() > 0 ? VK_TRUE : VK_FALSE,
        .minSampleShading = f32(Settings::values.sample_shading.GetValue()) / 100.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable =
            supports_alpha_output && key.state.alpha_to_coverage_enabled != 0 ? VK_TRUE : VK_FALSE,
        .alphaToOneEnable = supports_alpha_output && alpha_to_one_supported &&
                           key.state.alpha_to_one_enabled != 0 ? VK_TRUE : VK_FALSE,
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = dynamic.depth_test_enable,
        .depthWriteEnable = dynamic.depth_write_enable,
        .depthCompareOp = dynamic.depth_test_enable
                              ? MaxwellToVK::ComparisonOp(dynamic.DepthTestFunc())
                              : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = dynamic.depth_bounds_enable && device.IsDepthBoundsSupported(),
        .stencilTestEnable = dynamic.stencil_enable,
        .front = GetStencilFaceState(dynamic.front),
        .back = GetStencilFaceState(dynamic.back),
        .minDepthBounds = f32(key.state.depth_bounds_min),
        .maxDepthBounds = f32(key.state.depth_bounds_max),
    };
    if (dynamic.depth_bounds_enable && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
    }
    static_vector<VkPipelineColorBlendAttachmentState, Maxwell::NumRenderTargets> cb_attachments;
    const size_t num_attachments{NumAttachments(key.state)};
    for (size_t index = 0; index < num_attachments; ++index) {
        static constexpr std::array mask_table{
            VK_COLOR_COMPONENT_R_BIT,
            VK_COLOR_COMPONENT_G_BIT,
            VK_COLOR_COMPONENT_B_BIT,
            VK_COLOR_COMPONENT_A_BIT,
        };
        const auto& blend{key.state.attachments[index]};
        const std::array mask{blend.Mask()};
        VkColorComponentFlags write_mask{};
        for (size_t i = 0; i < mask_table.size(); ++i) {
            write_mask |= mask[i] ? mask_table[i] : 0;
        }
        if (((WrittenColorTargets() >> index) & 1) == 0) {
            write_mask = 0;
        }
        cb_attachments.push_back({
            .blendEnable = blend.enable != 0,
            .srcColorBlendFactor = MaxwellToVK::BlendFactor(blend.SourceRGBFactor()),
            .dstColorBlendFactor = MaxwellToVK::BlendFactor(blend.DestRGBFactor()),
            .colorBlendOp = MaxwellToVK::BlendEquation(blend.EquationRGB()),
            .srcAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.SourceAlphaFactor()),
            .dstAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.DestAlphaFactor()),
            .alphaBlendOp = MaxwellToVK::BlendEquation(blend.EquationAlpha()),
            .colorWriteMask = write_mask,
        });
    }
    const VkPipelineColorBlendStateCreateInfo color_blend_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = dynamic.logic_op_enable != 0,
        .logicOp = static_cast<VkLogicOp>(dynamic.logic_op.Value()),
        .attachmentCount = static_cast<u32>(cb_attachments.size()),
        .pAttachments = cb_attachments.data(),
        .blendConstants = {}
    };
    static_vector<VkDynamicState, 34> dynamic_states{
        VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,         VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,       VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };
    if (key.state.extended_dynamic_state) {
        static constexpr std::array extended{
            VK_DYNAMIC_STATE_CULL_MODE_EXT,
            VK_DYNAMIC_STATE_FRONT_FACE_EXT,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_OP_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());

        // VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT
        if (!key.state.dynamic_vertex_input) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT);
        }
    }

    // VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
    if (key.state.dynamic_vertex_input) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
    }

    // EDS2 - Core (3 states)
    if (key.state.extended_dynamic_state_2) {
        static constexpr std::array extended2{
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended2.begin(), extended2.end());
    }

    // EDS2 - LogicOp (granular)
    if (key.state.extended_dynamic_state_2_logic_op) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_EXT);
    }

    // EDS3 - Blending (composite: 3 states)
    if (key.state.extended_dynamic_state_3_blend) {
        static constexpr std::array extended3{
            VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
            VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
            VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended3.begin(), extended3.end());
    }

    // VK_EXT_color_write_enable fallback for fully on/off render targets when EDS3 blending is not available.
    if (!key.state.extended_dynamic_state_3_blend && key.state.color_write_enable_dynamic) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT);
    }

    // EDS3 - Enables (composite: per-feature)
    if (key.state.extended_dynamic_state_3_enables) {
        if (device.SupportsDynamicState3DepthClampEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LogicOpEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LineRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3ConservativeRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3LineStippleEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToCoverageEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToOneEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT);
        }
    }

    const VkPipelineDynamicStateCreateInfo dynamic_state_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    [[maybe_unused]] const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .requiredSubgroupSize = GuestWarpSize,
    };
    static_vector<VkPipelineShaderStageCreateInfo, 5> shader_stages;
    for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        if (!spv_modules[stage]) {
            continue;
        }
        [[maybe_unused]] auto& stage_ci =
            shader_stages.emplace_back(VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = MaxwellToVK::ShaderStage(Shader::StageFromIndex(stage)),
                .module = *spv_modules[stage],
                .pName = "main",
                .pSpecializationInfo = nullptr,
            });
    }
    VkPipelineCreateFlags flags{};
    if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
        flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    if (uses_descriptor_buffer) {
        flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }

    pipeline = device.GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_ci,
        .pInputAssemblyState = &input_assembly_ci,
        .pTessellationState = &tessellation_ci,
        .pViewportState = &viewport_ci,
        .pRasterizationState = &rasterization_ci,
        .pMultisampleState = &multisample_ci,
        .pDepthStencilState = &depth_stencil_ci,
        .pColorBlendState = &color_blend_ci,
        .pDynamicState = &dynamic_state_ci,
        .layout = *pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    }, *pipeline_cache);

    // Log graphics pipeline creation
    if (GPU::Logging::IsActive()) {
        const std::string pipeline_info = fmt::format(
            "GraphicsPipeline created: stages={}, attachments={}",
            shader_stages.size(),
            color_blend_ci.attachmentCount
        );
        GPU::Logging::GPULogger::GetInstance().LogPipelineStateChange(pipeline_info);
    }
}

void GraphicsPipeline::Validate() {
    size_t num_images{};
    for (const auto& info : stage_infos) {
        num_images += Shader::NumDescriptors(info.texture_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.image_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.texture_descriptors);
        num_images += Shader::NumDescriptors(info.image_descriptors);
    }
    ASSERT(num_images == num_image_elements);
}

} // namespace Vulkan
