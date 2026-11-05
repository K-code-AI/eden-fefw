// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdint>

namespace VideoCommon::NativeDrawOptimizations {
// Temporary controls for same-scene performance comparisons.
inline std::atomic<bool> skip_unchanged_state{true};
inline std::atomic<bool> reuse_usage_words{true};
inline std::atomic<std::uint32_t> cache_lock_spins{0};
inline std::atomic<bool> reuse_sparse_uniforms{true};
inline std::atomic<bool> batch_texture_handle_runs{true};
inline std::atomic<bool> optimize_macro_tail{true};
inline std::atomic<bool> fast_gpu_range_queries{true};
inline std::atomic<bool> short_macro_parameters{true};
inline std::atomic<bool> guard_macro_debug{true};
inline std::atomic<bool> batch_vulkan_work{true};
inline std::atomic<bool> gpu_split_enabled{true};
inline std::atomic<bool> park_front_waits{true};
inline std::atomic<bool> batch_native_registers{true};
inline std::atomic<std::uint32_t> front_verify_interval{1};
inline std::atomic<bool> defer_front_verification{true};
inline std::atomic<bool> batch_shader_invalidation{true};
inline std::atomic<bool> flat_sampler_lookup{true};
inline std::atomic<bool> compact_fence_flushes{false};
inline std::atomic<bool> vector_texture_handle_scan{false};
inline std::atomic<bool> tight_texture_loops{true};
// Recomputes each tight-loop result with the original loop and counts differences.
inline std::atomic<bool> check_texture_loops{false};
inline std::atomic<std::uint64_t> texture_array_checks{0};
inline std::atomic<std::uint64_t> texture_array_mismatches{0};
// Checks where guest memory changed between the two passes; not loop differences.
inline std::atomic<std::uint64_t> texture_array_unstable{0};
inline std::atomic<bool> thread_contiguous_runs{true};
inline std::atomic<bool> skip_empty_rejections{true};
// Recomputes contiguous sizes and rejection emptiness under the original locks.
inline std::atomic<bool> check_uniform_locks{false};
inline std::atomic<std::uint64_t> contiguous_checks{0};
inline std::atomic<std::uint64_t> contiguous_mismatches{0};
// Checks where a GPU mapping changed between the two lookups; not cache differences.
inline std::atomic<std::uint64_t> contiguous_unstable{0};
inline std::atomic<std::uint64_t> rejection_checks{0};
inline std::atomic<std::uint64_t> rejection_mismatches{0};
} // namespace VideoCommon::NativeDrawOptimizations
