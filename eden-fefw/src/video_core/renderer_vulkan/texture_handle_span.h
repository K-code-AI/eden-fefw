// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Vulkan {

// Valid for one descriptor array only; resolve the current mapping again on every draw.
class TextureHandleSpan {
public:
    template <typename Memory>
    bool Prepare(Memory& memory, std::uint64_t primary_base, std::uint32_t primary_offset,
                 std::uint64_t secondary_base, std::uint32_t secondary_offset,
                 std::uint32_t count, std::uint32_t size_shift, bool has_secondary,
                 std::uint32_t primary_shift, std::uint32_t secondary_shift) {
        primary = secondary = nullptr;
        if (!count || size_shift >= 32 || (has_secondary &&
            (primary_shift >= 32 || secondary_shift >= 32))) {
            return false;
        }
        const std::uint64_t last_offset = std::uint64_t{count - 1} << size_shift;
        const std::uint64_t span_size = last_offset + sizeof(std::uint32_t);
        // Keep wrapping 32-bit constant-buffer offsets on the general reader.
        if (span_size > 65536 || last_offset > UINT32_MAX - primary_offset ||
            (has_secondary && last_offset > UINT32_MAX - secondary_offset)) {
            return false;
        }
        const auto resolve = [&](std::uint64_t base, std::uint32_t offset) {
            if (base > UINT64_MAX - offset || base + offset > UINT64_MAX - span_size) {
                return static_cast<const std::uint8_t*>(nullptr);
            }
            return static_cast<const std::uint8_t*>(memory.GetSpan(base + offset, span_size));
        };
        primary = resolve(primary_base, primary_offset);
        if (!primary) {
            return false;
        }
        if (has_secondary) {
            secondary = resolve(secondary_base, secondary_offset);
            if (!secondary) {
                primary = nullptr;
                return false;
            }
        }
        stride_shift = size_shift;
        lhs_shift = primary_shift;
        rhs_shift = secondary_shift;
        return true;
    }

    std::uint32_t Read(std::uint32_t index) const noexcept {
        const std::size_t offset = std::size_t{index} << stride_shift;
        std::uint32_t lhs;
        std::memcpy(&lhs, primary + offset, sizeof(lhs));
        if (!secondary) {
            return lhs;
        }
        std::uint32_t rhs;
        std::memcpy(&rhs, secondary + offset, sizeof(rhs));
        return (lhs << lhs_shift) | (rhs << rhs_shift);
    }

    // Read each current handle once; only coalesce adjacent equal values in this array.
    template <typename Emit>
    void ForEachRun(std::uint32_t count, Emit emit, bool vector_scan = false) const {
        if (!count) return;
#if defined(_M_X64) || defined(__x86_64__)
        if (vector_scan && stride_shift == 2 && count >= 64 && PreferVectorScan(count)) {
            if (secondary) {
                ForEachRunVector<true>(count, emit);
            } else {
                ForEachRunVector<false>(count, emit);
            }
            return;
        }
#else
        (void)vector_scan;
#endif
        std::uint32_t start = 0;
        std::uint32_t value = Read(0);
        for (std::uint32_t index = 1; index < count; ++index) {
            const std::uint32_t next = Read(index);
            if (next == value) continue;
            emit(start, index - start, value);
            start = index;
            value = next;
        }
        emit(start, count - start, value);
    }

    // Packed 32-bit handles that Read(index) loads from offset index * 4, or null.
    const std::uint8_t* PackedWords() const noexcept {
        return !secondary && stride_shift == 2 ? primary : nullptr;
    }

private:
#if defined(_M_X64) || defined(__x86_64__)
    bool PreferVectorScan(std::uint32_t count) const noexcept {
        // Select an algorithm only; none of these probe values are retained or emitted.
        const auto middle = count / 2 - 8;
        const unsigned matches = unsigned(Read(0) == Read(15)) +
                                 unsigned(Read(middle) == Read(middle + 15));
        if (matches != 1) return matches == 2;
        return Read(count - 16) == Read(count - 1);
    }

#if defined(__AVX512F__)
    static constexpr std::uint32_t VectorWords = 16;
#elif defined(__AVX2__)
    static constexpr std::uint32_t VectorWords = 8;
#else
    static constexpr std::uint32_t VectorWords = 4;
#endif

    template <bool HasSecondary>
    std::uint32_t EqualPrefix(std::uint32_t index, std::uint32_t value) const noexcept {
        const auto offset = std::size_t{index} * sizeof(std::uint32_t);
#if defined(__AVX512F__)
        auto current = _mm512_loadu_si512(primary + offset);
        if constexpr (HasSecondary) {
            const auto other = _mm512_loadu_si512(secondary + offset);
            current = _mm512_or_si512(
                _mm512_sll_epi32(current, _mm_cvtsi32_si128(lhs_shift)),
                _mm512_sll_epi32(other, _mm_cvtsi32_si128(rhs_shift)));
        }
        const auto equal = _mm512_cmpeq_epi32_mask(current, _mm512_set1_epi32(std::bit_cast<int>(value)));
        const auto different = static_cast<std::uint32_t>(~equal) & 0xffffU;
#elif defined(__AVX2__)
        auto current = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(primary + offset));
        if constexpr (HasSecondary) {
            const auto other = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(secondary + offset));
            current = _mm256_or_si256(
                _mm256_sll_epi32(current, _mm_cvtsi32_si128(lhs_shift)),
                _mm256_sll_epi32(other, _mm_cvtsi32_si128(rhs_shift)));
        }
        const auto equal = _mm256_cmpeq_epi32(current, _mm256_set1_epi32(std::bit_cast<int>(value)));
        const auto different = static_cast<std::uint32_t>(~_mm256_movemask_ps(_mm256_castsi256_ps(equal))) & 0xffU;
#else
        auto current = _mm_loadu_si128(reinterpret_cast<const __m128i*>(primary + offset));
        if constexpr (HasSecondary) {
            const auto other = _mm_loadu_si128(reinterpret_cast<const __m128i*>(secondary + offset));
            current = _mm_or_si128(
                _mm_sll_epi32(current, _mm_cvtsi32_si128(lhs_shift)),
                _mm_sll_epi32(other, _mm_cvtsi32_si128(rhs_shift)));
        }
        const auto equal = _mm_cmpeq_epi32(current, _mm_set1_epi32(std::bit_cast<int>(value)));
        const auto different = static_cast<std::uint32_t>(~_mm_movemask_ps(_mm_castsi128_ps(equal))) & 0xfU;
#endif
        return different ? static_cast<std::uint32_t>(std::countr_zero(different)) : VectorWords;
    }

    template <bool HasSecondary, typename Emit>
    void ForEachRunVector(std::uint32_t count, Emit& emit) const {
        std::uint32_t start = 0;
        std::uint32_t value = Read(0);
        std::uint32_t index = 1;
        while (index < count) {
            const auto next = Read(index);
            if (next != value) {
                // Discard every lookahead value after this boundary. The callback may
                // update later guest handles, which must be read again before use.
                emit(start, index - start, value);
                start = index;
                value = next;
            } else if (index - start >= 4 && count - index >= VectorWords) {
                // Test the current word first so a short run ending here does not pay
                // for a vector comparison that cannot advance it.
                const auto equal = EqualPrefix<HasSecondary>(index, value);
                index += equal != 0 ? equal : 1;
                continue;
            }
            ++index;
        }
        emit(start, count - start, value);
    }
#endif

    const std::uint8_t* primary{};
    const std::uint8_t* secondary{};
    std::uint32_t stride_shift{};
    std::uint32_t lhs_shift{};
    std::uint32_t rhs_shift{};
};

} // namespace Vulkan
