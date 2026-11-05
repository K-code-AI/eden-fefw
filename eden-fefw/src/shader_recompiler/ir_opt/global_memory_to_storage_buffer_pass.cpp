// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <optional>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/breadth_first_search.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
/// Address in constant buffers to the storage buffer descriptor
struct StorageBufferAddr {
    auto operator<=>(const StorageBufferAddr&) const noexcept = default;

    u32 index;
    u32 offset;
};

/// Block iterator to a global memory instruction and the storage buffer it uses
struct StorageInst {
    StorageBufferAddr storage_buffer;
    IR::Inst* inst;
    IR::Block* block;
};

/// Bias towards a certain range of constant buffers when looking for storage buffers
struct Bias {
    u32 index;
    u32 offset_begin;
    u32 offset_end;
    u32 alignment;
};

using boost::container::flat_set;
using boost::container::small_vector;
using StorageBufferSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;
using StorageInstVector = small_vector<StorageInst, 24>;
using StorageWritesSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;

struct StorageInfo {
    StorageBufferSet set;
    StorageInstVector to_replace;
    StorageWritesSet writes;
    bool native_runtime_bounds{};
};

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemory(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemoryWrite(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Converts a global memory opcode to its storage buffer equivalent
IR::Opcode GlobalToStorage(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadGlobalS8:
        return IR::Opcode::LoadStorageS8;
    case IR::Opcode::LoadGlobalU8:
        return IR::Opcode::LoadStorageU8;
    case IR::Opcode::LoadGlobalS16:
        return IR::Opcode::LoadStorageS16;
    case IR::Opcode::LoadGlobalU16:
        return IR::Opcode::LoadStorageU16;
    case IR::Opcode::LoadGlobal32:
        return IR::Opcode::LoadStorage32;
    case IR::Opcode::LoadGlobal64:
        return IR::Opcode::LoadStorage64;
    case IR::Opcode::LoadGlobal128:
        return IR::Opcode::LoadStorage128;
    case IR::Opcode::WriteGlobalS8:
        return IR::Opcode::WriteStorageS8;
    case IR::Opcode::WriteGlobalU8:
        return IR::Opcode::WriteStorageU8;
    case IR::Opcode::WriteGlobalS16:
        return IR::Opcode::WriteStorageS16;
    case IR::Opcode::WriteGlobalU16:
        return IR::Opcode::WriteStorageU16;
    case IR::Opcode::WriteGlobal32:
        return IR::Opcode::WriteStorage32;
    case IR::Opcode::WriteGlobal64:
        return IR::Opcode::WriteStorage64;
    case IR::Opcode::WriteGlobal128:
        return IR::Opcode::WriteStorage128;
    case IR::Opcode::GlobalAtomicIAdd32:
        return IR::Opcode::StorageAtomicIAdd32;
    case IR::Opcode::GlobalAtomicSMin32:
        return IR::Opcode::StorageAtomicSMin32;
    case IR::Opcode::GlobalAtomicUMin32:
        return IR::Opcode::StorageAtomicUMin32;
    case IR::Opcode::GlobalAtomicSMax32:
        return IR::Opcode::StorageAtomicSMax32;
    case IR::Opcode::GlobalAtomicUMax32:
        return IR::Opcode::StorageAtomicUMax32;
    case IR::Opcode::GlobalAtomicInc32:
        return IR::Opcode::StorageAtomicInc32;
    case IR::Opcode::GlobalAtomicDec32:
        return IR::Opcode::StorageAtomicDec32;
    case IR::Opcode::GlobalAtomicAnd32:
        return IR::Opcode::StorageAtomicAnd32;
    case IR::Opcode::GlobalAtomicOr32:
        return IR::Opcode::StorageAtomicOr32;
    case IR::Opcode::GlobalAtomicXor32:
        return IR::Opcode::StorageAtomicXor32;
    case IR::Opcode::GlobalAtomicExchange32:
        return IR::Opcode::StorageAtomicExchange32;
    case IR::Opcode::GlobalAtomicIAdd64:
        return IR::Opcode::StorageAtomicIAdd64;
    case IR::Opcode::GlobalAtomicSMin64:
        return IR::Opcode::StorageAtomicSMin64;
    case IR::Opcode::GlobalAtomicUMin64:
        return IR::Opcode::StorageAtomicUMin64;
    case IR::Opcode::GlobalAtomicSMax64:
        return IR::Opcode::StorageAtomicSMax64;
    case IR::Opcode::GlobalAtomicUMax64:
        return IR::Opcode::StorageAtomicUMax64;
    case IR::Opcode::GlobalAtomicAnd64:
        return IR::Opcode::StorageAtomicAnd64;
    case IR::Opcode::GlobalAtomicOr64:
        return IR::Opcode::StorageAtomicOr64;
    case IR::Opcode::GlobalAtomicXor64:
        return IR::Opcode::StorageAtomicXor64;
    case IR::Opcode::GlobalAtomicExchange64:
        return IR::Opcode::StorageAtomicExchange64;
    case IR::Opcode::GlobalAtomicIAdd32x2:
        return IR::Opcode::StorageAtomicIAdd32x2;
    case IR::Opcode::GlobalAtomicSMin32x2:
        return IR::Opcode::StorageAtomicSMin32x2;
    case IR::Opcode::GlobalAtomicUMin32x2:
        return IR::Opcode::StorageAtomicUMin32x2;
    case IR::Opcode::GlobalAtomicSMax32x2:
        return IR::Opcode::StorageAtomicSMax32x2;
    case IR::Opcode::GlobalAtomicUMax32x2:
        return IR::Opcode::StorageAtomicUMax32x2;
    case IR::Opcode::GlobalAtomicAnd32x2:
        return IR::Opcode::StorageAtomicAnd32x2;
    case IR::Opcode::GlobalAtomicOr32x2:
        return IR::Opcode::StorageAtomicOr32x2;
    case IR::Opcode::GlobalAtomicXor32x2:
        return IR::Opcode::StorageAtomicXor32x2;
    case IR::Opcode::GlobalAtomicExchange32x2:
        return IR::Opcode::StorageAtomicExchange32x2;
    case IR::Opcode::GlobalAtomicAddF32:
        return IR::Opcode::StorageAtomicAddF32;
    case IR::Opcode::GlobalAtomicAddF16x2:
        return IR::Opcode::StorageAtomicAddF16x2;
    case IR::Opcode::GlobalAtomicMinF16x2:
        return IR::Opcode::StorageAtomicMinF16x2;
    case IR::Opcode::GlobalAtomicMaxF16x2:
        return IR::Opcode::StorageAtomicMaxF16x2;
    case IR::Opcode::GlobalAtomicAddF32x2:
        return IR::Opcode::StorageAtomicAddF32x2;
    case IR::Opcode::GlobalAtomicMinF32x2:
        return IR::Opcode::StorageAtomicMinF32x2;
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return IR::Opcode::StorageAtomicMaxF32x2;
    default:
        throw InvalidArgument("Invalid global memory opcode {}", opcode);
    }
}

/// Returns true when a storage buffer address satisfies a bias
bool MeetsBias(const StorageBufferAddr& storage_buffer, const Bias& bias) noexcept {
    return storage_buffer.index == bias.index && storage_buffer.offset >= bias.offset_begin &&
           storage_buffer.offset < bias.offset_end;
}

struct LowAddrInfo {
    IR::U32 value;
    s32 imm_offset;
};

/// Tries to track the first 32-bits of a global memory instruction
std::optional<LowAddrInfo> TrackLowAddress(IR::Inst* inst) {
    // The first argument is the low level GPU pointer to the global memory instruction
    const IR::Value addr{inst->Arg(0)};
    if (addr.IsImmediate()) {
        // Not much we can do if it's an immediate
        return std::nullopt;
    }
    // This address is expected to either be a PackUint2x32, a IAdd64, or a CompositeConstructU32x2
    IR::Inst* addr_inst{addr.InstRecursive()};
    s32 imm_offset{0};
    if (addr_inst->GetOpcode() == IR::Opcode::IAdd64) {
        // If it's an IAdd64, get the immediate offset it is applying and grab the address
        // instruction. This expects for the instruction to be canonicalized having the address on
        // the first argument and the immediate offset on the second one.
        const IR::U64 imm_offset_value{addr_inst->Arg(1)};
        if (!imm_offset_value.IsImmediate()) {
            return std::nullopt;
        }
        imm_offset = static_cast<s32>(static_cast<s64>(imm_offset_value.U64()));
        const IR::U64 iadd_addr{addr_inst->Arg(0)};
        if (iadd_addr.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = iadd_addr.InstRecursive();
    }
    // With IAdd64 handled, now PackUint2x32 is expected
    if (addr_inst->GetOpcode() == IR::Opcode::PackUint2x32) {
        // PackUint2x32 is expected to be generated from a vector
        const IR::Value vector{addr_inst->Arg(0)};
        if (vector.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = vector.InstRecursive();
    }
    // The vector is expected to be a CompositeConstructU32x2
    if (addr_inst->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return std::nullopt;
    }
    // Grab the first argument from the CompositeConstructU32x2, this is the low address.
    return LowAddrInfo{
        .value{IR::U32{addr_inst->Arg(0)}},
        .imm_offset = imm_offset,
    };
}

/// Tries to track the storage buffer address used by a global memory instruction
std::optional<StorageBufferAddr> Track(const IR::Value& value, const Bias* bias, Environment* native_env = nullptr,
                                       small_vector<StorageBufferAddr, 4>* candidates = nullptr) {
    const auto pred{[bias, native_env](const IR::Inst* inst) -> std::optional<StorageBufferAddr> {
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32 &&
            inst->GetOpcode() != IR::Opcode::GetCbufU32x2) {
            return std::nullopt;
        }
        const IR::Value index{inst->Arg(0)};
        const IR::Value offset{inst->Arg(1)};
        if (!index.IsImmediate()) {
            // Definitely not a storage buffer if it's read from a
            // non-immediate index
            return std::nullopt;
        }
        if (!offset.IsImmediate()) {
            // TODO: Support SSBO arrays
            return std::nullopt;
        }
        const StorageBufferAddr storage_buffer{
            .index = index.U32(),
            .offset = offset.U32(),
        };
        const u32 alignment{bias ? bias->alignment : 8U};
        if (!Common::IsAligned(storage_buffer.offset, alignment)) {
            // The SSBO pointer has to be aligned
            return std::nullopt;
        }
        if (bias && !MeetsBias(storage_buffer, *bias)) {
            // We have to blacklist some addresses in case we wrongly
            // point to them
            return std::nullopt;
        }
        if (native_env && !native_env->IsCbufStorageBufferPointer(storage_buffer.index,
                                                                 storage_buffer.offset)) {
            return std::nullopt;
        }
        return storage_buffer;
    }};
    if (!native_env) {
        return BreadthFirstSearch(value, pred);
    }
    // A base pointer survives addition and selection, but not multiplication or a
    // predicate calculation. Searching those expressions can find strides first.
    small_vector<const IR::Inst*, 16> pending;
    const auto enqueue = [&](const IR::Value& candidate) {
        if (candidate.IsImmediate()) return;
        const auto* candidate_inst = candidate.InstRecursive();
        if (std::ranges::find(pending, candidate_inst) == pending.end()) {
            pending.push_back(candidate_inst);
        }
    };
    enqueue(value);
    // Keep searching after the first pointer: a phi of a predicated address (for example a register
    // that held another buffer's address before) can reach several pointers.
    std::optional<StorageBufferAddr> first;
    small_vector<StorageBufferAddr, 4> others;
    for (size_t next = 0; next < pending.size(); ++next) {
        const auto* inst = pending[next];
        if (const auto result = pred(inst)) {
            if (!first) {
                first = result;
            } else if (result->offset != first->offset &&
                       std::ranges::find_if(others, [&](const auto& o) { return o.offset == result->offset; }) ==
                           others.end()) {
                others.push_back(*result);
            }
            continue;
        }
        switch (inst->GetOpcode()) {
        case IR::Opcode::IAdd32:
        case IR::Opcode::Phi:
            for (size_t arg = inst->NumArgs(); arg--;) enqueue(inst->Arg(arg));
            break;
        case IR::Opcode::SelectU32:
            enqueue(inst->Arg(1));
            enqueue(inst->Arg(2));
            break;
        case IR::Opcode::CompositeExtractU32x2:
            if (inst->Arg(1).IsImmediate() && inst->Arg(1).U32() == 0) {
                enqueue(inst->Arg(0));
            }
            break;
        default:
            break;
        }
    }
    if (candidates && first) {
        candidates->push_back(*first);
        candidates->insert(candidates->end(), others.begin(), others.end());
    }
    return first;
}

/// Collects the storage buffer used by a global memory instruction and the instruction itself
void CollectStorageBuffers(IR::Block& block, IR::Inst& inst, StorageInfo& info, Environment& env) {
    // NVN puts storage buffers in a specific range, we have to bias towards these addresses to
    // avoid getting false positives
    static constexpr Bias nvn_bias{
        .index = 0,
        .offset_begin = 0x100,
        .offset_end = 0x700,
        .alignment = 16,
    };
    // Track the low address of the instruction
    const std::optional<LowAddrInfo> low_addr_info{TrackLowAddress(&inst)};
    if (!low_addr_info) {
        // Failed to track the low address, use NVN fallbacks
        return;
    }
    const bool native = env.HasNativeRegisterCopies();
    // First try to find storage buffers in the NVN address
    const IR::U32 low_addr{low_addr_info->value};
    // Native kernels use eight-byte pointer arguments, without NVN descriptor spacing.
    // Applying the NVN bias can skip the pointer and select a dimension or stride instead.
    small_vector<StorageBufferAddr, 4> candidates;
    std::optional<StorageBufferAddr> storage_buffer{
        Track(low_addr, native ? nullptr : &nvn_bias, native ? &env : nullptr, native ? &candidates : nullptr)};
    if (native && candidates.size() > 1) {
        // The address may come from any of several pointers, so a single storage buffer cannot be
        // chosen statically. Bind them all and keep the access global: the backend then picks the
        // buffer whose bound range contains the address at run time.
        for (const StorageBufferAddr& candidate : candidates) {
            info.set.insert(candidate);
            if (IsGlobalMemoryWrite(inst)) {
                info.writes.insert(candidate);
            }
        }
        info.native_runtime_bounds = true;
        return;
    }
    if (!storage_buffer) {
        // If it fails, track without a bias
        storage_buffer = Track(low_addr, nullptr, native ? &env : nullptr);
        if (!storage_buffer) {
            // If that also fails, use NVN fallbacks
            LOG_WARNING(Shader, "Storage buffer failed to track, using global memory fallbacks");
            return;
        }
        LOG_DEBUG(Shader, "Storage buffer tracked without bias, index {} offset {}",
                    storage_buffer->index, storage_buffer->offset);
    }
    // Collect storage buffer and the instruction
    if (IsGlobalMemoryWrite(inst)) {
        info.writes.insert(*storage_buffer);
    }
    info.set.insert(*storage_buffer);
    info.to_replace.push_back(StorageInst{
        .storage_buffer{*storage_buffer},
        .inst = &inst,
        .block = &block,
    });
}

/// Returns the offset in indices (not bytes) for an equivalent storage instruction
IR::U32 StorageOffset(IR::Block& block, IR::Inst& inst, StorageBufferAddr buffer, u32 alignment) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    IR::U32 offset;
    if (const std::optional<LowAddrInfo> low_addr{TrackLowAddress(&inst)}) {
        offset = low_addr->value;
        if (low_addr->imm_offset != 0) {
            offset = ir.IAdd(offset, ir.Imm32(low_addr->imm_offset));
        }
    } else {
        offset = ir.UConvert(32, IR::U64{inst.Arg(0)});
    }
    // Subtract the least significant 32 bits from the guest offset. The result is the storage
    // buffer offset in bytes.
    IR::U32 low_cbuf{ir.GetCbuf(ir.Imm32(buffer.index), ir.Imm32(buffer.offset))};

    // Align the offset base to match the host alignment requirements
    low_cbuf = ir.BitwiseAnd(low_cbuf, ir.Imm32(~(alignment - 1U)));
    return ir.ISub(offset, low_cbuf);
}


// Express a low address as one storage pointer plus a bounded integer offset. Arithmetic is
// modulo 2^32, so the pointer may wrap, but pointer-free intermediates must not wrap. Unknown
// operations, loops, workgroup IDs and competing pointers deliberately retain the broad fallback.
struct AddressRange {
    bool pointer{};
    s64 min{};
    s64 max{};
};

std::optional<AddressRange> BoundAddress(IR::Value value, StorageBufferAddr buffer,
                                       const std::array<u32, 3>& workgroup, u32 depth = 0,
                                       bool low_word = false) {
    value = value.Resolve();
    if (depth >= 64 || (value.Type() != IR::Type::U32 &&
                        !(low_word && value.Type() == IR::Type::U64))) {
        return std::nullopt;
    }
    if (value.IsImmediate()) {
        const u32 word = value.Type() == IR::Type::U64 ? static_cast<u32>(value.U64()) : value.U32();
        return AddressRange{false, word, word};
    }
    const auto* inst = value.InstRecursive();
    const auto bound = [&](IR::Value arg) { return BoundAddress(arg, buffer, workgroup, depth + 1, low_word); };
    const auto matches_pointer = [&](const IR::Inst& read, u32 byte_offset) {
        return read.Arg(0).IsImmediate() && read.Arg(1).IsImmediate() &&
               read.Arg(0).U32() == buffer.index &&
               u64{read.Arg(1).U32()} + byte_offset == buffer.offset;
    };
    switch (inst->GetOpcode()) {
    case IR::Opcode::PackUint2x32: {
        const auto* vector = inst->Arg(0).TryInstRecursive();
        if (vector && vector->GetOpcode() == IR::Opcode::CompositeConstructU32x2) {
            return bound(vector->Arg(0));
        }
        break;
    }
    case IR::Opcode::GetCbufU32:
        if (matches_pointer(*inst, 0)) {
            return AddressRange{true, 0, 0};
        }
        break;
    case IR::Opcode::CompositeExtractU32x2:
    case IR::Opcode::CompositeExtractU32x3:
    case IR::Opcode::CompositeExtractU32x4: {
        if (!inst->Arg(1).IsImmediate() || inst->Arg(0).IsImmediate()) {
            break;
        }
        const u32 component = inst->Arg(1).U32();
        const auto* vector = inst->Arg(0).InstRecursive();
        if (component == 0 && vector->GetOpcode() == IR::Opcode::UnpackUint2x32) {
            return BoundAddress(vector->Arg(0), buffer, workgroup, depth + 1, true);
        }
        if (vector->GetOpcode() == IR::Opcode::LocalInvocationId && component < 3 &&
            workgroup[component] != 0) {
            return AddressRange{false, 0, workgroup[component] - 1};
        }
        if (vector->GetOpcode() == IR::Opcode::GetCbufU32x2 && component < 2 &&
            matches_pointer(*vector, component * 4)) {
            return AddressRange{true, 0, 0};
        }
        if ((vector->GetOpcode() == IR::Opcode::CompositeConstructU32x2 ||
             vector->GetOpcode() == IR::Opcode::CompositeConstructU32x3 ||
             vector->GetOpcode() == IR::Opcode::CompositeConstructU32x4) &&
            component < vector->NumArgs()) {
            return bound(vector->Arg(component));
        }
        break;
    }
    case IR::Opcode::IAdd32:
    case IR::Opcode::IAdd64:
    case IR::Opcode::ISub32: {
        const auto a = bound(inst->Arg(0));
        const auto b = bound(inst->Arg(1));
        // Only add a bounded scalar to a pointer; never cancel independently selected pointers.
        if (!a || !b || (a->pointer && b->pointer) ||
            (inst->GetOpcode() == IR::Opcode::ISub32 && b->pointer)) {
            break;
        }
        const bool subtract = inst->GetOpcode() == IR::Opcode::ISub32;
        AddressRange result{a->pointer || b->pointer,
                            a->min + (subtract ? -b->max : b->min),
                            a->max + (subtract ? -b->min : b->max)};
        const s64 limit = std::numeric_limits<u32>::max();
        if (result.min >= (result.pointer ? -limit : 0) && result.max <= limit) {
            return result;
        }
        break;
    }
    case IR::Opcode::IMul32:
    case IR::Opcode::ShiftLeftLogical32: {
        const auto a = bound(inst->Arg(0));
        auto b = bound(inst->Arg(1));
        if (!a || !b || a->pointer || b->pointer || a->min < 0 || b->min < 0) {
            break;
        }
        if (inst->GetOpcode() == IR::Opcode::ShiftLeftLogical32) {
            if (b->min != b->max || b->max >= 32) {
                break;
            }
            b = AddressRange{false, s64{1} << b->min, s64{1} << b->max};
        }
        const u64 maximum = u64(a->max) * u64(b->max);
        if (maximum <= std::numeric_limits<u32>::max()) {
            return AddressRange{false, a->min * b->min, static_cast<s64>(maximum)};
        }
        break;
    }
    case IR::Opcode::BitFieldUExtract: {
        const auto shift = inst->Arg(1).Resolve();
        const auto width = inst->Arg(2).Resolve();
        if (!shift.IsImmediate() || !width.IsImmediate() || shift.U32() >= 32 ||
            width.U32() > 32 - shift.U32()) {
            break;
        }
        const u32 mask = static_cast<u32>((u64{1} << width.U32()) - 1);
        const auto source = bound(inst->Arg(0));
        if (source && !source->pointer && source->min >= 0) {
            const u32 first = static_cast<u32>(source->min) >> shift.U32();
            const u32 last = static_cast<u32>(source->max) >> shift.U32();
            if ((first & ~mask) == (last & ~mask)) {
                return AddressRange{false, first & mask, last & mask};
            }
        }
        return AddressRange{false, 0, mask};
    }
    case IR::Opcode::BitwiseAnd32:
        for (size_t index = 0; index < 2; ++index) {
            if (inst->Arg(index).IsImmediate()) {
                return AddressRange{false, 0, inst->Arg(index).U32()};
            }
        }
        break;
    case IR::Opcode::SelectU32: {
        const auto a = bound(inst->Arg(1));
        const auto b = bound(inst->Arg(2));
        if (a && b && a->pointer == b->pointer) {
            return AddressRange{a->pointer, std::min(a->min, b->min), std::max(a->max, b->max)};
        }
        break;
    }
    default:
        break;
    }
    return std::nullopt;
}

u32 StorageSizeBound(const StorageInfo& info, StorageBufferAddr buffer,
                     const std::array<u32, 3>& workgroup, u64 alignment) {
    if (alignment == 0 || alignment > std::numeric_limits<u32>::max()) {
        return 0;
    }
    u64 size = 0;
    for (const auto& access : info.to_replace) {
        if (access.storage_buffer != buffer) {
            continue;
        }
        const auto low = TrackLowAddress(access.inst);
        if (!low) {
            return 0;
        }
        const auto range = BoundAddress(low->value, buffer, workgroup);
        if (!range || !range->pointer || range->min + low->imm_offset < 0) {
            return 0;
        }
        // Every global load/store/atomic handled by this pass touches at most 16 bytes. Include
        // the largest possible alignment prefix because the actual pointer is only known at bind.
        const u64 end = u64(range->max + low->imm_offset) + 16 + alignment - 1;
        if (end > std::numeric_limits<u32>::max()) {
            return 0;
        }
        size = std::max(size, end);
    }
    return static_cast<u32>(size);
}

/// Replace a global memory load instruction with its storage buffer equivalent
void ReplaceLoad(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                 const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{&*block.PrependNewInst(it, new_opcode, {storage_index, offset})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory write instruction with its storage buffer equivalent
void ReplaceWrite(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                  const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)});
    inst.Invalidate();
}

/// Replace an atomic operation on global memory instruction with its storage buffer equivalent
void ReplaceAtomic(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                   const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{
        &*block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory instruction with its storage buffer equivalent
void Replace(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
             const IR::U32& offset) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
        return ReplaceLoad(block, inst, storage_index, offset);
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
        return ReplaceWrite(block, inst, storage_index, offset);
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return ReplaceAtomic(block, inst, storage_index, offset);
    default:
        throw InvalidArgument("Invalid global memory opcode {}", inst.GetOpcode());
    }
}
} // Anonymous namespace

void GlobalMemoryToStorageBufferPass(Environment& env, IR::Program& program, const HostTranslateInfo& host_info) {
    StorageInfo info;
    size_t global_accesses = 0;
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsGlobalMemory(inst)) {
                continue;
            }
            ++global_accesses;
            CollectStorageBuffers(*block, inst, info, env);
        }
    }
    program.info.native_global_bounds |= info.native_runtime_bounds;
    const bool can_bound = env.HasNativeRegisterCopies() &&
                           env.ShaderStage() == Stage::Compute &&
                           !info.native_runtime_bounds && global_accesses == info.to_replace.size();
    for (const StorageBufferAddr& storage_buffer : info.set) {
        program.info.storage_buffers_descriptors.push_back({
            .cbuf_index = storage_buffer.index,
            .cbuf_offset = storage_buffer.offset,
            .count = 1,
            .is_written = info.writes.contains(storage_buffer),
            .size_bound = can_bound ? StorageSizeBound(info, storage_buffer, env.WorkgroupSize(),
                                                      host_info.min_ssbo_alignment) : 0,
        });
    }
    for (const StorageInst& storage_inst : info.to_replace) {
        const StorageBufferAddr storage_buffer{storage_inst.storage_buffer};
        const auto it{info.set.find(storage_inst.storage_buffer)};
        const IR::U32 index{IR::Value{static_cast<u32>(info.set.index_of(it))}};
        IR::Block* const block{storage_inst.block};
        IR::Inst* const inst{storage_inst.inst};
        const IR::U32 offset{
            StorageOffset(*block, *inst, storage_buffer, u32(host_info.min_ssbo_alignment))};
        Replace(*block, *inst, index, offset);
    }
}

void JoinStorageInfo(Info& base, Info& source) {
    base.native_global_bounds |= source.native_global_bounds;
    auto& descriptors = base.storage_buffers_descriptors;
    for (auto& desc : source.storage_buffers_descriptors) {
        auto it{std::ranges::find_if(descriptors, [&desc](const auto& existing) {
            return desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count;
        })};
        if (it != descriptors.end()) {
            it->is_written |= desc.is_written;
            it->size_bound = it->size_bound && desc.size_bound
                                 ? std::max(it->size_bound, desc.size_bound) : 0;
            continue;
        }
        descriptors.push_back(desc);
    }
}

} // namespace Shader::Optimization
