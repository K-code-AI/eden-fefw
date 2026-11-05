// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <optional>
#include <span>
#include "shader_recompiler/frontend/ampere/instruction.h"

namespace Shader::Ampere {
inline bool GeometryInstructionMatches(const Instruction& instruction, u32 a, u32 b, u32 c) {
    // The fourth word holds scheduling controls, not these instruction operands.
    return instruction.words[0] == a && instruction.words[1] == b && instruction.words[2] == c;
}

// Recognize the compiler's LEPC/MOV/IADD3/IADD3.X/BMOV.64 ATEXIT_PC setup.
// LEPC at instruction zero supplies this instruction's address. Its added byte
// displacement points at the exit handler, rather than at an ordinary CALL.
inline std::optional<u32> GeometryExitTarget(std::span<const Instruction> code) {
    if (code.size() < 5 ||
        !GeometryInstructionMatches(code[0], 0x0000734e, 0, 0) ||
        !GeometryInstructionMatches(code[1], 0x00037802, code[1].words[1], 0x00000f00) ||
        !GeometryInstructionMatches(code[2], 0x00007210, 3, 0x07f1e0ff) ||
        !GeometryInstructionMatches(code[3], 0x01017210, 0xff, 0x007fe4ff) ||
        !GeometryInstructionMatches(code[4], 0x00007357, 0, 0)) {
        return {};
    }
    return code[1].words[1];
}

inline bool HasTrivialGeometryExit(std::span<const Instruction> code) {
    const auto target = GeometryExitTarget(code);
    if (!target || *target % sizeof(Instruction) || *target / sizeof(Instruction) < 12 ||
        u64{*target} + 3 * sizeof(Instruction) > code.size_bytes()) return false;
    const u32 at = *target / sizeof(Instruction);
    // The physical output-buffer finalization emits no vertex or primitive.
    // Clearing ATEXIT_PC prevents recursion before its terminating EXIT.
    if (!GeometryInstructionMatches(code[at - 1], 0x0000794d, 0, 0x03800000) ||
        !GeometryInstructionMatches(code[at], 0x00ff7324, 0xff, 0) ||
        !GeometryInstructionMatches(code[at + 1], 0x00007957, 0, 0) ||
        !GeometryInstructionMatches(code[at + 2], 0x0000794d, 0, 0x03800000)) return false;
    // Prove the PC pair R0:R1 and carry P0 are overwritten before any use
    // outside the setup. Keep the MOV to R3: its displacement may remain live.
    return GeometryInstructionMatches(code[5], 0x00007919, 0, 0x00001d00) &&
           GeometryInstructionMatches(code[6], 0x000a7816, 0x7772, 0xff) &&
           GeometryInstructionMatches(code[7], 0x000b7816, 0x7770, 0xff) &&
           GeometryInstructionMatches(code[8], 0x0a00720c, 0xff, 0x03f03070) &&
           GeometryInstructionMatches(code[9], 0x00007202, 0xff, 0x00000f00) &&
           GeometryInstructionMatches(code[10], 0x0a017224, 0xb, 0x078e02ff);
}
} // namespace Shader::Ampere
