// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

// Eden's Xbyak configuration (its hash maps): every translation unit must see the same one.
#include "common/x64/xbyak.h"

#include "video_core/macro_turing.h"
#include "video_core/native_draw_optimizations.h"

namespace Tegra {

// Adapted from Eglantine c0643a675b1382cfa7357626a63a623b56a72666.
// Keeps Eden FE's DMA FIFO, register validation and code-upload ordering.
class TuringMacroJit {
public:
    using Word = std::uint32_t;
    using Result = TuringMacroInterpreter::Result;

    struct State {
        std::array<Word, 24> regs;
        Word carry;
        Word method;
        Word increment;
        Word loop_count;
        Word loop_start;
        Word loop_end;
        Word next_pc;
        Word steps;
        Word start;
        Word consumed;
        Word params_size;
        Word fail_pc;
        std::uint8_t end_next;
        std::uint8_t stopped;
        std::uint8_t cancel_end;
        std::uint8_t pad;
        Word scratch_x;
        Word scratch_y;
        const Word* params;
        Word* dram;
        void* user;
        Word (*emit)(State*, Word method, Word value);
        Word (*read)(State*, Word address);
        Word initialized;
        Word error;
        Word cursor;
        Word dma_count;
        Word address_high;
        Word address_low;
        Word recompile;
        std::array<Word, 1024> dma_fifo;
        Word (*extended)(State*);
        const std::uint64_t* generation;
        std::uint64_t executing_generation;
    };

    enum Error : Word {
        None = 0,
        FifoUnderflow,
        InvalidSource,
        InvalidDestination,
        UnsupportedExtended,
        InvalidMerge,
        InvalidLoop,
        DataRamReadOutOfBounds,
        DataRamWriteOutOfBounds,
        InvalidOutput,
        UnsupportedEmit,
        OutsideCode,
        StepLimit,
        Uninitialized,
        DmaLength,
        DmaRead,
        DmaMethod,
        Recompile,
    };

    static std::string_view ErrorText(Word error) {
        constexpr std::array<std::string_view, 18> texts{
            "",
            "parameter FIFO underflow",
            "invalid source register",
            "invalid destination register",
            "unsupported extended operation",
            "invalid merge destination",
            "invalid loop",
            "data RAM read out of bounds",
            "data RAM write out of bounds",
            "invalid output selector",
            "unsupported emitted method",
            "instruction outside uploaded RAM",
            "instruction limit exceeded",
            "uninitialized register read",
            "FIFO DMA length out of bounds",
            "FIFO DMA memory read failed",
            "unsupported DMA method",
            "macro code changed",
        };
        return error < texts.size() ? texts[error] : "unknown";
    }

    /// Compiles `code` (3 words per instruction) when it differs from what was compiled.
    void Prepare(std::span<const Word> code, std::uint64_t generation) {
        const bool optimize_tail = VideoCommon::NativeDrawOptimizations::optimize_macro_tail.load(
            std::memory_order_relaxed);
        if (program && generation == compiled_generation && optimize_tail == compiled_tail_mode) {
            return;
        }
        program = std::make_unique<Program>(code, optimize_tail);
        compiled_generation = generation;
        compiled_tail_mode = optimize_tail;
    }

    template <typename Read, typename Emit>
    Result Execute(std::span<const Word> code, const std::uint64_t& generation, Word start,
                   std::span<const Word> parameters, std::span<Word> data_ram, Read read,
                   Emit emit, TuringMacroInterpreter::MemoryRead memory_read = {}) {
        State state;
        state.regs.fill(0);
        state.carry = state.method = state.increment = 0;
        state.loop_count = state.loop_start = state.loop_end = 0;
        state.consumed = state.cursor = state.initialized = state.error = 0;
        state.dma_count = state.address_high = state.address_low = state.recompile = 0;
        state.fail_pc = start;
        state.end_next = state.stopped = state.cancel_end = 0;
        state.start = start;
        state.steps = 1000000;
        state.params = parameters.data();
        state.params_size = static_cast<Word>(parameters.size());
        state.dram = data_ram.data();
        state.generation = &generation;
        Thunks<Read, Emit> thunks{read, emit, memory_read};
        state.user = &thunks;
        state.emit = &Thunks<Read, Emit>::CallEmit;
        state.read = &Thunks<Read, Emit>::CallRead;
        state.extended = &Thunks<Read, Emit>::CallExtended;
        for (;;) {
            Prepare(code, generation);
            state.executing_generation = generation;
            state.recompile = 0;
            const Word error = program->entry(&state);
            if (error == Recompile) continue;
            return {error == None, state.fail_pc, state.consumed, ErrorText(error)};
        }
    }

private:
    template <typename Read, typename Emit>
    struct Thunks {
        Read& read;
        Emit& emit;
        TuringMacroInterpreter::MemoryRead& memory_read;
        static Word CallEmit(State* state, Word method, Word value) {
            Word result = None;
            switch (method) {
            case 0x154: state->address_high = value & 0xff; break;
            case 0x155: state->address_low = value; break;
            case 0x158:
                if (value > state->dma_fifo.size()) result = DmaLength;
                else state->dma_count = value;
                break;
            case 0x156: case 0x157: case 0x159: case 0x15a:
                result = DmaMethod; break;
            default:
                if (!static_cast<Thunks*>(state->user)->emit(method, value))
                    result = UnsupportedEmit;
                break;
            }
            state->recompile = *state->generation != state->executing_generation;
            return result;
        }
        static Word CallRead(State* state, Word address) {
            const Word result = static_cast<Thunks*>(state->user)->read(address);
            state->recompile = *state->generation != state->executing_generation;
            return result;
        }
        static Word CallExtended(State* state) {
            if (state->dma_count == 0) return None;
            auto& callback = static_cast<Thunks*>(state->user)->memory_read;
            const auto address = (std::uint64_t{state->address_high} << 32) | state->address_low;
            if (!callback || !callback(address, std::span{state->dma_fifo}.first(state->dma_count)))
                return DmaRead;
            state->params = state->dma_fifo.data();
            state->params_size = state->dma_count;
            state->cursor = state->dma_count = 0;
            state->recompile = *state->generation != state->executing_generation;
            return None;
        }
    };

    struct Program : Xbyak::CodeGenerator {
        explicit Program(std::span<const Word> code, bool optimize_tail_)
            : Xbyak::CodeGenerator(1 << 20, Xbyak::AutoGrow), optimize_tail{optimize_tail_} {
            Generate(code);
            ready();
            entry = getCode<Word (*)(State*)>();
        }

        Word (*entry)(State*) = nullptr;

    private:
        using Decoded = TuringMacroInterpreter::Instruction;
#ifdef _WIN32
        const Xbyak::Reg64 arg0 = rcx, arg1 = rdx, arg2 = r8;
#else
        const Xbyak::Reg64 arg0 = rdi, arg1 = rsi, arg2 = rdx;
#endif
        // rbx: State*; r12d/r13d: the instruction's FIFO loads; r14d/r15d: its ALU results.
        const Xbyak::Reg64 st = rbx;

        static constexpr std::size_t RegOffset(Word reg) {
            return offsetof(State, regs) + reg * sizeof(Word);
        }

        void Generate(std::span<const Word> code) {
            const Word count = static_cast<Word>(code.size() / 3);
            labels.resize(count);
            push(rbx); push(rbp); push(r12); push(r13); push(r14); push(r15);
            sub(rsp, 40);
            mov(st, arg0);
            mov(eax, dword[st + offsetof(State, start)]);
            jmp(dispatch, T_NEAR);
            for (Word pc = 0; pc < count; ++pc) {
                L(labels[pc]);
                mov(dword[st + offsetof(State, fail_pc)], pc);
                cmp(dword[st + offsetof(State, steps)], 0);
                je(fail_steps, T_NEAR);
                dec(dword[st + offsetof(State, steps)]);
                Instruction(pc, count, TuringMacroInterpreter::Decode(
                    std::span<const Word, 3>{code.data() + pc * 3, 3}));
            }
            L(dispatch);
            mov(dword[st + offsetof(State, fail_pc)], eax);
            cmp(dword[st + offsetof(State, steps)], 0);
            je(fail_steps, T_NEAR);
            cmp(eax, count);
            jae(fail_outside, T_NEAR);
            lea(rcx, ptr[rip + table]);
            jmp(ptr[rcx + rax * 8]);
            L(success);
            xor_(eax, eax);
            L(leave);
            add(rsp, 40);
            pop(r15); pop(r14); pop(r13); pop(r12); pop(rbp); pop(rbx);
            ret();
            const auto fail_with = [&](Xbyak::Label& label, Word error) {
                L(label); mov(eax, error); jmp(leave, T_NEAR);
            };
            fail_with(fail_underflow, FifoUnderflow);
            fail_with(fail_outside, OutsideCode);
            fail_with(fail_steps, StepLimit);
            L(fail_current);
            mov(eax, dword[st + offsetof(State, error)]);
            jmp(leave, T_NEAR);
            align(8);
            L(table);
            for (Word pc = 0; pc < count; ++pc) putL(labels[pc]);
        }

        void SetError(Word error) {
            Xbyak::Label already_failed;
            cmp(dword[st + offsetof(State, error)], 0);
            jne(already_failed);
            mov(dword[st + offsetof(State, error)], error);
            L(already_failed);
        }

        void Get(const Xbyak::Reg32& dst, const Decoded& d, Word reg, Word lane) {
            if (reg < 24) {
                Xbyak::Label initialized;
                test(dword[st + offsetof(State, initialized)], 1U << reg);
                jnz(initialized);
                SetError(Uninitialized);
                L(initialized);
                mov(dst, dword[st + RegOffset(reg)]);
                return;
            }
            switch (reg) {
            case 24: xor_(dst, dst); return;
            case 25: mov(dst, Word(std::int32_t(std::int16_t(d.imm[lane])))); return;
            case 26: mov(dst, Word(std::int32_t(std::int16_t(d.imm[1 - lane])))); return;
            case 27: mov(dst, (d.imm[0] << 16) | d.imm[1]); return;
            case 28: mov(dst, r12d); return;
            case 29: mov(dst, r13d); return;
            default: SetError(InvalidSource); xor_(dst, dst); return;
            }
        }

        static bool IsFlow(Word op) {
            return op >= 23 && op <= 29;
        }

        void Instruction(Word pc, Word count, const Decoded& d) {
            for (Word lane = 0; lane < 2; ++lane) {
                const auto load = lane == 0 ? r12d : r13d;
                if (!(d.load_mask & (1U << lane))) { xor_(load, load); continue; }
                mov(eax, dword[st + offsetof(State, cursor)]);
                cmp(eax, dword[st + offsetof(State, params_size)]);
                jae(fail_underflow, T_NEAR);
                mov(rcx, qword[st + offsetof(State, params)]);
                mov(load, dword[rcx + rax * 4]);
                inc(dword[st + offsetof(State, cursor)]);
                inc(dword[st + offsetof(State, consumed)]);
            }
            if (!d.predicated) { Body(pc, count, d, 15); return; }
            Xbyak::Label when_false;
            Get(eax, d, d.pred_reg, 0);
            test(eax, eax);
            jz(when_false, T_NEAR);
            Body(pc, count, d, d.predicate_masks[1]);
            L(when_false);
            Body(pc, count, d, d.predicate_masks[0]);
        }

        void Body(Word pc, Word count, const Decoded& d, Word mask) {
            // Keep the prior generator for controlled live comparisons.
            if (!optimize_tail) {
                BodyBaseline(pc, count, d, mask);
                return;
            }
            bool flow = false, callback = false;
            for (Word lane = 0; lane < 2; ++lane) {
                if (mask & (1U << lane)) {
                    flow |= IsFlow(d.op[lane]);
                    callback |= d.op[lane] == 7 || d.op[lane] == 22;
                }
                callback |= (mask & (4U << lane)) && d.out_emit[lane];
            }
            xor_(r14d, r14d); xor_(r15d, r15d); xor_(ebp, ebp);
            // Only flow instructions consult or overwrite the pending next PC.
            if (flow) mov(dword[st + offsetof(State, next_pc)], pc + 1);
            if (d.end_next) mov(byte[st + offsetof(State, cancel_end)], 0);
            Xbyak::Label commit;
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (1U << lane))) continue;
                cmp(dword[st + offsetof(State, error)], 0);
                jne(commit, T_NEAR);
                or_(ebp, 1U << lane);
                Get(ecx, d, d.src_a[lane], lane);
                Get(edx, d, d.src_b[lane], lane);
                Lane(pc, d, lane);
                mov(lane == 0 ? r14d : r15d, eax);
                if (d.dst[lane] > 24) SetError(InvalidDestination);
            }
            L(commit);
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (1U << lane)) || d.dst[lane] >= 24) continue;
                Xbyak::Label not_executed;
                test(ebp, 1U << lane);
                jz(not_executed);
                mov(dword[st + RegOffset(d.dst[lane])], lane == 0 ? r14d : r15d);
                or_(dword[st + offsetof(State, initialized)], 1U << d.dst[lane]);
                L(not_executed);
            }
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (4U << lane))) continue;
                cmp(dword[st + offsetof(State, error)], 0);
                jne(fail_current, T_NEAR);
                if (d.out_method[lane]) {
                    Output(eax, d, d.out_method[lane]);
                    mov(ecx, eax); and_(ecx, 0xfff);
                    mov(dword[st + offsetof(State, method)], ecx);
                    shr(eax, 12); and_(eax, 15);
                    mov(dword[st + offsetof(State, increment)], eax);
                }
                if (d.out_emit[lane]) {
                    Output(eax, d, d.out_emit[lane]);
                    Xbyak::Label skip_emit;
                    cmp(dword[st + offsetof(State, error)], 0);
                    jne(skip_emit);
                    mov(arg2.cvt32(), eax);
                    mov(arg1.cvt32(), dword[st + offsetof(State, method)]);
                    mov(arg0, st);
                    call(ptr[st + offsetof(State, emit)]);
                    mov(dword[st + offsetof(State, error)], eax);
                    L(skip_emit);
                    mov(eax, dword[st + offsetof(State, increment)]);
                    add(dword[st + offsetof(State, method)], eax);
                }
            }
            cmp(dword[st + offsetof(State, error)], 0);
            jne(fail_current, T_NEAR);
            cmp(byte[st + offsetof(State, end_next)], 0);
            jne(success, T_NEAR);
            // A previously stopped instruction already returned through success.
            if (flow) {
                cmp(byte[st + offsetof(State, stopped)], 0);
                jne(success, T_NEAR);
            }
            if (d.end_next) {
                mov(al, byte[st + offsetof(State, cancel_end)]);
                xor_(al, 1);
                mov(byte[st + offsetof(State, end_next)], al);
            } else {
                mov(byte[st + offsetof(State, end_next)], 0);
            }
            if (flow) mov(eax, dword[st + offsetof(State, next_pc)]);
            else mov(eax, pc + 1);
            Xbyak::Label no_loop;
            cmp(dword[st + offsetof(State, loop_count)], 0);
            je(no_loop);
            cmp(dword[st + offsetof(State, loop_end)], pc);
            jne(no_loop);
            dec(dword[st + offsetof(State, loop_count)]);
            mov(eax, dword[st + offsetof(State, loop_start)]);
            inc(eax);
            L(no_loop);
            // Only callbacks can set recompile. Every preceding instruction either
            // returned for recompilation or entered this instruction with it clear.
            if (callback) {
                Xbyak::Label unchanged;
                cmp(dword[st + offsetof(State, recompile)], 0);
                je(unchanged);
                mov(dword[st + offsetof(State, start)], eax);
                mov(eax, Recompile);
                jmp(leave, T_NEAR);
                L(unchanged);
            }
            if (pc + 1 < count) {
                cmp(eax, pc + 1);
                jne(dispatch, T_NEAR);
                jmp(labels[pc + 1], T_NEAR);
            } else {
                jmp(dispatch, T_NEAR);
            }
        }

        void BodyBaseline(Word pc, Word count, const Decoded& d, Word mask) {
            xor_(r14d, r14d); xor_(r15d, r15d); xor_(ebp, ebp);
            mov(dword[st + offsetof(State, next_pc)], pc + 1);
            mov(byte[st + offsetof(State, cancel_end)], 0);
            Xbyak::Label commit;
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (1U << lane))) continue;
                cmp(dword[st + offsetof(State, error)], 0);
                jne(commit, T_NEAR);
                or_(ebp, 1U << lane);
                Get(ecx, d, d.src_a[lane], lane);
                Get(edx, d, d.src_b[lane], lane);
                Lane(pc, d, lane);
                mov(lane == 0 ? r14d : r15d, eax);
                if (d.dst[lane] > 24) SetError(InvalidDestination);
            }
            L(commit);
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (1U << lane)) || d.dst[lane] >= 24) continue;
                Xbyak::Label not_executed;
                test(ebp, 1U << lane);
                jz(not_executed);
                mov(dword[st + RegOffset(d.dst[lane])], lane == 0 ? r14d : r15d);
                or_(dword[st + offsetof(State, initialized)], 1U << d.dst[lane]);
                L(not_executed);
            }
            for (Word lane = 0; lane < 2; ++lane) {
                if (!(mask & (4U << lane))) continue;
                cmp(dword[st + offsetof(State, error)], 0);
                jne(fail_current, T_NEAR);
                if (d.out_method[lane]) {
                    Output(eax, d, d.out_method[lane]);
                    mov(ecx, eax); and_(ecx, 0xfff);
                    mov(dword[st + offsetof(State, method)], ecx);
                    shr(eax, 12); and_(eax, 15);
                    mov(dword[st + offsetof(State, increment)], eax);
                }
                if (d.out_emit[lane]) {
                    Output(eax, d, d.out_emit[lane]);
                    Xbyak::Label skip_emit;
                    cmp(dword[st + offsetof(State, error)], 0);
                    jne(skip_emit);
                    mov(arg2.cvt32(), eax);
                    mov(arg1.cvt32(), dword[st + offsetof(State, method)]);
                    mov(arg0, st);
                    call(ptr[st + offsetof(State, emit)]);
                    mov(dword[st + offsetof(State, error)], eax);
                    L(skip_emit);
                    mov(eax, dword[st + offsetof(State, increment)]);
                    add(dword[st + offsetof(State, method)], eax);
                }
            }
            cmp(dword[st + offsetof(State, error)], 0);
            jne(fail_current, T_NEAR);
            cmp(byte[st + offsetof(State, end_next)], 0);
            jne(success, T_NEAR);
            cmp(byte[st + offsetof(State, stopped)], 0);
            jne(success, T_NEAR);
            if (d.end_next) {
                mov(al, byte[st + offsetof(State, cancel_end)]);
                xor_(al, 1);
                mov(byte[st + offsetof(State, end_next)], al);
            } else {
                mov(byte[st + offsetof(State, end_next)], 0);
            }
            Xbyak::Label no_loop;
            cmp(dword[st + offsetof(State, loop_count)], 0);
            je(no_loop);
            cmp(dword[st + offsetof(State, loop_end)], pc);
            jne(no_loop);
            dec(dword[st + offsetof(State, loop_count)]);
            mov(eax, dword[st + offsetof(State, loop_start)]);
            inc(eax);
            mov(dword[st + offsetof(State, next_pc)], eax);
            L(no_loop);
            mov(eax, dword[st + offsetof(State, next_pc)]);
            Xbyak::Label unchanged;
            cmp(dword[st + offsetof(State, recompile)], 0);
            je(unchanged);
            mov(dword[st + offsetof(State, start)], eax);
            mov(eax, Recompile);
            jmp(leave, T_NEAR);
            L(unchanged);
            if (pc + 1 < count) {
                cmp(eax, pc + 1);
                jne(dispatch, T_NEAR);
                jmp(labels[pc + 1], T_NEAR);
            } else {
                jmp(dispatch, T_NEAR);
            }
        }

        void Output(const Xbyak::Reg32& dst, const Decoded& d, Word selector) {
            switch (selector) {
            case 1: mov(dst, r14d); return;
            case 2: mov(dst, r15d); return;
            case 3: mov(dst, r12d); return;
            case 4: mov(dst, r13d); return;
            case 5: mov(dst, d.imm[0]); return;
            case 6: mov(dst, d.imm[1]); return;
            case 8: mov(dst, d.imm[0] >> 12); return;
            case 9: mov(dst, d.imm[1] >> 12); return;
            case 10: mov(dst, (d.imm[0] << 16) | d.imm[1]); return;
            default: SetError(InvalidOutput); xor_(dst, dst); return;
            }
        }

        // ecx = x, edx = y; leaves the value in eax.
        void Compare(Word op) {
            cmp(ecx, edx);
            switch (op) {
            case 17: case 25: setl(al); break;
            case 18: case 26: setb(al); break;
            case 19: case 27: setle(al); break;
            case 20: case 28: setbe(al); break;
            default: sete(al); break;
            }
            movzx(eax, al);
        }

        void Lane(Word pc, const Decoded& d, Word lane) {
            const Word imm = d.imm[lane];
            const Word op = d.op[lane];
            Xbyak::Label lane_exit;
            const auto generate = [&] {
            switch (op) {
            case 0:
                mov(eax, ecx);
                add(eax, edx);
                setc(r8b);
                movzx(r8d, r8b);
                mov(dword[st + offsetof(State, carry)], r8d);
                return;
            case 1:
                lea(eax, ptr[rcx + rdx]);
                add(eax, dword[st + offsetof(State, carry)]);
                return;
            case 2:
                mov(eax, ecx);
                sub(eax, edx);
                setc(r8b);
                movzx(r8d, r8b);
                mov(dword[st + offsetof(State, carry)], r8d);
                return;
            case 3:
                mov(eax, ecx);
                sub(eax, edx);
                sub(eax, dword[st + offsetof(State, carry)]);
                return;
            case 4:
                movsxd(rax, ecx);
                movsxd(rdx, edx);
                imul(rax, rdx);
                mov(rcx, rax);
                shr(rcx, 32);
                mov(dword[st + offsetof(State, carry)], ecx);
                return;
            case 6:
                mov(eax, ecx);
                mov(edx, edx);
                imul(rax, rdx);
                mov(rcx, rax);
                shr(rcx, 32);
                mov(dword[st + offsetof(State, carry)], ecx);
                return;
            case 5:
                mov(eax, dword[st + offsetof(State, carry)]);
                return;
            case 7: {
                Xbyak::Label bad, valid;
                cmp(ecx, 0x1000); jne(bad);
                cmp(edx, 1); je(valid);
                L(bad); SetError(UnsupportedExtended); xor_(eax, eax); jmp(lane_exit, T_NEAR);
                L(valid);
                mov(arg0, st);
                call(ptr[st + offsetof(State, extended)]);
                Xbyak::Label no_error;
                cmp(dword[st + offsetof(State, error)], 0);
                jne(no_error);
                mov(dword[st + offsetof(State, error)], eax);
                L(no_error);
                xor_(eax, eax);
                return;
            }
            case 8: {
                Xbyak::Label zero, done;
                test(ecx, ecx);
                jz(zero);
                bsr(eax, ecx);
                xor_(eax, 31);
                jmp(done);
                L(zero);
                mov(eax, 32);
                L(done);
                return;
            }
            case 9:
                mov(eax, ecx);
                mov(ecx, edx);
                and_(ecx, 31);
                shl(eax, cl);
                return;
            case 10:
                mov(eax, ecx);
                mov(ecx, edx);
                and_(ecx, 31);
                shr(eax, cl);
                return;
            case 11:
                mov(eax, ecx);
                mov(ecx, edx);
                and_(ecx, 31);
                sar(eax, cl);
                return;
            case 12: mov(eax, ecx); and_(eax, edx); return;
            case 13: mov(eax, ecx); and_(eax, edx); not_(eax); return;
            case 14: mov(eax, ecx); or_(eax, edx); return;
            case 15: mov(eax, ecx); xor_(eax, edx); return;
            case 16: {
                const Word dest = (imm >> 10) & 63, bit_count = (imm >> 5) & 31;
                const Word source = imm & 31;
                if (dest >= 32) {
                    SetError(InvalidMerge); xor_(eax, eax); jmp(lane_exit, T_NEAR);
                    return;
                }
                const Word bits_mask = (1U << bit_count) - 1;
                mov(eax, ecx);
                and_(eax, ~(bits_mask << dest));
                shr(edx, source);
                and_(edx, bits_mask);
                shl(edx, dest);
                or_(eax, edx);
                return;
            }
            case 17: case 18: case 19: case 20: case 21:
                Compare(op);
                neg(eax);
                return;
            case 22:
                lea(eax, ptr[rcx + rdx]);
                and_(eax, 0xffff);
                mov(arg1.cvt32(), eax);
                mov(arg0, st);
                call(ptr[st + offsetof(State, read)]);
                return;
            case 23: {
                if (imm < 2) {
                    SetError(InvalidLoop); xor_(eax, eax); jmp(lane_exit, T_NEAR);
                    return;
                }
                Xbyak::Label skip, done;
                cmp(dword[st + offsetof(State, loop_count)], 0);
                Xbyak::Label not_nested;
                je(not_nested);
                SetError(InvalidLoop); xor_(eax, eax); jmp(lane_exit, T_NEAR);
                L(not_nested);
                test(ecx, ecx);
                jz(skip);
                dec(ecx);
                mov(dword[st + offsetof(State, loop_count)], ecx);
                mov(dword[st + offsetof(State, loop_start)], pc);
                mov(dword[st + offsetof(State, loop_end)], pc + imm - 1);
                jmp(done);
                L(skip);
                mov(dword[st + offsetof(State, next_pc)], pc + imm);
                L(done);
                xor_(eax, eax);
                return;
            }
            case 24:
                mov(dword[st + offsetof(State, next_pc)], pc + (imm & 0x7fff));
                xor_(eax, eax);
                return;
            case 25: case 26: case 27: case 28: case 29: {
                Xbyak::Label not_taken;
                Compare(op);
                cmp(eax, (imm & 0x8000) ? 1 : 0);
                jne(not_taken);
                const auto offset = std::int32_t(imm << 19) >> 19;
                if (offset == -4096) {
                    mov(byte[st + offsetof(State, stopped)], 1);
                } else {
                    mov(dword[st + offsetof(State, next_pc)], Word(std::int32_t(pc) + offset));
                    mov(byte[st + offsetof(State, cancel_end)], (imm & 0x4000) ? 1 : 0);
                }
                L(not_taken);
                xor_(eax, eax);
                return;
            }
            case 30:
                cmp(ecx, 0xc00);
                { Xbyak::Label valid; jb(valid);
                  SetError(DataRamReadOutOfBounds); xor_(eax, eax); jmp(lane_exit, T_NEAR); L(valid); }
                mov(rax, qword[st + offsetof(State, dram)]);
                mov(eax, dword[rax + rcx * 4]);
                return;
            case 31:
                cmp(ecx, 0xc00);
                { Xbyak::Label valid; jb(valid);
                  SetError(DataRamWriteOutOfBounds); xor_(eax, eax); jmp(lane_exit, T_NEAR); L(valid); }
                mov(rax, qword[st + offsetof(State, dram)]);
                mov(dword[rax + rcx * 4], edx);
                xor_(eax, eax);
                return;
            default:
                xor_(eax, eax);
                return;
            }
            };
            generate();
            L(lane_exit);
        }

        const bool optimize_tail;
        std::vector<Xbyak::Label> labels;
        Xbyak::Label table, dispatch, success, leave;
        Xbyak::Label fail_underflow, fail_outside, fail_steps, fail_current;
    };

    std::unique_ptr<Program> program;
    std::uint64_t compiled_generation{~std::uint64_t{0}};
    bool compiled_tail_mode{};
};

} // namespace Tegra

