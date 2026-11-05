// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

#include <windows.h>

namespace Common {

// Scope this on the worker thread. The process default keeps both hardware
// threads of its chosen core available for the worker, including when new
// background threads are created. Existing CPU Set assignments take precedence.
class WindowsCpuSetReservation {
public:
    WindowsCpuSetReservation() = default;
    WindowsCpuSetReservation(const WindowsCpuSetReservation&) = delete;
    WindowsCpuSetReservation& operator=(const WindowsCpuSetReservation&) = delete;

    ~WindowsCpuSetReservation() {
        Reset();
    }

    bool Enable(unsigned logical_processor) {
        if (active || logical_processor >= std::numeric_limits<DWORD_PTR>::digits ||
            GetActiveProcessorGroupCount() != 1) {
            return false;
        }
        DWORD_PTR process_mask{}, system_mask{};
        GROUP_AFFINITY thread_affinity{};
        if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) ||
            !GetThreadGroupAffinity(GetCurrentThread(), &thread_affinity) ||
            thread_affinity.Group != 0 ||
            (process_mask & thread_affinity.Mask & (DWORD_PTR{1} << logical_processor)) == 0) {
            return false;
        }
        std::vector<ULONG> old_process, old_thread;
        if (!ReadSets(GetProcessDefaultCpuSets, GetCurrentProcess(), old_process) ||
            !ReadSets(GetThreadSelectedCpuSets, GetCurrentThread(), old_thread) ||
            !old_process.empty() || !old_thread.empty()) {
            return false;
        }

        ULONG bytes{};
        GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
        if (bytes == 0) {
            return false;
        }
        std::vector<unsigned char> data(bytes);
        if (!GetSystemCpuSetInformation(
                reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(data.data()), bytes, &bytes,
                GetCurrentProcess(), 0) || bytes > data.size()) {
            return false;
        }
        std::vector<SYSTEM_CPU_SET_INFORMATION> available;
        for (size_t offset = 0; offset < bytes;) {
            if (bytes - offset < 2 * sizeof(DWORD)) {
                return false;
            }
            DWORD size{}, type{};
            std::memcpy(&size, data.data() + offset, sizeof(size));
            std::memcpy(&type, data.data() + offset + sizeof(size), sizeof(type));
            if (size < 2 * sizeof(DWORD) || size > bytes - offset) {
                return false;
            }
            if (type == CpuSetInformation) {
                if (size < sizeof(SYSTEM_CPU_SET_INFORMATION)) {
                    return false;
                }
                SYSTEM_CPU_SET_INFORMATION info{};
                std::memcpy(&info, data.data() + offset, sizeof(info));
                const auto& cpu = info.CpuSet;
                if (cpu.Group == 0 && cpu.LogicalProcessorIndex < std::numeric_limits<DWORD_PTR>::digits &&
                    (process_mask & (DWORD_PTR{1} << cpu.LogicalProcessorIndex)) != 0 &&
                    (!cpu.Allocated || cpu.AllocatedToTargetProcess)) {
                    available.push_back(info);
                }
            }
            offset += size;
        }
        const auto chosen = std::find_if(available.begin(), available.end(), [&](const auto& info) {
            return info.CpuSet.LogicalProcessorIndex == logical_processor;
        });
        if (chosen == available.end()) {
            return false;
        }
        std::vector<ULONG> others;
        for (const auto& info : available) {
            if (info.CpuSet.CoreIndex != chosen->CpuSet.CoreIndex) {
                others.push_back(info.CpuSet.Id);
            }
        }
        if (others.empty()) {
            return false;
        }
        const ULONG selected = chosen->CpuSet.Id;
        if (!SetThreadSelectedCpuSets(GetCurrentThread(), &selected, 1)) {
            return false;
        }
        if (!SetProcessDefaultCpuSets(GetCurrentProcess(), others.data(),
                                     static_cast<ULONG>(others.size()))) {
            SetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0);
            return false;
        }
        process_sets = std::move(others);
        selected_set = selected;
        owner_thread = GetCurrentThreadId();
        active = true;
        if (!Matches(GetProcessDefaultCpuSets, GetCurrentProcess(), process_sets.data(),
                     process_sets.size()) ||
            !Matches(GetThreadSelectedCpuSets, GetCurrentThread(), &selected_set, 1)) {
            Reset();
            return false;
        }
        return true;
    }

    bool Reset() noexcept {
        if (!active) {
            return true;
        }
        if (GetCurrentThreadId() != owner_thread) {
            return false;
        }
        bool restored = true;
        // Respect a newer assignment made by another component while active.
        if (Matches(GetProcessDefaultCpuSets, GetCurrentProcess(), process_sets.data(),
                    process_sets.size())) {
            restored = SetProcessDefaultCpuSets(GetCurrentProcess(), nullptr, 0) != FALSE;
        }
        if (Matches(GetThreadSelectedCpuSets, GetCurrentThread(), &selected_set, 1)) {
            restored = (SetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0) != FALSE) && restored;
        }
        active = !restored;
        return restored;
    }

private:
    template <typename Query>
    static bool ReadSets(Query query, HANDLE handle, std::vector<ULONG>& result) {
        ULONG count{};
        if (!query(handle, nullptr, 0, &count) && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return false;
        }
        result.resize(count);
        if (count == 0) {
            return true;
        }
        return query(handle, result.data(), count, &count) && count == result.size();
    }

    template <typename Query>
    static bool Matches(Query query, HANDLE handle, const ULONG* expected,
                        size_t count) noexcept {
        std::array<ULONG, 64> values{};
        ULONG returned{};
        return query(handle, values.data(), static_cast<ULONG>(values.size()), &returned) &&
               returned == count && std::equal(values.begin(), values.begin() + returned, expected);
    }

    std::vector<ULONG> process_sets;
    ULONG selected_set{};
    DWORD owner_thread{};
    bool active{};
};

} // namespace Common

#endif
