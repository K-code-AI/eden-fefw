// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>
#include "common/common_types.h"
namespace VideoCommon::ReadbackTrace {
inline bool Enabled() {
    static const bool enabled = std::getenv("EDEN_NATIVE_READBACK_TRACE") != nullptr;
    return enabled;
}
struct Writer { const char* kind = "other"; u64 hash{}, program{}; };
struct Write { u64 serial{}, address{}, size{}; u32 line{}; Writer writer{}; };
struct Reader {
    u64 gpu_base{}, source_size{}, offset{}, stride{}, count{}, shader{};
    u64 device{}, gpu_segment{}, segment_size{};
};
struct State {
    Writer writer{};
    Reader reader{};
    std::array<Write, 512> writes{};
    u64 serial{};
    u32 poll{}, remaining{}, sequence{};
    bool reading{};
    std::set<u64> dumped_shaders;
};
inline State& Get() { static thread_local State state; return state; }
struct WriterScope {
    bool active = Enabled(); Writer previous{};
    WriterScope(const char* kind, u64 hash, u64 program) {
        if (active) { previous=Get().writer; Get().writer={kind,hash,program}; }
    }
    ~WriterScope() { if (active) Get().writer=previous; }
};
inline void RecordWrite(u64 address, u64 size, u32 line) {
    if (!Enabled()) return;
    auto& s=Get();auto serial=++s.serial;
    s.writes[serial%s.writes.size()]={serial,address,size,line,s.writer};
}
struct ReaderScope {
    bool active=Enabled();Reader previous{};bool previous_reading{};
    ReaderScope(u64 base,u64 size,u64 offset,u64 stride,u64 count,u64 shader) {
        if (!active) return;
        auto& s=Get();previous=s.reader;previous_reading=s.reading;
        if (!s.remaining && (++s.poll%256)==0) {
            std::error_code ec;
            if (std::filesystem::remove("native-readback.request",ec)) s.remaining=32;
        }
        s.reader={base,size,offset,stride,count,shader};s.reading=true;
    }
    ~ReaderScope() {if(active){Get().reader=previous;Get().reading=previous_reading;}}
};
inline void Segment(u64 device,u64 gpu,u64 size) {
    if (!Enabled()) return;
    auto& r=Get().reader;r.device=device;r.gpu_segment=gpu;r.segment_size=size;
}
inline bool Capturing() {return Enabled() && Get().reading && Get().remaining;}
inline void WriteFile(const std::string& path,const void* data,size_t size) {
    std::ofstream out(path,std::ios::binary);out.write(static_cast<const char*>(data),size);
}
template<class DeviceMemory,class GpuMemory>
void ObserveCopy(DeviceMemory& memory,GpuMemory& gpu,u64 address,const u8* after,u64 size) {
    if (!Capturing()) return;
    auto& s=Get();const auto& r=s.reader;
    const auto begin=(std::max)(address,r.device);
    const auto end=(std::min)(address+size,r.device+r.segment_size);
    if (end<=begin || end-begin>1024*1024) return;
    --s.remaining;
    const std::string prefix="native-readback-"+std::to_string(s.sequence++);
    std::vector<u8> before(end-begin);memory.ReadBlockUnsafe(begin,before.data(),before.size());
    WriteFile(prefix+"-before.bin",before.data(),before.size());
    WriteFile(prefix+"-after.bin",after+(begin-address),before.size());
    std::ofstream meta(prefix+".txt");
    meta<<std::hex<<"base "<<r.gpu_base<<"\nsource_size "<<r.source_size<<"\noffset "<<r.offset<<"\nstride "<<r.stride<<"\ncount "<<r.count<<"\nshader "<<r.shader
        <<"\ncopy_address "<<address<<"\ncopy_size "<<size<<"\ncaptured_device "<<begin<<"\ncaptured_gpu "<<r.gpu_segment+begin-r.device<<"\ncaptured_size "<<end-begin<<"\n";
    for(const auto& w:s.writes) {
        if(!w.serial)continue;
        meta<<"writer "<<w.serial<<' '<<w.address<<' '<<w.size<<' '<<w.line<<' '<<w.writer.kind<<' '<<w.writer.hash<<' '<<w.writer.program<<'\n';
        if(w.writer.program && s.dumped_shaders.size()<64 && s.dumped_shaders.insert(w.writer.program).second) {
            size_t bytes{};
            while(bytes<65536 && gpu.IsFullyMappedRange(w.writer.program+bytes,4096))bytes+=4096;
            if(bytes) {
                std::vector<u8> code(bytes);gpu.ReadBlockUnsafe(w.writer.program,code.data(),bytes);
                WriteFile("native-readback-shader-"+std::to_string(w.writer.program)+".bin",code.data(),bytes);
            }
        }
    }
}
}
