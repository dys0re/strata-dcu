// Experimental Linux IPC adapter for a DFlash2 draft worker. No effect unless STRATA_DFLASH_FD is set.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <cerrno>
#include <cstring>
#include <memory>
#if !defined(STRATA_GPU_METAL)
#include "strata/core/dflash_native.hpp"
#endif
#if defined(__linux__)
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#endif
namespace strata::core {
class NativeHead;
class NativeEmbed;
inline bool dflash_enabled() { return std::getenv("STRATA_DFLASH_FD") != nullptr || std::getenv("STRATA_DFLASH_NATIVE") != nullptr; }
// Captures only the five completed-layer residual means expected by this checkpoint.
// Copies inside captured graphs; reducing on the host keeps this prototype backend-independent.
class DflashFeatures {
    float* data_ = nullptr;
    int cap_ = 0, n_ = 0, hc_ = 0;
public:
    const float* device_data() const { return data_; }
    int capacity() const { return cap_; }
    ~DflashFeatures() { if (data_) cudaFree(data_); }
    bool init(int cap, int n, int hc, std::string& err) {
        if (!dflash_enabled()) return true;
        if (data_) { cudaFree(data_); data_ = nullptr; }
        cap_ = cap; n_ = n; hc_ = hc;
        if (cap < 1 || n != 2560 || hc != 4 || cudaMalloc((void**) &data_, (size_t) 5 * cap * n * hc * 4) != cudaSuccess) {
            err = "DFlash2: feature allocation/geometry failed"; return false;
        }
        return true;
    }
    void capture(int completed_layer, const float* residual, int rows, int row0, cudaStream_t stream) {
        if (!data_) return;
        constexpr int taps[5] = {5,15,25,35,45};
        for (int j=0;j<5;++j) if (completed_layer==taps[j]) {
            cudaMemcpyAsync(data_ + ((size_t) j * cap_ + row0) * hc_ * n_, residual,
                            (size_t) rows * hc_ * n_ * 4, cudaMemcpyDeviceToDevice, stream);
        }
    }
    bool read(int rows, std::vector<float>& out, std::string& err) const {
        if (!data_ || rows<1 || rows>cap_) { err="DFlash2: invalid feature rows"; return false; }
        out.resize((size_t) rows*5*n_);
        std::vector<float> raw((size_t) rows*hc_*n_);
        for (int j=0;j<5;++j) {
            if (cudaMemcpy(raw.data(),data_+(size_t)j*cap_*hc_*n_,raw.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess) {
                err="DFlash2: feature download failed"; return false;
            }
            for (int t=0;t<rows;++t) for (int i=0;i<n_;++i) {
                float v=0; for(int h=0;h<hc_;++h) v+=raw[((size_t)t*hc_+h)*n_+i];
                v/=hc_;
                if (!std::isfinite(v)) { err="DFlash2: nonfinite target features";return false; }
                out[((size_t)t*5+j)*n_+i]=v;
            }
        }
        return true;
    }
};
class DflashBridge {
    int fd_=-1;
    uint64_t weights_=0;
#if !defined(STRATA_GPU_METAL)
    std::unique_ptr<DflashNative> native_;
#endif
    static constexpr int64_t magic=0x44464c32;
    bool transfer(void* p, size_t n, bool write, std::string& err) {
#if defined(__linux__)
        auto* b=(uint8_t*)p;
        while(n) {
            pollfd f{fd_,(short)(write?POLLOUT:POLLIN),0};
            int ready;do { ready=::poll(&f,1,120000); } while(ready<0&&errno==EINTR);
            if(ready<=0) { err="DFlash2: worker timeout or poll failure";return false; }
            const ssize_t k=write ? ::send(fd_,b,n,MSG_NOSIGNAL) : ::recv(fd_,b,n,0);
            if(k<0&&errno==EINTR)continue;
            if(k<=0) { err="DFlash2: worker disconnected";return false; }
            b+=k;n-=(size_t)k;
        }
        return true;
#else
        (void)p;(void)n;(void)write;err="DFlash2 IPC requires Linux";return false;
#endif
    }
    bool command(int kind,int rows,int64_t pos,int anchor,int k,const float* data,int32_t* ids,int* got,std::string& err) {
        int64_t h[6]={magic,kind,rows,pos,anchor,k};
        if(!transfer(h,sizeof h,true,err))return false;
        if(rows && !transfer((void*)data,(size_t)rows*12800*4,true,err))return false;
        int64_t r[2]={};if(!transfer(r,sizeof r,false,err))return false;
        if(r[0]!=magic||r[1]<0||r[1]>k) { err="DFlash2: worker returned invalid response";return false; }
        if(got)*got=(int)r[1];
        return r[1]==0||transfer(ids,(size_t)r[1]*4,false,err);
    }
public:
    bool init_native(const std::string& dir,const NativeHead& head,const NativeEmbed& embed,
                     int max_context,bool graph,bool adaptive,std::string& err) {
#if !defined(STRATA_GPU_METAL)
        native_.reset(new DflashNative);
        if(!native_->load(dir,head,embed,max_context,graph,adaptive,err))return false;
        weights_=native_->weight_bytes();return true;
#else
        (void)dir;(void)head;(void)embed;(void)max_context;(void)graph;(void)adaptive;
        err="Native DFlash2 currently requires CUDA/HIP";return false;
#endif
    }
    bool init(std::string& err) {
        const char* e=std::getenv("STRATA_DFLASH_FD");if(!e)return true;
        char* end=nullptr;long n=std::strtol(e,&end,10);
        if(!*e||*end||n<3||n>100000) { err="DFlash2: invalid inherited descriptor";return false; }
        fd_=(int)n;
        uint64_t hello[2]={};
        if(!transfer(hello,sizeof hello,false,err)||hello[0]!=(uint64_t)magic||!hello[1]) { err="DFlash2: invalid worker weight handshake";return false; }
        weights_=hello[1];return true;
    }
    bool native() const {
#if !defined(STRATA_GPU_METAL)
        return native_ != nullptr;
#else
        return false;
#endif
    }
    bool enabled() const { return native() || fd_>=0; }
    uint64_t weight_bytes() const { return weights_; }
    bool reset(std::string& err) {
#if !defined(STRATA_GPU_METAL)
        if(native_)return native_->reset(err);
#endif
        return !enabled()||command(0,0,0,0,0,nullptr,nullptr,nullptr,err);
    }
    bool discard_feedback(std::string& err) {
#if !defined(STRATA_GPU_METAL)
        if(native_){native_->discard_feedback();return true;}
#endif
        return !enabled()||command(3,0,0,0,0,nullptr,nullptr,nullptr,err);
    }
    bool append(const DflashFeatures& source,int rows,int64_t pos,std::string& err) {
#if !defined(STRATA_GPU_METAL)
        if(native_)return native_->append(source.device_data(),source.capacity(),rows,pos,err);
#endif
        std::vector<float> features;
        return source.read(rows,features,err)&&append(features,rows,pos,err);
    }
    bool append(const std::vector<float>& features,int rows,int64_t pos,std::string& err) {
        if(features.size()!=(size_t)rows*12800) {err="DFlash2: feature size mismatch";return false;}
        return command(1,rows,pos,0,0,features.data(),nullptr,nullptr,err);
    }
    bool draft(int64_t pos,int anchor,int k,int32_t* ids,int& got,std::string& err) {
#if !defined(STRATA_GPU_METAL)
        if(native_)return native_->draft(pos,anchor,k,ids,got,err);
#endif
        return command(2,0,pos,anchor,k,nullptr,ids,&got,err);
    }
    bool draft_seeded(int64_t pos,int anchor,int k,const int32_t* prefix,int n_prefix,double preparation_ms,
                      int32_t* ids,int& got,std::string& err) {
#if !defined(STRATA_GPU_METAL)
        if(native_)return native_->draft_seeded(pos,anchor,k,prefix,n_prefix,preparation_ms,ids,got,err);
#endif
        err="joint MTP/DFlash2 requires the native DFlash2 backend";return false;
    }
};
} // namespace strata::core
