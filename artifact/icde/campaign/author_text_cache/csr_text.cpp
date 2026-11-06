// CPU-only byte-exact serialization of the frozen author's integer edge list.
// No graph transformation, weighting, kernel, scheduling or CUDA code.
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

extern "C" int csr_text_write(uint64_t n, uint64_t m, const int32_t* off,
        const int32_t* ind, const double* weights, const char* output,
        uint64_t* written, char* error, uint64_t error_size) noexcept {
    FILE* stream=nullptr;
    try {
        if(!off || (!ind && m) || (!weights && m) || !written)
            throw std::runtime_error("missing array/output pointer");
        if(n==0 || n>=uint64_t(INT32_MAX) || m>=uint64_t(INT32_MAX))
            throw std::runtime_error("qualified int32 CSR domain exceeded");
        if(off[0]!=0 || off[n]!=int64_t(m)) throw std::runtime_error("invalid CSR bounds");
        if(output) {
            int fd=open(output,O_WRONLY|O_CREAT|O_EXCL,0644);
            if(fd<0)throw std::runtime_error("output exists or cannot create");
            stream=fdopen(fd,"wb");
            if(!stream){close(fd);throw std::runtime_error("fdopen failed");}
        }
        std::vector<char> buffer(8<<20);size_t used=0;uint64_t bytes=0;
        auto flush=[&](){
            if(stream && used && fwrite(buffer.data(),1,used,stream)!=used)
                throw std::runtime_error("write failed");
            bytes+=used;used=0;
        };
        for(uint64_t u=0;u<n;++u) {
            if(off[u]<0 || off[u+1]<off[u] || uint64_t(off[u+1])>m)
                throw std::runtime_error("invalid CSR offsets");
            for(int64_t j=off[u];j<off[u+1];++j) {
                if(ind[j]<0 || uint64_t(ind[j])>=n) throw std::runtime_error("invalid destination");
                double weight=weights[j];
                if(!std::isfinite(weight) || weight<0 || weight>UINT32_MAX || std::floor(weight)!=weight)
                    throw std::runtime_error("weights outside qualified exact uint32 domain");
                if(used+40>buffer.size())flush();
                char* at=buffer.data()+used;
                at=std::to_chars(at,buffer.data()+buffer.size(),uint32_t(u)).ptr;*at++=' ';
                at=std::to_chars(at,buffer.data()+buffer.size(),uint32_t(ind[j])).ptr;*at++=' ';
                at=std::to_chars(at,buffer.data()+buffer.size(),uint32_t(weight)).ptr;*at++='\n';
                used=size_t(at-buffer.data());
            }
        }
        flush();
        if(stream) {
            bool bad=fflush(stream)!=0;
            if(!bad)bad=fsync(fileno(stream))!=0;
            if(fclose(stream)!=0)bad=true;
            stream=nullptr;
            if(bad)throw std::runtime_error("output sync failed");
        }
        *written=bytes;
        return 0;
    } catch(const std::exception& e) {
        if(stream)fclose(stream);
        if(error && error_size) {
            std::strncpy(error,e.what(),error_size-1);error[error_size-1]='\0';
        }
        return 1;
    }
}
