// Frozen w16-v1 payload generator. Input CSR is validated by prepare_inputs.py.
#include <openssl/sha.h>
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Mapping {
    int fd; size_t bytes; void *data;
    Mapping(const char *path, size_t size, bool output=false): bytes(size) {
        fd=open(path, output ? O_RDWR|O_CREAT|O_EXCL : O_RDONLY, 0600);
        if(fd<0) throw std::runtime_error("Cannot open payload");
        if(output && ftruncate(fd, size)) throw std::runtime_error("Cannot size output");
        struct stat s; if(fstat(fd,&s) || size_t(s.st_size)!=size) throw std::runtime_error("Payload size differs");
        data=mmap(nullptr,size,output?PROT_READ|PROT_WRITE:PROT_READ,MAP_SHARED,fd,0);
        if(data==MAP_FAILED) throw std::runtime_error("Cannot map payload");
    }
    ~Mapping(){munmap(data,bytes);close(fd);}
};

int main(int argc,char **argv) {
    try {
        if(argc!=8) throw std::runtime_error("weights OFFSETS INDICES OUTPUT N M DATASET_ID DIRECTED");
        const int64_t n=std::stoll(argv[4]),m=std::stoll(argv[5]);
        if(n<=0 || n>=INT32_MAX || m<=0 || m>=INT32_MAX) throw std::runtime_error("Requires nonempty signed-int32 CSR");
        const bool directed=std::string(argv[7])=="1";
        if(std::string(argv[7])!="0" && !directed) throw std::runtime_error("Invalid directed flag");
        Mapping om(argv[1],size_t(n+1)*4),im(argv[2],size_t(m)*4),wm(argv[3],size_t(m)*8,true);
        auto offsets=static_cast<const int32_t*>(om.data),indices=static_cast<const int32_t*>(im.data);
        auto weights=static_cast<double*>(wm.data);
        if(offsets[0] || offsets[n]!=m) throw std::runtime_error("CSR bounds differ");
        const std::string prefix="eggpu-w16-v1|20261104|"+std::string(argv[6])+"|";
        if(prefix.size()>180) throw std::runtime_error("Dataset ID too long");
        int bad=0;
        #pragma omp parallel for schedule(dynamic,256) reduction(|:bad)
        for(int64_t u=0;u<n;u++) {
            if(offsets[u]<0 || offsets[u+1]<offsets[u] || offsets[u+1]>m) {bad=1;continue;}
            char key[256]; std::memcpy(key,prefix.data(),prefix.size());
            for(int64_t j=offsets[u];j<offsets[u+1];j++) {
                int64_t v=indices[j]; if(v<0 || v>=n || v==u) {bad=1;continue;}
                auto x=directed?u:std::min(u,v),y=directed?v:std::max(u,v);
                char *end=std::to_chars(key+prefix.size(),key+256,x).ptr;*end++='|';
                end=std::to_chars(end,key+256,y).ptr;
                unsigned char digest[SHA256_DIGEST_LENGTH];
                SHA256(reinterpret_cast<const unsigned char*>(key),end-key,digest);
                // Modulo 16 of the first big-endian uint64 equals digest[7]&15.
                weights[j]=double(1+(digest[7]&15));
            }
        }
        if(bad) throw std::runtime_error("Invalid CSR row or endpoint");
        if(msync(wm.data,wm.bytes,MS_SYNC)) throw std::runtime_error("Output flush failed");
        std::cout<<"{\"status\":\"PASS_GENERATION\",\"weights\":"<<m<<"}\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
