#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

// Exact original d/a author text. This performs no graph algorithm or update.
extern "C" int write_updates(uint64_t n,uint64_t m,uint64_t k,const int32_t* off,
 const int32_t* ind,const double* weight,const int64_t* positions,const char* path,
 uint64_t* bytes,char* error,uint64_t error_size) noexcept {
 FILE* file=nullptr;
 try {
  if(!n||n>=INT32_MAX||m>=INT32_MAX||k>m||off[0]!=0||off[n]!=(int64_t)m)throw std::runtime_error("invalid CSR domain");
  for(uint64_t i=0;i<n;++i)if(off[i]<0||off[i]>off[i+1])throw std::runtime_error("invalid CSR offsets");
  std::vector<uint32_t> src(k);uint64_t row=0;
  for(uint64_t j=0;j<k;++j){auto p=positions[j];
   if(p<0||(uint64_t)p>=m||(j&&p<=positions[j-1])||ind[p]<0||(uint64_t)ind[p]>=n||weight[p]<1||weight[p]>16||std::floor(weight[p])!=weight[p])throw std::runtime_error("invalid frozen position/weight");
   while(row+1<n&&off[row+1]<=p)++row;src[j]=row;
  }
  if(!path){
   auto digits=[](uint32_t v){uint64_t d=1;while(v>=10){v/=10;++d;}return d;};
   uint64_t weight_digits[17]={0};
   for(uint32_t w=1;w<=16;++w)for(uint32_t v=1;v<=20;++v)weight_digits[w]+=digits(1+(w-1+v-1)%16)+digits(1+(w-1+v)%16);
   uint64_t total=0;for(uint64_t j=0;j<k;++j)total+=40*(digits(src[j])+digits(ind[positions[j]])+5)+weight_digits[(uint32_t)weight[positions[j]]];
   *bytes=total;return 0;
  }
  if(path){int fd=::open(path,O_WRONLY|O_CREAT|O_EXCL,0644);if(fd<0)throw std::runtime_error("output must be new");file=fdopen(fd,"wb");if(!file){::close(fd);throw std::runtime_error("fdopen failed");}}
  std::vector<char> buffer(8<<20);size_t used=0;uint64_t total=0;
  auto flush=[&](){if(file&&used&&fwrite(buffer.data(),1,used,file)!=used)throw std::runtime_error("write failed");used=0;};
  auto append=[&](char kind,uint32_t u,uint32_t v,uint32_t w){char line[96];char* p=line;*p++=kind;*p++=' ';
   auto number=[&](uint32_t value,char separator){auto result=std::to_chars(p,line+sizeof(line)-1,value);if(result.ec!=std::errc())throw std::runtime_error("integer format overflow");p=result.ptr;*p++=separator;};
   number(u,' ');number(v,' ');number(w,'\n');size_t length=p-line;total+=length;
   if(file){if(used+length>buffer.size())flush();memcpy(buffer.data()+used,line,length);used+=length;}
  };
  for(uint32_t version=1;version<=20;++version)for(uint32_t action=0;action<2;++action)
   for(uint64_t j=0;j<k;++j){uint64_t p=positions[j];uint32_t w=1+((uint32_t)weight[p]-1+version-(action?0:1))%16;append(action?'a':'d',src[j],ind[p],w);}
  flush();if(file){if(fflush(file)||fsync(fileno(file)))throw std::runtime_error("flush failed");int result=fclose(file);file=nullptr;if(result)throw std::runtime_error("close failed");}
  *bytes=total;return 0;
 } catch(const std::exception& e){if(file)fclose(file);if(error_size){strncpy(error,e.what(),error_size-1);error[error_size-1]=0;}return 1;}
}
