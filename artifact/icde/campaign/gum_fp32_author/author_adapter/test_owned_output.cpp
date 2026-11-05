#include "fp32_owned_output.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <cstring>
#include <limits>
#include <stdexcept>
int main(){
 std::mt19937 rng(9183);std::size_t cases=0;
 for(std::size_t n: {0,1,2,17,4097}) for(bool identity: {false,true}) {
  std::vector<float> values(n);std::vector<int64_t> ids(n);
  for(std::size_t i=0;i<n;++i){ids[i]=i;uint32_t bits=rng()&0x7f7fffffU;std::memcpy(&values[i],&bits,4);}
  if(n>0) values[0]=-0.0f;if(n>1)values[1]=std::numeric_limits<float>::denorm_min();
  if(!identity)std::shuffle(ids.begin(),ids.end(),rng);
  std::unique_ptr<double[]> a(new double[n]),b(new double[n]);
  gum_fp32_dense_output(a.get(),values.data(),ids.data(),n,identity);
  gum_fp32_dense_output(b.get(),values.data(),ids.data(),n,identity);
  for(std::size_t i=0;i<n;++i){double expected=values[i];auto j=identity?i:ids[i];if(std::memcmp(a.get()+j,&expected,8)||std::memcmp(b.get()+j,&expected,8))throw std::runtime_error("full output mismatch");}
  if(n){b[0]=123;if(a[0]==123)throw std::runtime_error("owners aliased");}++cases;
 }return cases==10?0:1;
}
