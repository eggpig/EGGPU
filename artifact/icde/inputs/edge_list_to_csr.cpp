// Stream the official SNAP text or SuiteSparse Matrix Market entry from stdin.
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static uint64_t number(const char *&p,const char *end) {
    while(p<end && (*p==' ' || *p=='\t' || *p=='\r')) ++p;
    uint64_t value=0;auto r=std::from_chars(p,end,value);
    if(r.ec!=std::errc()) throw std::runtime_error("Invalid edge endpoint");
    p=r.ptr;return value;
}
int main(int argc,char **argv) {
 try {
    if(argc!=5) throw std::runtime_error("edge_list_to_csr snap|mm N EXPECTED_SOURCE_ARCS OUT_DIR");
    const bool mm=std::string(argv[1])=="mm";
    if(!mm && std::string(argv[1])!="snap") throw std::runtime_error("Unknown input format");
    const uint64_t n=std::stoull(argv[2]),expected=std::stoull(argv[3]);
    if(!n || n>=INT32_MAX || expected>=INT32_MAX) throw std::runtime_error("Signed-int32 domain exceeded");
    std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
    std::vector<uint64_t> edges;edges.reserve(expected);
    std::string line;uint64_t input_rows=0,raw_arcs=0,loops=0,declared_rows=0;
    bool dimensions=!mm,symmetric=false,header=!mm;
    while(std::getline(std::cin,line)) {
      if(line.empty())continue;
      if(mm && !header) {
        if(line.rfind("%%MatrixMarket matrix coordinate ",0)!=0)throw std::runtime_error("Invalid MatrixMarket header");
        symmetric=line.find(" symmetric")!=std::string::npos;
        if(!symmetric && line.find(" general")==std::string::npos)throw std::runtime_error("Unsupported MM symmetry");
        header=true;continue;
      }
      if(line[0]=='#' || line[0]=='%')continue;
      const char *p=line.data(),*end=p+line.size();
      auto u=number(p,end),v=number(p,end);
      if(!dimensions) {
        declared_rows=number(p,end);
        if(u!=n || v!=n) throw std::runtime_error("Matrix dimensions differ from frozen source");
        dimensions=true;continue;
      }
      if(mm) {if(!u || !v)throw std::runtime_error("MM IDs must be one based");--u;--v;}
      if(u>=n || v>=n)throw std::runtime_error("Noncanonical ID: explicit relabeling required");
      input_rows++;raw_arcs++;
      if(u==v) {loops++;continue;}
      edges.push_back((u<<32)|v);
      if(symmetric) {edges.push_back((v<<32)|u);raw_arcs++;}
    }
    if(!header || !dimensions || (mm && input_rows!=declared_rows) || raw_arcs!=expected)
      throw std::runtime_error("Official source row/arc count mismatch");
    std::sort(edges.begin(),edges.end());auto last=std::unique(edges.begin(),edges.end());
    uint64_t duplicates=edges.end()-last;edges.erase(last,edges.end());
    if(edges.size()>=INT32_MAX)throw std::runtime_error("Stored CSR exceeds signed-int32");
    std::vector<int32_t> offsets(n+1,0);
    for(auto edge:edges)offsets[(edge>>32)+1]++;
    for(size_t i=1;i<offsets.size();i++)offsets[i]+=offsets[i-1];
    std::ofstream of(std::string(argv[4])+"/offsets.bin",std::ios::binary);
    std::ofstream ind(std::string(argv[4])+"/indices.bin",std::ios::binary);
    std::ofstream stats(std::string(argv[4])+"/conversion.json");
    of.exceptions(std::ios::badbit|std::ios::failbit);ind.exceptions(std::ios::badbit|std::ios::failbit);stats.exceptions(std::ios::badbit|std::ios::failbit);
    of.write(reinterpret_cast<const char*>(offsets.data()),offsets.size()*4);
    std::vector<int32_t> block;block.reserve(1<<20);
    for(auto edge:edges) {
      block.push_back(int32_t(edge&0xffffffff));
      if(block.size()==(1<<20)) {ind.write(reinterpret_cast<const char*>(block.data()),block.size()*4);block.clear();}
    }
    if(!block.empty())ind.write(reinterpret_cast<const char*>(block.data()),block.size()*4);
    stats<<"{\"vertices\":"<<n<<",\"raw_arcs\":"<<raw_arcs<<",\"arcs\":"<<edges.size()<<",\"self_loops_removed\":"<<loops<<",\"duplicate_arcs_removed\":"<<duplicates<<",\"source_rows\":"<<input_rows<<"}\n";
    std::cout<<"{\"status\":\"PASS_TOPOLOGY_CONVERSION\",\"vertices\":"<<n<<",\"arcs\":"<<edges.size()<<"}\n";
 } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
 return 0;
}
