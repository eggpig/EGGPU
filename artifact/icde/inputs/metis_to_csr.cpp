// Strict unweighted DIMACS10/METIS adjacency reader; no GPU or endpoint array.
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static std::vector<uint64_t> numbers(const std::string& line) {
    std::vector<uint64_t> result;
    const char *p=line.data(), *end=p+line.size();
    while (p<end) {
        while (p<end && std::isspace(static_cast<unsigned char>(*p))) ++p;
        if (p==end || *p=='%') break;
        uint64_t value=0;
        auto parsed=std::from_chars(p,end,value);
        if (parsed.ec!=std::errc{} || parsed.ptr==p ||
            (parsed.ptr<end && !std::isspace(static_cast<unsigned char>(*parsed.ptr)) && *parsed.ptr!='%'))
            throw std::runtime_error("Invalid nonnegative integer in METIS source");
        result.push_back(value); p=parsed.ptr;
    }
    return result;
}
static bool comment(const std::string& line) {
    auto p=line.find_first_not_of(" \t\r\n");
    return p!=std::string::npos && line[p]=='%';
}
template<class T> static void write(std::ofstream& out,const T* p,size_t n) {
    out.write(reinterpret_cast<const char*>(p),static_cast<std::streamsize>(n*sizeof(T)));
    if (!out) throw std::runtime_error("Output write failed");
}
int main(int argc,char**argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: metis_to_csr EXPECTED_N EXPECTED_UNDIRECTED_M OUTPUT_DIR < source.graph");
        const uint64_t expected_n=std::stoull(argv[1]),expected_m=std::stoull(argv[2]);
        if(expected_n>=uint64_t(1)<<31 || expected_m>=(uint64_t(1)<<30)) throw std::runtime_error("int32 CSR capacity exceeded");
        const uint16_t endian=1;
        if(*reinterpret_cast<const unsigned char*>(&endian)!=1) throw std::runtime_error("Little-endian build required");
        std::string line; std::vector<uint64_t> header;
        while(std::getline(std::cin,line)) { if(comment(line)) continue; header=numbers(line); if(!header.empty()) break; }
        if(header.size()<2 || header.size()>3 || (header.size()==3 && header[2]!=0)) throw std::runtime_error("Only unweighted METIS fmt=0 is supported");
        if(header[0]!=expected_n || header[1]!=expected_m) throw std::runtime_error("Published n/m mismatch");
        fs::path out=argv[3]; fs::create_directories(out);
        for(const auto* name:{"offsets.bin","indices.bin","conversion.json","offsets.bin.partial","indices.bin.partial"})
            if(fs::exists(out/name)) throw std::runtime_error("Refuse to overwrite existing conversion output");
        std::ofstream indices(out/"indices.bin.partial",std::ios::binary);
        if(!indices) throw std::runtime_error("Cannot create indices");
        std::vector<int32_t> offsets(expected_n+1); uint64_t arcs=0; size_t reordered=0;
        for(uint64_t u=0;u<expected_n;++u) {
            do { if(!std::getline(std::cin,line)) throw std::runtime_error("Truncated METIS vertex rows"); } while(comment(line));
            auto values=numbers(line);
            if(!std::is_sorted(values.begin(),values.end())) { std::sort(values.begin(),values.end()); ++reordered; }
            std::vector<int32_t> row; row.reserve(values.size());
            for(size_t k=0;k<values.size();++k) {
                auto v=values[k];
                if(v<1 || v>expected_n) throw std::runtime_error("One-based endpoint outside complete node domain");
                if(v==u+1) throw std::runtime_error("Self-loop in declared simple source");
                if(k && v==values[k-1]) throw std::runtime_error("Duplicate arc in declared simple source");
                row.push_back(static_cast<int32_t>(v-1));
            }
            arcs+=row.size();
            if(arcs>2*expected_m) throw std::runtime_error("Arc count exceeds published 2m");
            offsets[u+1]=static_cast<int32_t>(arcs); write(indices,row.data(),row.size());
        }
        while(std::getline(std::cin,line)) if(!numbers(line).empty()) throw std::runtime_error("Unexpected rows after complete node domain");
        if(std::cin.bad() || arcs!=2*expected_m) throw std::runtime_error("Source read failure or published 2m mismatch");
        indices.close(); if(!indices) throw std::runtime_error("Cannot flush indices");
        std::ofstream rows(out/"offsets.bin.partial",std::ios::binary); write(rows,offsets.data(),offsets.size()); rows.close();
        if(!rows) throw std::runtime_error("Cannot flush offsets");
        fs::rename(out/"offsets.bin.partial",out/"offsets.bin"); fs::rename(out/"indices.bin.partial",out/"indices.bin");
        std::ofstream info(out/"conversion.json");
        info << "{\"vertices\":"<<expected_n<<",\"arcs\":"<<arcs<<",\"raw_arcs\":"<<arcs
             <<",\"source_undirected_edges\":"<<expected_m<<",\"source_format\":\"DIMACS10 METIS fmt=0\","
             <<"\"normalization\":\"preserve all vertex rows; subtract one from source IDs; sort within each row; reject loops and duplicate arcs; no symmetrization\","
             <<"\"reordered_rows\":"<<reordered<<",\"symmetry_validation\":\"required by preparation before PASS\"}\n";
        info.close(); if(!info) throw std::runtime_error("Cannot flush conversion metadata");
        std::cout<<"converted "<<expected_n<<" vertices and "<<arcs<<" stored arcs\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
