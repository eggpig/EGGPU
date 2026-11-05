#pragma once
#include <ctime>
#include <cstdint>
#include <unistd.h>
#include <ostream>
namespace baseline_memory {
inline std::uint64_t monotonic_ns() {
  struct timespec stamp;
  clock_gettime(CLOCK_MONOTONIC, &stamp);
  return static_cast<std::uint64_t>(stamp.tv_sec)*1000000000ULL+stamp.tv_nsec;
}
inline void interval(std::ostream& out, int rank, const char* phase, int round,
                     std::uint64_t begin, std::uint64_t end) {
  const auto pid = getpid();
  out << "{\"kind\":\"begin\",\"phase\":\"" << phase << "\",\"query_id\":" << round
      << ",\"rank\":" << rank << ",\"pid\":" << pid << ",\"id\":\"" << pid << ':' << begin
      << "\",\"t_ns\":" << begin << "}\n";
  out << "{\"kind\":\"end\",\"pid\":" << pid << ",\"id\":\"" << pid << ':' << begin
      << "\",\"t_ns\":" << end << "}\n";
}
}
