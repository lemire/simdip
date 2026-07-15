#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "counters/bench.h"
#include <arpa/inet.h>
#include "avx512ip.h"
#include "vtlmks_ipv6.h"
#include "serialize_ipv4.h"
#include "serialize_ipv6.h"
double pretty_print(const std::string &name, size_t num_values,
                    counters::event_aggregate agg) {
  std::print("{:<50} : ", name);
    std::print(" {:9.3f} ns ", agg.fastest_elapsed_ns() / double(num_values));
    std::print(" {:9.2f} Mv/s ", double(num_values) * 1000 / agg.fastest_elapsed_ns());
  if (counters::has_performance_counters()) {
        std::print(" {:7.2f} GHz ", agg.cycles() / double(agg.elapsed_ns()));
        std::print(" {:7.2f} c ", agg.fastest_cycles() / double(num_values));
        std::print(" {:7.2f} i ", agg.fastest_instructions() / double(num_values));
        std::print(" {:7.2f} i/c ",
               agg.fastest_instructions() / double(agg.fastest_cycles()));
  }
  std::print("\n");
  return double(num_values) / agg.fastest_elapsed_ns();
}

std::vector<std::string> generate_random_ipv6_addresses(size_t count) {
    std::vector<std::string> strings;
    strings.reserve(count);

    // High-quality random number generator
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    char buf[INET6_ADDRSTRLEN];

    for (size_t i = 0; i < count; ++i) {
        // Generate two 64-bit parts → 128-bit IPv6 address
        uint64_t high = dist(gen);
        uint64_t low  = dist(gen);

        struct in6_addr addr{};

        // Fill the 16 bytes in network byte order (big-endian)
        for (int j = 0; j < 8; ++j) {
            addr.s6_addr[j]     = static_cast<uint8_t>(high >> (56 - j * 8));
            addr.s6_addr[j + 8] = static_cast<uint8_t>(low  >> (56 - j * 8));
        }

        // Convert binary to IPv6 string (compressed form)
        if (inet_ntop(AF_INET6, &addr, buf, sizeof(buf)) != nullptr) {
            strings.emplace_back(buf);
        }
    }

    return strings;
}

// Runs every IPv6 parser over the same set of strings and prints a row each.
void benchmark_ipv6(const std::vector<std::string>& strings) {
  size_t number_strings = strings.size();
  volatile uint64_t counter = 0;

  // We only consume a single byte of each result. Summing all 16 bytes would
  // add work unrelated to parsing and dilute the measurement; reading one byte
  // is enough to keep the parse from being optimized away.
  auto count_classic = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (inet_pton(AF_INET6, ip_str.c_str(), &addr) == 1) {
            c += addr.s6_addr[0];
        }
        // Invalid addresses are silently skipped
    }
    counter = c;
  };
  pretty_print("inet_pton", number_strings, counters::bench(count_classic));
  auto count_avx512 = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (parse_ipv6_avx512(ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
        // Invalid addresses are silently skipped
    }
    counter = c;
  };
  pretty_print("AVX-512", number_strings, counters::bench(count_avx512));
  auto count_vtlmks = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (vtlmks::parse_ipv6_avx512(ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
        // Invalid addresses are silently skipped
    }
    counter = c;
  };
  pretty_print("AVX-512 (vtlmks)", number_strings, counters::bench(count_vtlmks));
  auto count_ada = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (parse_ipv6_ada(ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
        // Invalid addresses are silently skipped
    }
    counter = c;
  };
  pretty_print("ada", number_strings, counters::bench(count_ada));
}

void collect_benchmark_results(size_t input_size, size_t number_strings) {
  benchmark_ipv6(generate_random_ipv6_addresses(number_strings));
}

// Loads IPv6 addresses, one per line, from a file (e.g. the TUM IPv6 Hitlist).
// Lines starting with '#', blank lines, and addresses longer than 45 chars
// (the SIMD fast-path bound) are skipped.
std::vector<std::string> load_ipv6_addresses(const std::string& path) {
    std::vector<std::string> strings;
    std::ifstream in(path);
    if (!in) {
        std::print("cannot open {}\n", path);
        return strings;
    }
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#' || line.size() > 45) {
            continue;
        }
        strings.push_back(line);
    }
    return strings;
}

// Cross-checks every loaded address against inet_pton (the ground truth) and
// reports, per parser, how many disagree on accept/reject or on the 16 bytes.
// Real-world data is a far stronger correctness probe than the curated tests.
void verify_dataset(const std::vector<std::string>& strings) {
    size_t pton_accept = 0;
    size_t avx_mismatch = 0, vtlmks_mismatch = 0, ada_mismatch = 0;
    for (const auto& s : strings) {
        struct in6_addr ref{}, got{}, got_vt{}, got_ada{};
        int ref_ok = inet_pton(AF_INET6, s.c_str(), &ref);
        int avx_ok = parse_ipv6_avx512(s.data(), s.size(), got.s6_addr);
        int vt_ok = vtlmks::parse_ipv6_avx512(s.data(), s.size(), got_vt.s6_addr);
        int ada_ok = parse_ipv6_ada(s.data(), s.size(), got_ada.s6_addr);
        pton_accept += (ref_ok == 1);
        bool ref_accept = (ref_ok == 1);
        if ((avx_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got.s6_addr, 16) != 0)) {
            avx_mismatch++;
        }
        if ((vt_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_vt.s6_addr, 16) != 0)) {
            vtlmks_mismatch++;
        }
        if ((ada_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_ada.s6_addr, 16) != 0)) {
            ada_mismatch++;
        }
    }
    std::print("verification: {} addresses, {} accepted by inet_pton\n",
               strings.size(), pton_accept);
    std::print("  AVX-512 mismatches: {}\n", avx_mismatch);
    std::print("  AVX-512 (vtlmks) mismatches: {}\n", vtlmks_mismatch);
    std::print("  ada mismatches: {}\n", ada_mismatch);
}

std::vector<std::string> generate_random_ipv4_addresses(size_t count) {
    std::vector<std::string> strings;
    strings.reserve(count);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

    char buf[INET_ADDRSTRLEN];

    for (size_t i = 0; i < count; ++i) {
        struct in_addr addr{};
        addr.s_addr = htonl(dist(gen));
        if (inet_ntop(AF_INET, &addr, buf, sizeof(buf)) != nullptr) {
            strings.emplace_back(buf);
        }
    }

    return strings;
}

void collect_ipv4_benchmark_results(size_t number_strings) {
  std::vector<std::string> strings = generate_random_ipv4_addresses(number_strings);
  volatile uint64_t counter = 0;

  // As with IPv6, consume only a single byte of each result.
  auto count_classic = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in_addr addr{};
        if (inet_pton(AF_INET, ip_str.c_str(), &addr) == 1) {
            c += ((const uint8_t*)&addr.s_addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("inet_pton", number_strings, counters::bench(count_classic));
  auto count_avx512 = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_avx512vl(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512", number_strings, counters::bench(count_avx512));
  auto count_ada = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_ada(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("ada", number_strings, counters::bench(count_ada));
}

// Scalar reference serializer from ada-url, included here for comparison.
// Digit pair LUT for fast decimal write: index 0..99 -> two chars.
constexpr std::array<char, 200> make_digit_pairs() noexcept {
  std::array<char, 200> t{};
  for (size_t i = 0; i < 100; ++i) {
    t[i * 2] = static_cast<char>('0' + i / 10);
    t[i * 2 + 1] = static_cast<char>('0' + i % 10);
  }
  return t;
}

constexpr auto digit_pairs = make_digit_pairs();

// Writes the decimal representation of an octet (0..255); returns the new
// write position. Reconstructed to match ada-url's digit_pairs LUT usage.
inline char* write_u8(char* point, uint8_t value) {
  if (value < 10) {
    *point = static_cast<char>('0' + value);
    return point + 1;
  } else if (value < 100) {
    std::memcpy(point, &digit_pairs[value * 2], 2);
    return point + 2;
  }
  *point = static_cast<char>('0' + value / 100);
  std::memcpy(point + 1, &digit_pairs[(value % 100) * 2], 2);
  return point + 3;
}

// out needs at least 16 bytes; NUL-terminated on return. Returns the length.
size_t ipv4(const uint64_t address, char* out) {
  char* point = out;
  point = write_u8(point, static_cast<uint8_t>(address >> 24));
  *point++ = '.';
  point = write_u8(point, static_cast<uint8_t>(address >> 16));
  *point++ = '.';
  point = write_u8(point, static_cast<uint8_t>(address >> 8));
  *point++ = '.';
  point = write_u8(point, static_cast<uint8_t>(address));
  *point = '\0';
  return static_cast<size_t>(point - out);
}

void collect_ipv4_serialize_benchmark(size_t number_values) {
  // Random 32-bit values to serialize (stored as four network-order octets).
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
  std::vector<uint32_t> values(number_values);
  for (auto& x : values) { x = htonl(dist(gen)); }
  volatile uint64_t counter = 0;

  // Consume the length and first byte so the work cannot be optimized away.
  auto count_ntop = [&values, &counter]() {
    size_t c = 0;
    char buf[INET_ADDRSTRLEN];
    for (uint32_t x : values) {
        struct in_addr addr{};
        addr.s_addr = x;
        if (inet_ntop(AF_INET, &addr, buf, sizeof(buf)) != nullptr) {
            c += std::strlen(buf) + (uint8_t)buf[0];
        }
    }
    counter = c;
  };
  pretty_print("inet_ntop", number_values, counters::bench(count_ntop));
  auto count_scalar = [&values, &counter]() {
    size_t c = 0;
    char buf[16];
    for (uint32_t x : values) {
        int n = serialize_ipv4_scalar((const uint8_t*)&x, buf);
        c += n + (uint8_t)buf[0];
    }
    counter = c;
  };
  pretty_print("scalar", number_values, counters::bench(count_scalar));
  auto count_simd = [&values, &counter]() {
    size_t c = 0;
    char buf[16];
    for (uint32_t x : values) {
        int n = serialize_ipv4_simd((const uint8_t*)&x, buf);
        c += n + (uint8_t)buf[0];
    }
    counter = c;
  };
  pretty_print("SIMD (SSE multiply-add)", number_values, counters::bench(count_simd));
  auto count_new = [&values, &counter]() {
    size_t c = 0;
    char buf[16];
    for (uint32_t x : values) {
        size_t n = new_ipv4_to_string(x, buf);
        c += n + (uint8_t)buf[0];
    }
    counter = c;
  };
  pretty_print("new_ipv4_to_string (AVX-512 compress)", number_values, counters::bench(count_new));
  auto count_ada = [&values, &counter]() {
    size_t c = 0;
    char buf[16];
    for (uint32_t x : values) {
        // ada's ipv4() takes a host-order integer (first octet in the high
        // byte); values are network order, so byte-swap to feed the same address.
        size_t n = ipv4(ntohl(x), buf);
        c += n + (uint8_t)buf[0];
    }
    counter = c;
  };
  pretty_print("ada ipv4 (LUT, char*)", number_values, counters::bench(count_ada));
#if defined(__AVX512IFMA__) && defined(__AVX512VBMI__)
  auto count_ifma = [&values, &counter]() {
    size_t c = 0;
    char buf[16];
    for (uint32_t x : values) {
        int n = serialize_ipv4_ifma((const uint8_t*)&x, buf);
        c += n + (uint8_t)buf[0];
    }
    counter = c;
  };
  pretty_print("SIMD (AVX-512 IFMA, 16-digit)", number_values, counters::bench(count_ifma));
  auto count_ifma8 = [&values, &counter]() {
    size_t c = 0;
    char buf[16];
    for (uint32_t x : values) {
        int n = serialize_ipv4_ifma8((const uint8_t*)&x, buf);
        c += n + (uint8_t)buf[0];
    }
    counter = c;
  };
  pretty_print("SIMD (AVX-512 IFMA, 2x8-digit)", number_values, counters::bench(count_ifma8));
#endif
}

void collect_ipv6_serialize_benchmark(size_t number_values) {
  // Random 16-byte addresses to serialize. A fraction are forced to have zero
  // runs / IPv4-mapped tails so the "::" and dotted-quad paths are exercised.
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  std::vector<std::array<uint8_t, 16>> values(number_values);
  for (size_t i = 0; i < number_values; ++i) {
    uint64_t hi = dist(gen), lo = dist(gen);
    for (int j = 0; j < 8; ++j) {
        values[i][j]     = uint8_t(hi >> (56 - j * 8));
        values[i][j + 8] = uint8_t(lo >> (56 - j * 8));
    }
    if (i % 4 == 0) { for (int j = 2; j < 10; ++j) values[i][j] = 0; }
    else if (i % 4 == 1) {
        for (int j = 0; j < 10; ++j) values[i][j] = 0;
        values[i][10] = 0xff; values[i][11] = 0xff;
    }
  }
  volatile uint64_t counter = 0;

  auto count_ntop = [&values, &counter]() {
    size_t c = 0;
    char buf[INET6_ADDRSTRLEN];
    for (const auto& v : values) {
        struct in6_addr addr{};
        std::memcpy(addr.s6_addr, v.data(), 16);
        if (inet_ntop(AF_INET6, &addr, buf, sizeof(buf)) != nullptr) {
            c += std::strlen(buf) + (uint8_t)buf[0];
        }
    }
    counter = c;
  };
  pretty_print("inet_ntop", number_values, counters::bench(count_ntop));
  auto count_simd = [&values, &counter]() {
    size_t c = 0;
    char buf[INET6_ADDRSTRLEN];
    for (const auto& v : values) {
        int n = serialize_ipv6(v.data(), buf);
        c += n + (uint8_t)buf[0];
    }
    counter = c;
  };
  pretty_print("serialize_ipv6", number_values, counters::bench(count_simd));
}

bool run_ipv4_serialize_tests() {
    // The SIMD serializer must match inet_ntop on every input. Octets are
    // independent, so we exhaustively vary each of the four positions across
    // 0..255 (with the others fixed) and add a large random cross-product
    // sample to exercise mixed-length compaction.
    bool all_ok = true;
    auto check = [&all_ok](uint32_t netorder) {
        char ref[INET_ADDRSTRLEN];
        struct in_addr addr{};
        addr.s_addr = netorder;
        inet_ntop(AF_INET, &addr, ref, sizeof(ref));
        char simd[16], scal[16];
        int ns = serialize_ipv4_simd((const uint8_t*)&netorder, simd);
        int nc = serialize_ipv4_scalar((const uint8_t*)&netorder, scal);
        bool bad = std::strcmp(simd, ref) != 0 || ns != (int)std::strlen(ref) ||
                   std::strcmp(scal, ref) != 0 || nc != ns;
        // new_ipv4_to_string takes the packed value directly (byte 0 = first
        // octet) and compress-fills the tail with zeros, so it is NUL-terminated.
        char nw[16];
        int nn = (int)new_ipv4_to_string(netorder, nw);
        nw[nn] = '\0';
        bad = bad || std::strcmp(nw, ref) != 0 || nn != ns;
#if defined(__AVX512IFMA__) && defined(__AVX512VBMI__)
        char ifma[16], ifma8[16];
        int ni = serialize_ipv4_ifma((const uint8_t*)&netorder, ifma);
        int ni8 = serialize_ipv4_ifma8((const uint8_t*)&netorder, ifma8);
        bad = bad || std::strcmp(ifma, ref) != 0 || ni != ns ||
              std::strcmp(ifma8, ref) != 0 || ni8 != ns;
#endif
        if (bad) {
            std::print("FAIL  ipv4 serialize: SIMD '{}' scalar '{}' new '{}' want '{}'\n",
                       simd, scal, nw, ref);
            all_ok = false;
        }
    };

    for (int pos = 0; pos < 4; pos++) {
        for (int b = 0; b < 256; b++) {
            uint8_t octets[4] = {1, 2, 3, 4};
            octets[pos] = (uint8_t)b;
            uint32_t v;
            std::memcpy(&v, octets, 4);
            check(v);
        }
    }

    std::mt19937_64 gen(12345);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (int i = 0; i < 100000; i++) { check(dist(gen)); }

    if (all_ok) {
        std::print("ok    ipv4 serialization (scalar/SSE/IFMA/new == inet_ntop)\n");
    }
    return all_ok;
}

bool run_ipv6_serialize_tests() {
    bool all_ok = true;
    // IPv6 serialization: serialize_ipv6 must match inet_ntop byte-for-byte, and
    // parsing the output back must recover the original 16 bytes (round-trip).
    auto check6 = [&all_ok](const uint8_t bytes[16]) {
        char ref[INET6_ADDRSTRLEN], got[INET6_ADDRSTRLEN];
        struct in6_addr addr{};
        std::memcpy(addr.s6_addr, bytes, 16);
        inet_ntop(AF_INET6, &addr, ref, sizeof(ref));
        int n = serialize_ipv6(bytes, got);
        struct in6_addr back{};
        bool round_trip = (parse_ipv6_avx512(got, n, back.s6_addr) == 1) &&
                          std::memcmp(back.s6_addr, bytes, 16) == 0;
        if (std::strcmp(got, ref) != 0 || n != (int)std::strlen(ref) ||
            !round_trip) {
            std::print("FAIL  ipv6 serialize: got '{}' want '{}' round_trip={}\n",
                       got, ref, round_trip);
            all_ok = false;
        }
    };

    // Curated forms (compression at every position, IPv4-mapped, boundaries).
    const char* v6cases[] = {
        "2001:db8:85a3::8a2e:370:7334", "::1", "::", "1::", "fe80::",
        "::ffff:192.168.1.1", "::192.168.0.1", "64:ff9b::192.0.2.33",
        "1:2:3:4:5:6:7:8", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        "2001:db8::", "1:0:0:2:0:0:0:3", "0:0:1:0:0:0:0:1",
    };
    for (const char* s : v6cases) {
        struct in6_addr a{};
        inet_pton(AF_INET6, s, &a);
        check6(a.s6_addr);
    }
    // Random addresses, plus zero-run and IPv4-mapped variants.
    std::mt19937_64 g6(67890);
    std::uniform_int_distribution<uint64_t> d6(0, UINT64_MAX);
    for (int i = 0; i < 100000; i++) {
        uint8_t b[16];
        uint64_t hi = d6(g6), lo = d6(g6);
        for (int j = 0; j < 8; j++) {
            b[j] = uint8_t(hi >> (56 - j * 8));
            b[j + 8] = uint8_t(lo >> (56 - j * 8));
        }
        if (i % 3 == 0) { for (int j = 2; j < 10; j++) b[j] = 0; }
        else if (i % 3 == 1) {
            for (int j = 0; j < 10; j++) b[j] = 0;
            b[10] = 0xff; b[11] = 0xff;
        }
        check6(b);
    }
    if (all_ok) {
        std::print("ok    ipv6 serialization (serialize_ipv6 == inet_ntop, round-trips)\n");
    }
    return all_ok;
}

bool run_tests() {
    // A broad set of valid IPv6 textual forms. Each must be accepted by the
    // reference (inet_pton) and yield identical 16-byte results from every
    // parser. Covers full form, leading-zero omission, "::" compression at the
    // start/middle/end and of one or many groups, embedded IPv4, mixed case,
    // and the boundary values.
    const std::vector<std::string> examples = {
        // --- full / uncompressed forms ---
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334",       // full form
        "2001:db8:85a3:0:0:8a2e:370:7334",               // leading zeroes omitted
        "1:2:3:4:5:6:7:8",                               // eight distinct groups
        "0:0:0:0:0:0:0:1",                               // ::1 fully expanded
        "fe80:0:0:0:1ff:fe23:4567:890a",                 // link-local, expanded

        // --- "::" compression in the middle ---
        "2001:db8:85a3::8a2e:370:7334",                  // compress two groups
        "2001:db8::1",                                   // compress several groups
        "2001:db8::ff00:42:8329",                        // compress in the middle
        "fe80::1ff:fe23:4567:890a",                      // link-local, compressed
        "1::8",                                          // compress six middle groups
        "1:2:3:4:5:6:7::",                               // trailing "::" (one group)

        // --- "::" at the boundaries ---
        "::1",                                           // loopback
        "::",                                            // unspecified (all zero)
        "::8",                                           // single trailing group
        "::2:3:4:5:6:7:8",                               // leading "::" (one group)
        "1::",                                           // single leading group
        "2001:db8::",                                    // trailing compression
        "fe80::",                                        // link-local prefix

        // --- mixed / upper case ---
        "2001:DB8::1",                                   // uppercase hex
        "2001:Db8:85A3::8A2E:370:7334",                  // mixed case

        // --- embedded IPv4 ---
        "::ffff:192.168.1.1",                            // IPv4-mapped
        "::ffff:0:0",                                    // IPv4-translated prefix
        "64:ff9b::192.0.2.33",                           // NAT64 (RFC 6052)
        "2001:db8::192.168.0.1",                         // embedded IPv4, compressed
        "::192.168.0.1",                                 // IPv4-compatible (deprecated)
        "0000:0000:0000:0000:0000:ffff:255.255.255.255", // longest textual form

        // --- boundary values ---
        "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",       // all ones
    };

    bool all_ok = true;
    for (const auto& s : examples) {
        struct in6_addr ref{};
        struct in6_addr got{};
        struct in6_addr got_ada{};
        struct in6_addr got_vt{};
        int ref_ok = inet_pton(AF_INET6, s.c_str(), &ref);
        int got_ok = parse_ipv6_avx512(s.data(), s.size(), got.s6_addr);
        int ada_ok = parse_ipv6_ada(s.data(), s.size(), got_ada.s6_addr);
        int vt_ok = vtlmks::parse_ipv6_avx512(s.data(), s.size(), got_vt.s6_addr);

        if (ref_ok != 1) {
            std::print("FAIL  inet_pton rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (got_ok != 1) {
            std::print("FAIL  AVX-512 rejected   {}\n", s);
            all_ok = false;
            continue;
        }
        if (ada_ok != 1) {
            std::print("FAIL  ada rejected       {}\n", s);
            all_ok = false;
            continue;
        }
        if (vt_ok != 1) {
            std::print("FAIL  vtlmks rejected    {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(ref.s6_addr, got.s6_addr, 16) != 0) {
            std::print("FAIL  AVX-512 mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(ref.s6_addr, got_ada.s6_addr, 16) != 0) {
            std::print("FAIL  ada mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(ref.s6_addr, got_vt.s6_addr, 16) != 0) {
            std::print("FAIL  vtlmks mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
        std::print("ok    {}\n", s);
    }
    return all_ok;
}

bool run_ipv4_tests() {
    // Standard dotted-decimal forms accepted by every parser, including the
    // strict SIMD fast path. inet_pton provides the reference result.
    const std::vector<std::string> standard = {
        "0.0.0.0", "192.168.0.1", "8.8.8.8", "255.255.255.255",
    };
    // Unusual-but-valid forms (octal, hex, fewer than four parts). The strict
    // SIMD path rejects these; the ada fallback must recover them. inet_aton is
    // the permissive reference (inet_pton would reject them all).
    const std::vector<std::string> unusual = {
        "192.168.1",      // three parts -> 192.168.0.1
        "1",              // one part    -> 0.0.0.1
        "16909060",       // one part    -> 1.2.3.4
        "0x7f000001",     // hex         -> 127.0.0.1
        "127.0.0.0x1",    // hex segment -> 127.0.0.1
        "0300.0250.0.1",  // octal       -> 192.168.0.1
    };

    bool all_ok = true;

    for (const auto& s : standard) {
        struct in_addr ref{};
        uint32_t got = 0, got_ada = 0;
        int ref_ok = inet_pton(AF_INET, s.c_str(), &ref);
        int got_ok = parse_ipv4_avx512vl(s.data(), s.size(), &got);
        int ada_ok = parse_ipv4_ada(s.data(), s.size(), &got_ada);
        if (ref_ok != 1 || got_ok != 1 || ada_ok != 1) {
            std::print("FAIL  ipv4 (standard) rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(&ref.s_addr, &got, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_ada, 4) != 0) {
            std::print("FAIL  ipv4 (standard) mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
        std::print("ok    {}\n", s);
    }

    for (const auto& s : unusual) {
        struct in_addr ref{};
        uint32_t got = 0, got_ada = 0;
        int ref_ok = inet_aton(s.c_str(), &ref);  // permissive reference
        int got_ok = parse_ipv4_avx512vl(s.data(), s.size(), &got);  // via fallback
        int ada_ok = parse_ipv4_ada(s.data(), s.size(), &got_ada);
        if (ref_ok != 1) {
            std::print("FAIL  inet_aton rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (got_ok != 1) {
            std::print("FAIL  AVX-512 fallback rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (ada_ok != 1) {
            std::print("FAIL  ada rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(&ref.s_addr, &got, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_ada, 4) != 0) {
            std::print("FAIL  ipv4 (unusual) mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
        std::print("ok    {} (fallback)\n", s);
    }

    return all_ok;
}

// Named groups selectable on the command line. Each maps to a correctness test
// (run under --tests) and a random-input benchmark (run under --bench).
static const char* kGroups[] = {
    "ipv6-parse", "ipv4-parse", "ipv4-serialize", "ipv6-serialize",
};

static void print_usage(const char* prog) {
    std::print("Usage: {} [options] [ipv6-address-file]\n\n", prog);
    std::print("Options:\n");
    std::print("  --only NAME[,NAME...]  Run only the named group(s); repeatable.\n");
    std::print("                         Also accepts --only=NAME,NAME.\n");
    std::print("  --no-tests             Skip correctness tests.\n");
    std::print("  --no-bench             Skip benchmarks.\n");
    std::print("  --data FILE            Real-world IPv6 address file to verify+benchmark.\n");
    std::print("  --list                 List the available group names and exit.\n");
    std::print("  -h, --help             Show this help and exit.\n\n");
    std::print("Groups:");
    for (const char* g : kGroups) { std::print(" {}", g); }
    std::print("\n");
    std::print("A bare (non-option) argument is treated as the IPv6 address file.\n");
}

// Splits "a,b,c" into the set, validating each name against kGroups.
static bool add_names(const std::string& csv, std::set<std::string>& out) {
    size_t start = 0;
    while (start <= csv.size()) {
        size_t comma = csv.find(',', start);
        std::string name = csv.substr(start, comma - start);
        if (!name.empty()) {
            bool known = false;
            for (const char* g : kGroups) { known |= (name == g); }
            if (!known) {
                std::print("unknown group '{}'\n", name);
                return false;
            }
            out.insert(name);
        }
        if (comma == std::string::npos) { break; }
        start = comma + 1;
    }
    return true;
}

int main(int argc, char **argv) {
    std::set<std::string> only;   // empty => all groups
    bool do_tests = true, do_bench = true;
    std::string datafile;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::print("{} requires an argument\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "--list") {
            for (const char* g : kGroups) { std::print("{}\n", g); }
            return 0;
        }
        else if (arg == "--no-tests") { do_tests = false; }
        else if (arg == "--no-bench") { do_bench = false; }
        else if (arg == "--data") { datafile = next("--data"); }
        else if (arg.starts_with("--data=")) { datafile = arg.substr(7); }
        else if (arg == "--only") { if (!add_names(next("--only"), only)) return 1; }
        else if (arg.starts_with("--only=")) { if (!add_names(arg.substr(7), only)) return 1; }
        else if (arg.starts_with("-")) {
            std::print("unknown option '{}'\n", arg);
            print_usage(argv[0]);
            return 1;
        }
        else { datafile = arg; }  // bare argument: IPv6 address file
    }

    auto want = [&](const char* g) { return only.empty() || only.count(g); };

    if (do_tests) {
        bool ok = true;
        if (want("ipv6-parse"))      { ok = run_tests() && ok; }
        if (want("ipv4-parse"))      { ok = run_ipv4_tests() && ok; }
        if (want("ipv4-serialize"))  { ok = run_ipv4_serialize_tests() && ok; }
        if (want("ipv6-serialize"))  { ok = run_ipv6_serialize_tests() && ok; }
        if (!ok) {
            std::print("Tests failed; aborting benchmark.\n");
            return 1;
        }
    }

    if (do_bench) {
        if (want("ipv6-parse")) {
            std::print("\nIPv6 parse (random):\n");
            collect_benchmark_results(1024, 100000);
        }
        if (want("ipv4-parse")) {
            std::print("\nIPv4 parse (random):\n");
            collect_ipv4_benchmark_results(100000);
        }
        if (want("ipv4-serialize")) {
            std::print("\nIPv4 serialize (random):\n");
            collect_ipv4_serialize_benchmark(100000);
        }
        if (want("ipv6-serialize")) {
            std::print("\nIPv6 serialize (random):\n");
            collect_ipv6_serialize_benchmark(100000);
        }
    }

    // Optional: verify+benchmark on a real-world address file (one per line),
    // such as the TUM IPv6 Hitlist:
    //   curl -O https://alcatraz.net.in.tum.de/ipv6-hitlist-service/open/responsive-addresses.txt.xz
    //   unxz responsive-addresses.txt.xz
    //   ./build/benchmark responsive-addresses.txt
    if (!datafile.empty() && want("ipv6-parse")) {
        std::vector<std::string> real = load_ipv6_addresses(datafile);
        if (real.empty()) {
            return 1;
        }
        std::print("\nIPv6 (real: {}, {} addresses):\n", datafile, real.size());
        verify_dataset(real);
        if (do_bench) { benchmark_ipv6(real); }
    }
}
