#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <print>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "counters/bench.h"
#include <arpa/inet.h>
#include <netdb.h>       // getaddrinfo, AI_NUMERICHOST
#include "avx512ip.h"
#include "ada_ip.h"
#include "vtlmks_ipv6.h"
#include "vtlmks_ipv6_opt.h"
#include "mula_sse_ipv4.h"
#include "simdzone_ipv4.h"

// Competitor libraries fetched via CPM (see CMakeLists.txt).
#include "ipv6.h"                       // jrepp/ipv6-parse (pure C)
#include <ipaddress/ipaddress.hpp>      // VladimirShaleev/ipaddress (header-only)
#include <boost/asio/ip/address.hpp>    // Boost.Asio ip::make_address_v4/v6

// --- Small adapters giving each competitor the project's
// (const char*, size_t, out) -> int (1 success / 0 fail) signature. ---

// getaddrinfo with AI_NUMERICHOST: the "what applications actually call" path,
// overhead included. Works for both families via the af argument.
static int parse_getaddrinfo(int af, const char *input, size_t len, void *out) {
    std::string s(input, len);  // getaddrinfo needs a NUL-terminated string
    struct addrinfo hints{};
    hints.ai_family = af;
    hints.ai_flags = AI_NUMERICHOST;
    struct addrinfo *res = nullptr;
    if (getaddrinfo(s.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) {
        return 0;
    }
    if (af == AF_INET) {
        std::memcpy(out, &((struct sockaddr_in *)res->ai_addr)->sin_addr, 4);
    } else {
        std::memcpy(out, &((struct sockaddr_in6 *)res->ai_addr)->sin6_addr, 16);
    }
    freeaddrinfo(res);
    return 1;
}

// jrepp/ipv6-parse: components[] are hextets in machine (left-to-right) order;
// repack them into 16 network-order bytes to match inet_pton.
static int parse_ipv6_jrepp(const char *input, size_t len, uint8_t *out) {
    ipv6_address_full_t addr{};
    if (!ipv6_from_str(input, len, &addr)) { return 0; }
    for (int i = 0; i < 8; i++) {
        out[2 * i] = uint8_t(addr.address.components[i] >> 8);
        out[2 * i + 1] = uint8_t(addr.address.components[i] & 0xff);
    }
    return 1;
}

// VladimirShaleev/ipaddress: non-throwing parse via the error_code overload.
// bytes() is already network order (first octet/hextet first).
static int parse_ipv4_ipaddress(const char *input, size_t len, uint32_t *out) {
    ipaddress::error_code ec = ipaddress::error_code::no_error;
    auto a = ipaddress::ipv4_address::parse(std::string_view(input, len), ec);
    if (ec != ipaddress::error_code::no_error) { return 0; }
    std::memcpy(out, a.bytes().data(), 4);
    return 1;
}
static int parse_ipv6_ipaddress(const char *input, size_t len, uint8_t *out) {
    ipaddress::error_code ec = ipaddress::error_code::no_error;
    auto a = ipaddress::ipv6_address::parse(std::string_view(input, len), ec);
    if (ec != ipaddress::error_code::no_error) { return 0; }
    std::memcpy(out, a.bytes().data(), 16);
    return 1;
}

// Boost.Asio: make_address_v4/v6 with an error_code (no exceptions on failure).
static int parse_ipv4_boost(const char *input, size_t len, uint32_t *out) {
    boost::system::error_code ec;
    auto a = boost::asio::ip::make_address_v4(std::string_view(input, len), ec);
    if (ec) { return 0; }
    auto b = a.to_bytes();  // network order
    std::memcpy(out, b.data(), 4);
    return 1;
}
static int parse_ipv6_boost(const char *input, size_t len, uint8_t *out) {
    boost::system::error_code ec;
    auto a = boost::asio::ip::make_address_v6(std::string_view(input, len), ec);
    if (ec) { return 0; }
    auto b = a.to_bytes();  // network order
    std::memcpy(out, b.data(), 16);
    return 1;
}
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
  auto count_vtlmks_opt = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (vtlmks_opt::parse_ipv6_avx512_opt(ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512 (vtlmks, optimized)", number_strings, counters::bench(count_vtlmks_opt));
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
  auto count_getaddrinfo = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (parse_getaddrinfo(AF_INET6, ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
    }
    counter = c;
  };
  pretty_print("getaddrinfo", number_strings, counters::bench(count_getaddrinfo));
  auto count_jrepp = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (parse_ipv6_jrepp(ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
    }
    counter = c;
  };
  pretty_print("jrepp/ipv6-parse", number_strings, counters::bench(count_jrepp));
  auto count_ipaddress = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (parse_ipv6_ipaddress(ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
    }
    counter = c;
  };
  pretty_print("ipaddress", number_strings, counters::bench(count_ipaddress));
  auto count_boost = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in6_addr addr{};
        if (parse_ipv6_boost(ip_str.data(), ip_str.size(), addr.s6_addr)) {
            c += addr.s6_addr[0];
        }
    }
    counter = c;
  };
  pretty_print("Boost.Asio", number_strings, counters::bench(count_boost));
}

void collect_benchmark_results(size_t number_strings) {
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
    size_t vtlmks_mismatch = 0, vtlmks_opt_mismatch = 0, ada_mismatch = 0;
    size_t jrepp_mismatch = 0, ipaddress_mismatch = 0, boost_mismatch = 0;
    for (const auto& s : strings) {
        struct in6_addr ref{}, got_vt{}, got_ada{};
        struct in6_addr got_jr{}, got_ia{}, got_bo{};
        int ref_ok = inet_pton(AF_INET6, s.c_str(), &ref);
        int vt_ok = vtlmks::parse_ipv6_avx512(s.data(), s.size(), got_vt.s6_addr);
        struct in6_addr got_vo{};
        int vo_ok = vtlmks_opt::parse_ipv6_avx512_opt(s.data(), s.size(), got_vo.s6_addr);
        int ada_ok = parse_ipv6_ada(s.data(), s.size(), got_ada.s6_addr);
        int jr_ok = parse_ipv6_jrepp(s.data(), s.size(), got_jr.s6_addr);
        int ia_ok = parse_ipv6_ipaddress(s.data(), s.size(), got_ia.s6_addr);
        int bo_ok = parse_ipv6_boost(s.data(), s.size(), got_bo.s6_addr);
        pton_accept += (ref_ok == 1);
        bool ref_accept = (ref_ok == 1);
        if ((vt_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_vt.s6_addr, 16) != 0)) {
            vtlmks_mismatch++;
        }
        if ((vo_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_vo.s6_addr, 16) != 0)) {
            vtlmks_opt_mismatch++;
        }
        if ((ada_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_ada.s6_addr, 16) != 0)) {
            ada_mismatch++;
        }
        if ((jr_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_jr.s6_addr, 16) != 0)) {
            jrepp_mismatch++;
        }
        if ((ia_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_ia.s6_addr, 16) != 0)) {
            ipaddress_mismatch++;
        }
        if ((bo_ok == 1) != ref_accept ||
            (ref_accept && std::memcmp(ref.s6_addr, got_bo.s6_addr, 16) != 0)) {
            boost_mismatch++;
        }
    }
    std::print("verification: {} addresses, {} accepted by inet_pton\n",
               strings.size(), pton_accept);
    std::print("  AVX-512 (vtlmks) mismatches: {}\n", vtlmks_mismatch);
    std::print("  AVX-512 (vtlmks, optimized) mismatches: {}\n", vtlmks_opt_mismatch);
    std::print("  ada mismatches: {}\n", ada_mismatch);
    std::print("  jrepp/ipv6-parse mismatches: {}\n", jrepp_mismatch);
    std::print("  ipaddress mismatches: {}\n", ipaddress_mismatch);
    std::print("  Boost.Asio mismatches: {}\n", boost_mismatch);
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
  auto count_avx512_notab = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_avx512vl_notab(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512 (no table)", number_strings, counters::bench(count_avx512_notab));
  auto count_avx512_notab2 = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_avx512vl_notab2(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512 (no table 2)", number_strings, counters::bench(count_avx512_notab2));
  auto count_avx512_notab3 = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_avx512vl_notab3(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512 (no table 3)", number_strings, counters::bench(count_avx512_notab3));
  auto count_avx512_notab4 = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_avx512vl_notab4(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512 (no table 4)", number_strings, counters::bench(count_avx512_notab4));
  auto count_avx512_notab5 = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_avx512vl_notab5(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512 (no table 5)", number_strings, counters::bench(count_avx512_notab5));
  auto count_avx512_tiny = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_avx512vl_tiny(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("AVX-512 (512-byte table)", number_strings, counters::bench(count_avx512_tiny));
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
  auto count_mula = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_mula_sse(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("Mula (SSE)", number_strings, counters::bench(count_mula));
  auto count_simdzone = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_simdzone(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("simdzone (SSE)", number_strings, counters::bench(count_simdzone));
  auto count_aton = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        struct in_addr addr{};
        if (inet_aton(ip_str.c_str(), &addr) != 0) {
            c += ((const uint8_t*)&addr.s_addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("inet_aton", number_strings, counters::bench(count_aton));
  auto count_getaddrinfo = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_getaddrinfo(AF_INET, ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("getaddrinfo", number_strings, counters::bench(count_getaddrinfo));
  auto count_ipaddress = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_ipaddress(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("ipaddress", number_strings, counters::bench(count_ipaddress));
  auto count_boost = [&strings, &counter]() {
    size_t c = 0;
    for (const auto& ip_str : strings) {
        uint32_t addr = 0;
        if (parse_ipv4_boost(ip_str.data(), ip_str.size(), &addr)) {
            c += ((const uint8_t*)&addr)[0];
        }
    }
    counter = c;
  };
  pretty_print("Boost.Asio", number_strings, counters::bench(count_boost));
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
        struct in6_addr got_ada{};
        struct in6_addr got_vt{};
        int ref_ok = inet_pton(AF_INET6, s.c_str(), &ref);
        int ada_ok = parse_ipv6_ada(s.data(), s.size(), got_ada.s6_addr);
        int vt_ok = vtlmks::parse_ipv6_avx512(s.data(), s.size(), got_vt.s6_addr);
        struct in6_addr got_vo{};
        int vo_ok = vtlmks_opt::parse_ipv6_avx512_opt(s.data(), s.size(), got_vo.s6_addr);

        if (ref_ok != 1) {
            std::print("FAIL  inet_pton rejected {}\n", s);
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
        if (vo_ok != 1) {
            std::print("FAIL  vtlmks-opt rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(ref.s6_addr, got_ada.s6_addr, 16) != 0) {
            std::print("FAIL  ada mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(ref.s6_addr, got_vo.s6_addr, 16) != 0) {
            std::print("FAIL  vtlmks-opt mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(ref.s6_addr, got_vt.s6_addr, 16) != 0) {
            std::print("FAIL  vtlmks mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
    }
    if (all_ok) { std::print("ok    ipv6 parse ({} forms)\n", examples.size()); }
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
        uint32_t got = 0, got_nt = 0, got_nt2 = 0, got_nt3 = 0, got_nt4 = 0, got_nt5 = 0, got_tiny = 0, got_ada = 0, got_mula = 0, got_sz = 0;
        int ref_ok = inet_pton(AF_INET, s.c_str(), &ref);
        int got_ok = parse_ipv4_avx512vl(s.data(), s.size(), &got);  // masked load: safe
        int nt_ok = parse_ipv4_avx512vl_notab(s.data(), s.size(), &got_nt);
        int nt2_ok = parse_ipv4_avx512vl_notab2(s.data(), s.size(), &got_nt2);
        int nt3_ok = parse_ipv4_avx512vl_notab3(s.data(), s.size(), &got_nt3);
        int nt4_ok = parse_ipv4_avx512vl_notab4(s.data(), s.size(), &got_nt4);
        int nt5_ok = parse_ipv4_avx512vl_notab5(s.data(), s.size(), &got_nt5);
        int tiny_ok = parse_ipv4_avx512vl_tiny(s.data(), s.size(), &got_tiny);
        int ada_ok = parse_ipv4_ada(s.data(), s.size(), &got_ada);
        // Mula and simdzone read 16 bytes; give them a padded buffer so the
        // over-read stays in-bounds during the test.
        char buf[16] = {0};
        std::memcpy(buf, s.data(), s.size());
        int mula_ok = parse_ipv4_mula_sse(buf, s.size(), &got_mula);
        int sz_ok = parse_ipv4_simdzone(buf, s.size(), &got_sz);
        if (ref_ok != 1 || got_ok != 1 || nt_ok != 1 || nt2_ok != 1 || nt3_ok != 1 || nt4_ok != 1 || nt5_ok != 1 || tiny_ok != 1 || ada_ok != 1 || mula_ok != 1 || sz_ok != 1) {
            std::print("FAIL  ipv4 (standard) rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(&ref.s_addr, &got, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt2, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt3, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt4, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt5, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_tiny, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_ada, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_mula, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_sz, 4) != 0) {
            std::print("FAIL  ipv4 (standard) mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
    }

    for (const auto& s : unusual) {
        struct in_addr ref{};
        uint32_t got = 0, got_nt = 0, got_nt2 = 0, got_nt3 = 0, got_nt4 = 0, got_nt5 = 0, got_tiny = 0, got_ada = 0;
        int ref_ok = inet_aton(s.c_str(), &ref);  // permissive reference
        int got_ok = parse_ipv4_avx512vl(s.data(), s.size(), &got);  // via fallback
        int nt_ok = parse_ipv4_avx512vl_notab(s.data(), s.size(), &got_nt);
        int nt2_ok = parse_ipv4_avx512vl_notab2(s.data(), s.size(), &got_nt2);
        int nt3_ok = parse_ipv4_avx512vl_notab3(s.data(), s.size(), &got_nt3);
        int nt4_ok = parse_ipv4_avx512vl_notab4(s.data(), s.size(), &got_nt4);
        int nt5_ok = parse_ipv4_avx512vl_notab5(s.data(), s.size(), &got_nt5);
        int tiny_ok = parse_ipv4_avx512vl_tiny(s.data(), s.size(), &got_tiny);
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
        if (nt_ok != 1) {
            std::print("FAIL  AVX-512 (no table) fallback rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (nt2_ok != 1 || nt3_ok != 1 || nt4_ok != 1 || nt5_ok != 1 || tiny_ok != 1) {
            std::print("FAIL  AVX-512 (no table 2) fallback rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (ada_ok != 1) {
            std::print("FAIL  ada rejected {}\n", s);
            all_ok = false;
            continue;
        }
        if (std::memcmp(&ref.s_addr, &got, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt2, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt3, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt4, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt5, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_tiny, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_ada, 4) != 0) {
            std::print("FAIL  ipv4 (unusual) mismatch on {}\n", s);
            all_ok = false;
            continue;
        }
    }

    // Malformed pure-decimal hosts that still look like "digits and three dots"
    // (the shape a naive AVX-512 pre-check might accept) but violate group
    // structure: each group must be 1-3 digits and non-empty. From
    // https://github.com/ada-url/ada/pull/1212 — must be rejected on every path.
    const std::vector<std::string> malformed_groups = {
        "1234.5.6.7",   // group longer than three digits
        "1.2345.6.7",   // group longer than three digits
        "1..2.34",      // empty group / consecutive dots
        "12.3..4",      // empty group / consecutive dots
    };
    for (const auto& s : malformed_groups) {
        uint32_t got = 0, got_nt = 0, got_nt2 = 0, got_nt3 = 0, got_nt4 = 0, got_nt5 = 0, got_tiny = 0, got_ada = 0, got_mula = 0, got_sz = 0;
        int got_ok = parse_ipv4_avx512vl(s.data(), s.size(), &got);
        int nt_ok = parse_ipv4_avx512vl_notab(s.data(), s.size(), &got_nt);
        int nt2_ok = parse_ipv4_avx512vl_notab2(s.data(), s.size(), &got_nt2);
        int nt3_ok = parse_ipv4_avx512vl_notab3(s.data(), s.size(), &got_nt3);
        int nt4_ok = parse_ipv4_avx512vl_notab4(s.data(), s.size(), &got_nt4);
        int nt5_ok = parse_ipv4_avx512vl_notab5(s.data(), s.size(), &got_nt5);
        int tiny_ok = parse_ipv4_avx512vl_tiny(s.data(), s.size(), &got_tiny);
        int ada_ok = parse_ipv4_ada(s.data(), s.size(), &got_ada);
        char buf[16] = {0};
        std::memcpy(buf, s.data(), s.size());
        int mula_ok = parse_ipv4_mula_sse(buf, s.size(), &got_mula);
        int sz_ok = parse_ipv4_simdzone(buf, s.size(), &got_sz);
        if (got_ok != 0) {
            std::print("FAIL  AVX-512 accepted malformed {}\n", s);
            all_ok = false;
        }
        if (nt_ok != 0) {
            std::print("FAIL  AVX-512 (no table) accepted malformed {}\n", s);
            all_ok = false;
        }
        if (nt2_ok != 0 || nt3_ok != 0 || nt4_ok != 0 || nt5_ok != 0 || tiny_ok != 0) {
            std::print("FAIL  AVX-512 (no table 2) accepted malformed {}\n", s);
            all_ok = false;
        }
        if (ada_ok != 0) {
            std::print("FAIL  ada accepted malformed {}\n", s);
            all_ok = false;
        }
        if (mula_ok != 0) {
            std::print("FAIL  mula accepted malformed {}\n", s);
            all_ok = false;
        }
        if (sz_ok != 0) {
            std::print("FAIL  simdzone accepted malformed {}\n", s);
            all_ok = false;
        }
    }

    // Every 1..3-digit layout of a strict dotted-quad (81 combinations).
    // Completeness check for the table shuffle and both table-free
    // permutes. Values 1 / 12 / 123, no leading zeros.
    {
        const int kVal[4] = {0, 1, 12, 123};
        for (int l0 = 1; l0 <= 3; l0++)
        for (int l1 = 1; l1 <= 3; l1++)
        for (int l2 = 1; l2 <= 3; l2++)
        for (int l3 = 1; l3 <= 3; l3++) {
            char s[16];
            char *p = s;
            auto app = [&](int l) {
                int v = kVal[l];
                if (v >= 100) { *p++ = char('0' + v / 100); }
                if (v >= 10)  { *p++ = char('0' + (v / 10) % 10); }
                *p++ = char('0' + v % 10);
            };
            app(l0); *p++ = '.'; app(l1); *p++ = '.'; app(l2); *p++ = '.'; app(l3);
            const size_t slen = size_t(p - s);
            uint8_t want[4] = {uint8_t(kVal[l0]), uint8_t(kVal[l1]),
                               uint8_t(kVal[l2]), uint8_t(kVal[l3])};
            struct in_addr ref{};
            uint32_t got = 0, got_nt = 0, got_nt2 = 0, got_nt3 = 0, got_nt4 = 0, got_nt5 = 0, got_tiny = 0, got_ada = 0, got_mula = 0, got_sz = 0;
            int ref_ok = inet_pton(AF_INET, std::string(s, slen).c_str(), &ref);
            int got_ok = parse_ipv4_avx512vl(s, slen, &got);
            int nt_ok = parse_ipv4_avx512vl_notab(s, slen, &got_nt);
            int nt2_ok = parse_ipv4_avx512vl_notab2(s, slen, &got_nt2);
            int nt3_ok = parse_ipv4_avx512vl_notab3(s, slen, &got_nt3);
            int nt4_ok = parse_ipv4_avx512vl_notab4(s, slen, &got_nt4);
            int nt5_ok = parse_ipv4_avx512vl_notab5(s, slen, &got_nt5);
            int tiny_ok = parse_ipv4_avx512vl_tiny(s, slen, &got_tiny);
            int ada_ok = parse_ipv4_ada(s, slen, &got_ada);
            char buf[16] = {0};
            std::memcpy(buf, s, slen);
            int mula_ok = parse_ipv4_mula_sse(buf, slen, &got_mula);
            int sz_ok = parse_ipv4_simdzone(buf, slen, &got_sz);
            if (ref_ok != 1 || got_ok != 1 || nt_ok != 1 || nt2_ok != 1 || nt3_ok != 1 || nt4_ok != 1 || nt5_ok != 1 || tiny_ok != 1 || ada_ok != 1 || mula_ok != 1 || sz_ok != 1 ||
                std::memcmp(&ref.s_addr, want, 4) != 0 ||
                std::memcmp(&got, want, 4) != 0 ||
                std::memcmp(&got_nt, want, 4) != 0 ||
                std::memcmp(&got_nt2, want, 4) != 0 ||
                std::memcmp(&got_nt3, want, 4) != 0 ||
                std::memcmp(&got_nt4, want, 4) != 0 ||
                std::memcmp(&got_nt5, want, 4) != 0 ||
                std::memcmp(&got_tiny, want, 4) != 0 ||
                std::memcmp(&got_ada, want, 4) != 0 ||
                std::memcmp(&got_mula, want, 4) != 0 ||
                std::memcmp(&got_sz, want, 4) != 0) {
                std::print("FAIL  ipv4 layout {} ({}-{}-{}-{})\n",
                           std::string_view(s, slen), l0, l1, l2, l3);
                all_ok = false;
            }
        }
    }

    // Overflow: four well-spaced digit groups, but an octet > 255.
    for (const char *s : {"256.1.1.1", "1.256.1.1", "1.1.256.1", "1.1.1.256",
                          "999.0.0.1", "1.2.3.999"}) {
        uint32_t got = 0, got_nt = 0, got_nt2 = 0, got_nt3 = 0, got_nt4 = 0, got_nt5 = 0, got_tiny = 0, got_ada = 0;
        if (parse_ipv4_avx512vl(s, std::strlen(s), &got) != 0) {
            std::print("FAIL  AVX-512 accepted overflow {}\n", s);
            all_ok = false;
        }
        if (parse_ipv4_avx512vl_notab(s, std::strlen(s), &got_nt) != 0) {
            std::print("FAIL  AVX-512 (no table) accepted overflow {}\n", s);
            all_ok = false;
        }
        if (parse_ipv4_avx512vl_notab2(s, std::strlen(s), &got_nt2) != 0) {
            std::print("FAIL  AVX-512 (no table 2) accepted overflow {}\n", s);
            all_ok = false;
        }
        if (parse_ipv4_avx512vl_notab3(s, std::strlen(s), &got_nt3) != 0) {
            std::print("FAIL  AVX-512 (no table 3) accepted overflow {}\n", s);
            all_ok = false;
        }
        if (parse_ipv4_avx512vl_notab5(s, std::strlen(s), &got_nt5) != 0) {
            std::print("FAIL  AVX-512 (no table 5) accepted overflow {}\n", s);
            all_ok = false;
        }
        if (parse_ipv4_avx512vl_tiny(s, std::strlen(s), &got_tiny) != 0) {
            std::print("FAIL  AVX-512 (512-byte table) accepted overflow {}\n", s);
            all_ok = false;
        }
        if (parse_ipv4_ada(s, std::strlen(s), &got_ada) != 0) {
            std::print("FAIL  ada accepted overflow {}\n", s);
            all_ok = false;
        }
    }

    // Same length as "1234.5.6.7" / "1.2345.6.7", but well-formed — must parse.
    {
        const std::string ok_host = "12.34.56.78";
        struct in_addr ref{};
        uint32_t got = 0, got_nt = 0, got_nt2 = 0, got_nt3 = 0, got_nt4 = 0, got_nt5 = 0, got_tiny = 0, got_ada = 0, got_mula = 0, got_sz = 0;
        int ref_ok = inet_pton(AF_INET, ok_host.c_str(), &ref);
        int got_ok = parse_ipv4_avx512vl(ok_host.data(), ok_host.size(), &got);
        int nt_ok = parse_ipv4_avx512vl_notab(ok_host.data(), ok_host.size(), &got_nt);
        int nt2_ok = parse_ipv4_avx512vl_notab2(ok_host.data(), ok_host.size(), &got_nt2);
        int nt3_ok = parse_ipv4_avx512vl_notab3(ok_host.data(), ok_host.size(), &got_nt3);
        int nt4_ok = parse_ipv4_avx512vl_notab4(ok_host.data(), ok_host.size(), &got_nt4);
        int nt5_ok = parse_ipv4_avx512vl_notab5(ok_host.data(), ok_host.size(), &got_nt5);
        int tiny_ok = parse_ipv4_avx512vl_tiny(ok_host.data(), ok_host.size(), &got_tiny);
        int ada_ok = parse_ipv4_ada(ok_host.data(), ok_host.size(), &got_ada);
        char buf[16] = {0};
        std::memcpy(buf, ok_host.data(), ok_host.size());
        int mula_ok = parse_ipv4_mula_sse(buf, ok_host.size(), &got_mula);
        int sz_ok = parse_ipv4_simdzone(buf, ok_host.size(), &got_sz);
        if (ref_ok != 1 || got_ok != 1 || nt_ok != 1 || nt2_ok != 1 || nt3_ok != 1 || nt4_ok != 1 || nt5_ok != 1 || tiny_ok != 1 || ada_ok != 1 || mula_ok != 1 || sz_ok != 1 ||
            std::memcmp(&ref.s_addr, &got, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt2, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt3, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt4, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_nt5, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_tiny, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_ada, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_mula, 4) != 0 ||
            std::memcmp(&ref.s_addr, &got_sz, 4) != 0) {
            std::print("FAIL  ipv4 well-formed same-length host {}\n", ok_host);
            all_ok = false;
        }
    }

    if (all_ok) {
        std::print("ok    ipv4 parse ({} standard, {} unusual/fallback, "
                   "{} malformed-reject, 81 layouts, 6 overflow-reject, "
                   "1 same-length ok)\n",
                   standard.size(), unusual.size(), malformed_groups.size());
    }
    return all_ok;
}

// Named groups selectable on the command line. Each maps to a correctness test
// (run under --tests) and a random-input benchmark (run under --bench).
static const char* kGroups[] = {
    "ipv6-parse", "ipv4-parse",
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
        if (!ok) {
            std::print("Tests failed; aborting benchmark.\n");
            return 1;
        }
    }

    if (do_bench) {
        if (want("ipv6-parse")) {
            std::print("\nIPv6 parse (random):\n");
            collect_benchmark_results(100000);
        }
        if (want("ipv4-parse")) {
            std::print("\nIPv4 parse (random):\n");
            collect_ipv4_benchmark_results(100000);
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
