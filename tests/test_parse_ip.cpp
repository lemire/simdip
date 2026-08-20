// Tests for include/parse_ip.h.
//
// Builds standalone; it needs no dependency beyond the headers:
//
//   c++ -std=c++23 -O2 -Iinclude -o test_parse_ip tests/test_parse_ip.cpp
//   c++ -std=c++23 -O2 -march=native -Iinclude -o test_parse_ip tests/test_parse_ip.cpp
//
// Without AVX-512 this exercises the scalar fallbacks, which is still a real
// test of the dispatch and of the API.

#include "parse_ip.h"

#include <arpa/inet.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using simdip::ip_address;
using simdip::ipv4;
using simdip::ipv6;
using simdip::parse_error;
using simdip::parse_ip;

static int failures = 0;

static void fail(std::string_view what, std::string_view input) {
    std::printf("FAIL: %.*s  (input \"%.*s\")\n", (int)what.size(), what.data(),
                (int)input.size(), input.data());
    failures++;
}

// --- expectations -----------------------------------------------------------

static void expect_v4(std::string_view input, const char *dotted) {
    ipv4 want = 0;
    if (inet_pton(AF_INET, dotted, &want) != 1) {
        fail("bad test vector", dotted);
        return;
    }
    auto r = parse_ip(input);
    if (!r) { return fail("rejected, expected IPv4", input); }
    auto *got = std::get_if<ipv4>(&*r);
    if (got == nullptr) { return fail("parsed as IPv6, expected IPv4", input); }
    if (*got != want) { return fail("wrong IPv4 value", input); }
}

static void expect_v6(std::string_view input, const char *canonical) {
    ipv6 want{};
    if (inet_pton(AF_INET6, canonical, want.data()) != 1) {
        fail("bad test vector", canonical);
        return;
    }
    auto r = parse_ip(input);
    if (!r) { return fail("rejected, expected IPv6", input); }
    auto *got = std::get_if<ipv6>(&*r);
    if (got == nullptr) { return fail("parsed as IPv4, expected IPv6", input); }
    if (*got != want) { return fail("wrong IPv6 value", input); }
}

static void expect_error(std::string_view input, parse_error want) {
    auto r = parse_ip(input);
    if (r) { return fail("accepted, expected rejection", input); }
    if (r.error() != want) { return fail("wrong parse_error", input); }
}

// --- the input-shape claim the dispatch rests on ----------------------------

// Every input either parser accepts must be routed correctly. For inputs both
// reject, the route does not matter -- which is exactly what makes the 5-byte
// window sound. This checks the claim directly: over the whole string, does a
// colon occur at all, and if so, is the first one within the first five bytes?
static void check_window(std::string_view input) {
    const std::size_t first = input.find(':');
    const bool routed_v6 = simdip::detail::is_ipv6_form(input);
    if (routed_v6 != (first != std::string_view::npos && first < 5)) {
        fail("is_ipv6_form disagrees with a literal scan", input);
    }
    // Anything accepted as IPv6 must have had its first colon in the window.
    ipv6 scratch{};
    if (inet_pton(AF_INET6, std::string(input).c_str(), scratch.data()) == 1 &&
        !routed_v6) {
        fail("valid IPv6 routed to the IPv4 parser", input);
    }
    ipv4 scratch4 = 0;
    if (inet_pton(AF_INET, std::string(input).c_str(), &scratch4) == 1 &&
        routed_v6) {
        fail("valid IPv4 routed to the IPv6 parser", input);
    }
}

int main() {
    // --- IPv4, strict ---
    expect_v4("0.0.0.0", "0.0.0.0");
    expect_v4("1.2.3.4", "1.2.3.4");
    expect_v4("93.184.216.34", "93.184.216.34");
    expect_v4("192.168.0.1", "192.168.0.1");
    expect_v4("255.255.255.255", "255.255.255.255");

    // --- IPv4, permissive (inet_aton / WHATWG): accepted, unlike inet_pton ---
    expect_v4("192.168.1", "192.168.0.1");
    expect_v4("0x7f000001", "127.0.0.1");
    expect_v4("0177.0.0.1", "127.0.0.1");
    expect_v4("127.1", "127.0.0.1");
    expect_v4("0", "0.0.0.0");

    // --- IPv6 ---
    expect_v6("::", "::");
    expect_v6("::1", "::1");
    expect_v6("1::", "1::");
    expect_v6("2001:db8::1", "2001:db8::1");
    expect_v6("2001:0db8:0000:0000:0000:0000:0000:0001", "2001:db8::1");
    expect_v6("fe80::1", "fe80::1");
    expect_v6("1:2:3:4:5:6:7:8", "1:2:3:4:5:6:7:8");
    expect_v6("2001:DB8::1", "2001:db8::1");  // upper case
    expect_v6("::ffff:192.168.1.1", "::ffff:192.168.1.1");  // embedded IPv4
    expect_v6("64:ff9b::192.0.2.33", "64:ff9b::192.0.2.33");  // NAT64
    expect_v6("1:2:3:4:5:6:1.2.3.4", "1:2:3:4:5:6:1.2.3.4");

    // --- rejected, and rejected as the right family ---
    expect_error("", parse_error::empty);
    expect_error("hello", parse_error::invalid_ipv4);
    expect_error("1.2.3.4.5", parse_error::invalid_ipv4);
    expect_error("256.1.1.1", parse_error::invalid_ipv4);
    expect_error("1.2.3.4:80", parse_error::invalid_ipv4);   // colon at index 7
    expect_error("12345:6::1", parse_error::invalid_ipv4);   // colon at index 5
    expect_error(":", parse_error::invalid_ipv6);
    expect_error(":::", parse_error::invalid_ipv6);
    expect_error("1::2::3", parse_error::invalid_ipv6);      // two "::"
    expect_error("1:2:3:4:5:6:7:8:9", parse_error::invalid_ipv6);
    expect_error("::12345", parse_error::invalid_ipv6);      // 5-digit group
    expect_error("[::1]", parse_error::invalid_ipv6);        // brackets not accepted
    expect_error("fe80::1%eth0", parse_error::invalid_ipv6); // zone id not accepted

    // --- dispatch window, over every literal used above plus odd shapes ---
    for (std::string_view s :
         {"0.0.0.0", "255.255.255.255", "0x7f000001", "192.168.1", "::", "::1",
          "1::", "a::", "abcd::1", "abcde::1", "1:2:3:4:5:6:7:8", "1.2.3.4:80",
          "12345:6::1", ":", "1", "1.", "1.2", "1.2.3", "::ffff:1.2.3.4",
          "fe80::1%eth0", "[::1]", "hello", "hello:world"}) {
        check_window(s);
    }

    // --- randomised cross-check against inet_pton, both families ---
    std::mt19937_64 rng(42);
    char text[INET6_ADDRSTRLEN];
    for (int i = 0; i < 200000; i++) {
        if (i % 2 == 0) {
            std::uint32_t bits = (std::uint32_t)rng();
            inet_ntop(AF_INET, &bits, text, sizeof(text));
            expect_v4(text, text);
        } else {
            ipv6 bytes{};
            for (int b = 0; b < 16; b += 8) {
                std::uint64_t chunk = rng();
                std::memcpy(bytes.data() + b, &chunk, 8);
            }
            inet_ntop(AF_INET6, bytes.data(), text, sizeof(text));
            expect_v6(text, text);
            check_window(text);
        }
        if (failures > 20) { break; }
    }

    // --- the usage pattern from the API sketch ---
    if (auto r = parse_ip("2001:db8::1")) {
        if (auto *v4 = std::get_if<ipv4>(&*r)) {
            fail("dispatch", "2001:db8::1");
            (void)v4;
        } else {
            const ipv6 &v6 = std::get<ipv6>(*r);
            if (v6[0] != 0x20 || v6[1] != 0x01) { fail("value", "2001:db8::1"); }
        }
    } else {
        fail("rejected", "2001:db8::1");
    }

    if (failures == 0) {
        std::printf("all tests passed\n");
        return 0;
    }
    std::printf("%d failure(s)\n", failures);
    return 1;
}
