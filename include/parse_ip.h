#ifndef SIMDIP_PARSE_IP_H
#define SIMDIP_PARSE_IP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string_view>
#include <variant>

/**
 * A family-agnostic entry point: one call that parses either an IPv4 or an
 * IPv6 address and reports which one it found.
 *
 * The existing headers each parse a single family and use the project's
 * (const char *, size_t, out) -> int convention. This header adds a C++23
 * surface on top of them -- std::expected<std::variant<...>, parse_error> --
 * and the family dispatch that has to run in front of them. It includes the
 * existing headers but does not modify them; when the target lacks AVX-512 it
 * falls back to the portable scalar parsers in ada_ip.h, so this header also
 * compiles (and stays correct) on non-x86 hosts.
 *
 * Dispatch. A colon settles the question: ':' cannot occur anywhere in an IPv4
 * address, in either the strict dotted-quad grammar or the permissive
 * inet_aton/WHATWG one, and every IPv6 address contains at least one. Looking
 * for a dot instead would be wrong, since the embedded-IPv4 form
 * ("::ffff:192.168.1.1") has dots.
 *
 * Only the first five bytes need to be examined. An IPv6 address starts either
 * with "::" or with a group of one to four hexadecimal digits followed by a
 * colon (RFC 3986: h16 = 1*4HEXDIG), so in any valid IPv6 string the first
 * colon sits at index 0..4. Bracketed ("[::1]") and zone-id ("fe80::1%eth0")
 * spellings keep it in that window too. A colon that first appears at index 5
 * or later therefore belongs to an input that is not a valid address of either
 * family: sending it down the IPv4 path is safe, because that parser rejects
 * it. Misrouting can only ever happen to input that both parsers reject.
 *
 * That bound is what makes the test cheap: one 8-byte load and the classic
 * zero-byte SWAR trick, no loop and no data-dependent branch.
 *
 * Note that on an AVX-512 target the IPv6 parser recomputes a colon mask over
 * the whole string as its first step, so a fused entry point -- one masked
 * load feeding both the dispatch and the parser -- could drop the separate
 * probe. That would mean duplicating both parser bodies here, so it is left
 * undone; the probe is a couple of cycles.
 */

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && \
    defined(__AVX512VBMI__) && defined(__AVX512VNNI__)
#define SIMDIP_HAS_AVX512 1
#endif
#endif

#include "ada_ip.h"  // parse_ipv4_ada, parse_ipv6_ada (portable, scalar)
#ifdef SIMDIP_HAS_AVX512
#include "avx512ip.h"     // parse_ipv4_avx512vl  (SIMD, ada fallback)
#include "vtlmks_ipv6.h"  // vtlmks::parse_ipv6_avx512 (SIMD, scalar fallback)
#endif

namespace simdip {

/**
 * A parsed IPv4 address, in network byte order: the four octets occupy bytes
 * 0..3 of the object in the order they were written, which is the layout of
 * `struct in_addr`. Use ntohl() if you want the value as a host-order integer.
 * This matches every other parser in the project.
 */
using ipv4 = std::uint32_t;

/** A parsed IPv6 address: 16 bytes in network order. */
using ipv6 = std::array<std::uint8_t, 16>;

/** Either family. Which one is a property of the input, not of the caller. */
using ip_address = std::variant<ipv4, ipv6>;

/**
 * Why parsing failed. The family is part of the error: dispatch happens before
 * parsing, so a rejected input was rejected as one family or the other, and
 * saying which is more useful to the caller than a bare "invalid".
 */
enum class parse_error : std::uint8_t {
    empty,         ///< the input has no characters
    invalid_ipv4,  ///< no colon in the first five bytes, and not a valid IPv4 address
    invalid_ipv6,  ///< a colon in the first five bytes, and not a valid IPv6 address
};

namespace detail {

/**
 * True if `input` should be handed to the IPv6 parser: it has a ':' among its
 * first five bytes. See the note at the top of this header for why five bytes
 * decide it, and why the answer is exact for every input either parser
 * accepts.
 *
 * Branch-free: the bytes beyond min(len,5) stay zero, and 0x00 ^ ':' is
 * non-zero, so the padding can never produce a hit and needs no masking. The
 * has-a-zero-byte identity is exact (no false positives), and the whole thing
 * is endianness-independent, since it only ever asks whether some byte lane is
 * zero.
 *
 * @param input a non-empty string.
 */
[[nodiscard]] inline bool is_ipv6_form(std::string_view input) noexcept {
    std::uint64_t w = 0;
    const std::size_t n = input.size() < 5 ? input.size() : 5;
    std::memcpy(&w, input.data(), n);
    w ^= UINT64_C(0x3a3a3a3a3a3a3a3a);  // ':' is 0x3a
    return ((w - UINT64_C(0x0101010101010101)) & ~w &
            UINT64_C(0x8080808080808080)) != 0;
}

/** Fastest available IPv4 parser for this target. Returns 1 on success. */
[[nodiscard]] inline int parse_v4(const char *p, std::size_t n,
                                  ipv4 *out) noexcept {
#ifdef SIMDIP_HAS_AVX512
    return parse_ipv4_avx512vl(p, n, out);
#else
    return parse_ipv4_ada(p, n, out);
#endif
}

/** Fastest available IPv6 parser for this target. Returns 1 on success. */
[[nodiscard]] inline int parse_v6(const char *p, std::size_t n,
                                  std::uint8_t *out) noexcept {
#ifdef SIMDIP_HAS_AVX512
    return vtlmks::parse_ipv6_avx512(p, n, out);
#else
    return parse_ipv6_ada(p, n, out);
#endif
}

}  // namespace detail

/**
 * Parses a textual IP address of either family.
 *
 * IPv4 accepts the permissive inet_aton/WHATWG forms (octal and hexadecimal
 * octets, fewer than four parts) as well as the strict dotted-quad, matching
 * the rest of the project. IPv6 accepts the RFC 4291 forms, including "::"
 * compression and an embedded IPv4 tail. Neither surrounding brackets nor a
 * zone identifier is accepted; strip them first if your input carries them.
 *
 * @param input the address text; NUL termination is not required.
 * @return the parsed address, or the reason it was rejected.
 */
[[nodiscard]] inline std::expected<ip_address, parse_error> parse_ip(
    std::string_view input) noexcept {
    if (input.empty()) [[unlikely]] {
        return std::unexpected(parse_error::empty);
    }
    if (detail::is_ipv6_form(input)) {
        ipv6 value{};
        if (detail::parse_v6(input.data(), input.size(), value.data())) {
            return ip_address{value};
        }
        return std::unexpected(parse_error::invalid_ipv6);
    }
    ipv4 value = 0;
    if (detail::parse_v4(input.data(), input.size(), &value)) {
        return ip_address{value};
    }
    return std::unexpected(parse_error::invalid_ipv4);
}

}  // namespace simdip

#endif  // SIMDIP_PARSE_IP_H
