#ifndef ADA_IPV6_HEAD_H
#define ADA_IPV6_HEAD_H
//
// ada's *current* IPv6 host parser, lifted from ada-url/ada @ fa9a175
// (include/ada/url_ip-inl.h and src/url.cpp::url::parse_ipv6) and reshaped to
// this project's (src, len, out16) -> int signature.
//
// This is what people mean by "ada's AVX-512 IPv6 parser", and it is worth
// being precise about what that is: ada does *not* parse IPv6 with SIMD. It
// runs a scalar WHATWG piece parser and puts one 512-bit *prefilter* in front
// of it -- `ipv6_structure_plausible` -- which does a single masked load,
// counts colons and dots, and rejects shapes no valid address can have. Every
// accepted address is then parsed byte at a time exactly as before. So the
// vector unit only ever removes work on *invalid* input; on valid input the
// prefilter is pure added cost (a load, two compares, a popcount) that buys
// nothing, and the parse itself never leaves the scalar pipeline.
//
// Two entry points so the prefilter's contribution is measurable on its own:
//   parse_ipv6_ada_head        -- as shipped (prefilter on AVX-512BW+VL)
//   parse_ipv6_ada_head_scalar -- the same parser with the prefilter removed
//
// The one algorithmic difference from the older port in ada_ip.h is
// parse_hex_piece: a 256-byte nibble table with an unrolled 4-deep chain,
// instead of a loop over ada_is_ascii_hex_digit. The compress shift at the end
// is also a move rather than the older swap loop.
//
#include <cstddef>
#include <cstdint>
#include <array>

#if defined(__AVX512BW__) && defined(__AVX512VL__)
#include <immintrin.h>
#define ADA_HEAD_AVX512 1
#endif

namespace ada_head {

// 256-entry: 0xff = not hex, else nibble value.
inline constexpr std::array<uint8_t, 256> make_hex_nibble_table() noexcept {
    std::array<uint8_t, 256> t{};
    for (size_t i = 0; i < 256; ++i) { t[i] = 0xff; }
    for (size_t d = 0; d < 10; ++d) { t[size_t{'0'} + d] = uint8_t(d); }
    for (size_t d = 0; d < 6; ++d) {
        t[size_t{'a'} + d] = uint8_t(10 + d);
        t[size_t{'A'} + d] = uint8_t(10 + d);
    }
    return t;
}
inline constexpr auto hex_nibble = make_hex_nibble_table();

// Parse up to 4 hex digits. Returns digit count (0 if none).
inline int parse_hex_piece(const char *&pointer, const char *end,
                           uint16_t &value) noexcept {
    if (pointer == end) { return 0; }
    const uint8_t n0 = hex_nibble[(unsigned char)*pointer];
    if (n0 == 0xff) { return 0; }
    uint32_t v = n0;
    ++pointer;
    int length = 1;
    if (pointer != end) {
        const uint8_t n1 = hex_nibble[(unsigned char)*pointer];
        if (n1 != 0xff) {
            v = (v << 4) | n1;
            ++pointer;
            ++length;
            if (pointer != end) {
                const uint8_t n2 = hex_nibble[(unsigned char)*pointer];
                if (n2 != 0xff) {
                    v = (v << 4) | n2;
                    ++pointer;
                    ++length;
                    if (pointer != end) {
                        const uint8_t n3 = hex_nibble[(unsigned char)*pointer];
                        if (n3 != 0xff) {
                            v = (v << 4) | n3;
                            ++pointer;
                            ++length;
                        }
                    }
                }
            }
        }
    }
    value = uint16_t(v);
    return length;
}

#if defined(ADA_HEAD_AVX512)
// Classify an IPv6 host (no brackets) with one masked 512-bit load.
// Returns false if the colon/dot shape is impossible.
inline bool ipv6_structure_plausible(const char *data, size_t len) noexcept {
    if (len < 2 || len > 45) { return false; }
    const __mmask64 live = (__mmask64)((1ULL << len) - 1ULL);
    const __m512i input = _mm512_maskz_loadu_epi8(live, (const void *)data);
    const __mmask64 is_colon =
        _mm512_mask_cmpeq_epi8_mask(live, input, _mm512_set1_epi8(':'));
    const __mmask64 is_dot =
        _mm512_mask_cmpeq_epi8_mask(live, input, _mm512_set1_epi8('.'));
    const int colons = (int)_mm_popcnt_u64((uint64_t)is_colon);
    if (colons > 8) { return false; }
    const uint64_t doubles = (uint64_t)is_colon & ((uint64_t)is_colon << 1);
    if (doubles != 0 && (doubles & (doubles - 1)) != 0) {
        return false;  // more than one "::" (or ":::...")
    }
    const bool has_double = doubles != 0;
    const bool has_dot = is_dot != 0;
    if (!has_double && !has_dot && colons != 7) { return false; }
    return true;
}
#endif

// The WHATWG piece parser, verbatim from src/url.cpp except that the result is
// written as 16 network-order bytes instead of being serialized into a host
// string. `prefilter` selects whether the 512-bit shape check runs.
template <bool prefilter>
static inline int parse_ipv6_ada_head_impl(const char *input_ptr,
                                           size_t input_len, uint8_t *ptr) {
    if (input_len == 0 || input_len > 45) { return 0; }
#if defined(ADA_HEAD_AVX512)
    if constexpr (prefilter) {
        if (!ipv6_structure_plausible(input_ptr, input_len)) { return 0; }
    }
#endif
    uint16_t address[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const char *pointer = input_ptr;
    const char *const end = pointer + input_len;
    int piece_index = 0;
    int compress = -1;

    if (*pointer == ':') {
        if (input_len == 1 || pointer[1] != ':') { return 0; }
        pointer += 2;
        compress = ++piece_index;
    }

    while (pointer != end) {
        if (piece_index == 8) { return 0; }
        if (*pointer == ':') {
            if (compress != -1) { return 0; }
            ++pointer;
            compress = ++piece_index;
            continue;
        }

        uint16_t value = 0;
        const int length = parse_hex_piece(pointer, end, value);

        if (pointer != end && *pointer == '.') {
            if (length == 0) { return 0; }
            pointer -= length;
            if (piece_index > 6) { return 0; }

            int numbers_seen = 0;
            while (pointer != end) {
                int ipv4_piece = -1;
                if (numbers_seen > 0) {
                    if (*pointer == '.' && numbers_seen < 4) {
                        ++pointer;
                    } else {
                        return 0;
                    }
                }
                if (pointer == end || *pointer < '0' || *pointer > '9') { return 0; }
                ipv4_piece = *pointer - '0';
                ++pointer;
                if (pointer != end && *pointer >= '0' && *pointer <= '9') {
                    if (ipv4_piece == 0) { return 0; }
                    ipv4_piece = ipv4_piece * 10 + (*pointer - '0');
                    ++pointer;
                    if (pointer != end && *pointer >= '0' && *pointer <= '9') {
                        ipv4_piece = ipv4_piece * 10 + (*pointer - '0');
                        ++pointer;
                        if (ipv4_piece > 255) { return 0; }
                    }
                }
                address[piece_index] =
                    uint16_t(address[piece_index] * 0x100 + uint16_t(ipv4_piece));
                ++numbers_seen;
                if (numbers_seen == 2 || numbers_seen == 4) { ++piece_index; }
            }
            if (numbers_seen != 4) { return 0; }
            break;
        }

        if (length == 0) { return 0; }

        if (pointer != end && *pointer == ':') {
            ++pointer;
            if (pointer == end) { return 0; }
        } else if (pointer != end) {
            return 0;
        }

        address[piece_index] = value;
        ++piece_index;
    }

    if (compress != -1) {
        const int right = piece_index - compress;
        if (right > 0) {
            const size_t dest = size_t(8 - right);
            const size_t src = size_t(compress);
            if (dest != src) {
                for (size_t i = size_t(right); i-- > 0;) {
                    address[dest + i] = address[src + i];
                    address[src + i] = 0;
                }
            }
        }
    } else if (piece_index != 8) {
        return 0;
    }

    for (int i = 0; i < 8; i++) {
        ptr[2 * i] = uint8_t(address[i] >> 8);
        ptr[2 * i + 1] = uint8_t(address[i] & 0xff);
    }
    return 1;
}

// As shipped by ada on an AVX-512BW+VL build.
static inline int parse_ipv6_ada_head(const char *s, size_t n, uint8_t *out) {
    return parse_ipv6_ada_head_impl<true>(s, n, out);
}
// The same parser with the 512-bit prefilter removed.
static inline int parse_ipv6_ada_head_scalar(const char *s, size_t n, uint8_t *out) {
    return parse_ipv6_ada_head_impl<false>(s, n, out);
}

}  // namespace ada_head

#endif  // ADA_IPV6_HEAD_H
