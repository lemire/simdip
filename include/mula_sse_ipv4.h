// Vendored from https://github.com/WojciechMula/toys/blob/master/parseip4/sse.cpp
// (function `sse_parse_ipv4`).
//
// SSE IPv4 text-to-bytes parser by Wojciech Muła, described in
//   "Parsing IP addresses" (2023), https://0x80.pl/notesen/2023-04-09-faster-parse-ip.html
// Reproduced here as the classic vectorized (SSE) prior art for IPv4 parsing.
//
// The upstream code returns a `result` struct and takes a std::string; the body
// below is the same algorithm rewritten to the project's plain
// (const char*, size_t, uint32_t*) -> int signature so it can sit next to the
// other parsers benchmarked here. All credit to Wojciech Muła.
//
// NOTE: like the original, this uses an unaligned 16-byte load of the input, so
// the caller must guarantee 16 readable bytes at `input`. Every IPv4 address is
// at most 15 characters, so a libstdc++/libc++ std::string (small-string buffer
// of 16 bytes, or a NUL-terminated heap buffer) always satisfies this.
#ifndef MULA_SSE_IPV4_H
#define MULA_SSE_IPV4_H

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

/**
 * Parses an IPv4 address in strict dotted-decimal form (four decimal octets,
 * no leading zeros beyond a bare "0", no octal/hex) using SSE.
 *
 * The four octets are written to *out with the first octet in the low byte,
 * i.e. on a little-endian machine ((uint8_t*)out)[0] is the first octet, which
 * matches the network-order byte layout produced by inet_pton.
 *
 * @return 1 on success, 0 if the format is invalid.
 */
static inline int parse_ipv4_mula_sse(const char *input, size_t len, uint32_t *out) {
    if (len < 7 || len > 15) { return 0; }  // "0.0.0.0" .. "123.123.123.123"

    uint16_t mask = 0xffff;
    mask <<= len;
    mask = ~mask;

    const __m128i in = _mm_loadu_si128((const __m128i *)input);

    // 1. locate dots
    uint16_t dotmask;
    {
        const __m128i dot = _mm_set1_epi8('.');
        const __m128i t0 = _mm_cmpeq_epi8(in, dot);
        dotmask = uint16_t(_mm_movemask_epi8(t0));
        dotmask &= mask;
    }

    // there has to be exactly 3 dots
    if (__builtin_popcount(dotmask) != 3) { return 0; }

    // 2. validate that non-dot characters are digits '0'..'9'
    {
        const __m128i ascii0 = _mm_set1_epi8(char(-128 + '0'));
        const __m128i rangedigits = _mm_set1_epi8(char(-128 + ('9' - '0' + 1)));
        const __m128i t1 = _mm_sub_epi8(in, ascii0);
        const __m128i t2 = _mm_cmplt_epi8(t1, rangedigits);
        uint16_t less = uint16_t(_mm_movemask_epi8(t2));
        less &= mask;
        if ((less | dotmask) != mask) { return 0; }
    }

    // 3. add a virtual dot just past the last character
    dotmask |= uint16_t(uint16_t(1) << len);

    // 4. process the four components
    uint8_t byte[4];
    const uint8_t *data = (const uint8_t *)input;
    for (int i = 0; i < 4; i++) {
        const int n = __builtin_ctz(dotmask);
        switch (n) {
            case 1:
                byte[i] = uint8_t(data[0] - '0');
                data += 2;
                dotmask >>= 2;
                break;
            case 2: {
                const uint32_t tmp = 10 * (data[0] - '0') + (data[1] - '0');
                if (tmp < 10) { return 0; }  // leading zero
                byte[i] = uint8_t(tmp);
                data += 3;
                dotmask >>= 3;
                break;
            }
            case 3: {
                const uint32_t tmp =
                    100 * (data[0] - '0') + 10 * (data[1] - '0') + (data[2] - '0');
                if (tmp > 0xff) { return 0; }  // > 255
                if (tmp < 100) { return 0; }   // leading zero
                byte[i] = uint8_t(tmp);
                data += 4;
                dotmask >>= 4;
                break;
            }
            default:
                return 0;  // empty field or too many digits
        }
    }

    uint8_t *o = (uint8_t *)out;
    o[0] = byte[0];
    o[1] = byte[1];
    o[2] = byte[2];
    o[3] = byte[3];
    return 1;
}

#endif  // MULA_SSE_IPV4_H
