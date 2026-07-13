#ifndef SERIALIZE_IPV6_H
#define SERIALIZE_IPV6_H

#include <cstdint>

#include "serialize_ipv4.h"

/**
 * IPv6 serialization (the inverse of parse_ipv6_avx512): 16 bytes in network
 * order -> canonical text form.
 *
 * The output matches inet_ntop / RFC 5952: lowercase hex, leading zeros within
 * a hextet omitted, the leftmost-longest run of >= 2 zero hextets replaced by
 * "::", and the IPv4-mapped / IPv4-compatible tails printed in dotted-decimal
 * (reusing serialize_ipv4_simd). The zero-run search and "::" placement are
 * inherently control-flow bound, so this is a scalar routine with a branchless
 * hextet-to-hex helper; the embedded IPv4 tail still goes through the SIMD path.
 */

// Writes hextet w (0..0xffff) as 1-4 lowercase hex digits, no leading zeros.
static inline char *serialize_hextet(char *p, unsigned w) {
    static const char d[] = "0123456789abcdef";
    if (w >= 0x1000) { *p++ = d[(w >> 12) & 0xf]; }
    if (w >= 0x100)  { *p++ = d[(w >> 8) & 0xf]; }
    if (w >= 0x10)   { *p++ = d[(w >> 4) & 0xf]; }
    *p++ = d[w & 0xf];
    return p;
}

/**
 * @param src Pointer to 16 bytes in network order.
 * @param out Output buffer of at least 46 bytes; NUL-terminated on return.
 * @return The number of characters written (excluding the NUL).
 */
static int serialize_ipv6(const uint8_t *src, char *out) {
    unsigned words[8];
    for (int i = 0; i < 8; i++) {
        words[i] = (unsigned(src[2 * i]) << 8) | src[2 * i + 1];
    }

    // Find the leftmost longest run of zero hextets (length >= 2).
    int best_base = -1, best_len = 0, cur_base = -1, cur_len = 0;
    for (int i = 0; i < 8; i++) {
        if (words[i] == 0) {
            if (cur_base == -1) { cur_base = i; cur_len = 1; }
            else { cur_len++; }
        } else if (cur_base != -1) {
            if (cur_len > best_len) { best_base = cur_base; best_len = cur_len; }
            cur_base = -1;
        }
    }
    if (cur_base != -1 && cur_len > best_len) {
        best_base = cur_base;
        best_len = cur_len;
    }
    if (best_len < 2) { best_base = -1; }

    char *p = out;
    for (int i = 0; i < 8; i++) {
        if (best_base != -1 && i >= best_base && i < best_base + best_len) {
            if (i == best_base) { *p++ = ':'; }
            continue;
        }
        if (i != 0) { *p++ = ':'; }
        // IPv4-mapped / IPv4-compatible tail: print the last 32 bits as dotted
        // decimal, matching inet_ntop's special case.
        if (i == 6 && best_base == 0 &&
            (best_len == 6 || (best_len == 7 && words[7] != 0x0001) ||
             (best_len == 5 && words[5] == 0xffff))) {
            p += serialize_ipv4(src + 12, p);
            *p = '\0';
            return int(p - out);
        }
        p = serialize_hextet(p, words[i]);
    }
    if (best_base != -1 && best_base + best_len == 8) { *p++ = ':'; }
    *p = '\0';
    return int(p - out);
}

#endif
