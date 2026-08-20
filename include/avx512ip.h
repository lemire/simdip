#ifndef AVX512IP_H
#define AVX512IP_H

#include <x86intrin.h>

#include <cstddef>
#include <cstdint>

// Scalar ada/WHATWG fallback used by the IPv4 SIMD path for unusual-but-valid
// forms (octal/hex segments, fewer than four parts).
#include "ada_ip.h"

/**
 * AVX-512 based parsing of IPv4 addresses in dotted-decimal form.
 *
 * Digit placement is computed from the delimiter positions, with no shuffle
 * table. The masked load fills the tail with '.', so the terminator at `len` is
 * a real dot and the marker mask is just the dot compare -- it stays in a k
 * register. Compress of the constant [0..15] by that mask yields (q₀,q₁,q₂,q₃),
 * the three dots and the terminator. Broadcasting each qᵢ, adding [−4,−3,−2,−1],
 * and signed-max with qᵢ₋₁ (and −1 in lane 0) builds a shuffle index that points
 * digit bytes at the right sources and pad bytes either at the previous dot
 * (zeroed in the digit vector) or, in octet 0, at a negative index. One
 * `pshufb` then expands each octet to [0, hundreds, tens, ones]: a negative
 * control byte zeroes its lane for free, which is exactly what octet 0's pads
 * want, so this needs no AVX512-VBMI.
 *
 * `_mm_dpbusd_epi32` (VNNI) multiplies by {0,100,10,1} and horizontally sums
 * into four 32-bit octet values.
 *
 * Validation (exactly three dots, octet lengths 1..3, all-digits, no leading
 * zero, no overflow) is done with AVX-512 mask compares; unusual-but-valid
 * forms fall back to the ada parser.
 *
 * Unlike simdzone (and Mula's SSE parser), which unconditionally read 16 bytes,
 * this uses an AVX-512 masked load and reads exactly `len` bytes -- no over-read
 * past the string, so it is safe on inputs that end near an unmapped page.
 *
 */

/**
 * Parses an IPv4 address in dotted-decimal form (e.g., "192.168.0.1") using an
 * AVX-512 SIMD path. Unusual-but-valid forms (octal, hex, leading zeros, fewer
 * than four parts) fall back to the ada parser.
 *
 * @param input Pointer to the input string (NUL termination is not required).
 * @param len Length of the input string.
 * @param ptr Output pointer to a uint32_t receiving the IPv4 value in host order.
 * @return 1 on success, 0 if the format is invalid.
 */
static int parse_ipv4_avx512vl(const char *input, size_t len, uint32_t *ptr) {
    // Reject len > 15 (a strict dotted-quad is at most "255.255.255.255"). The
    // terminator is the first byte of the '.'-filled tail, which only exists
    // while len < 16. We only need the upper bound: shorter invalid strings
    // fail the layout checks and fall through to ada.
    if (len > 15) [[unlikely]] { return parse_ipv4_ada(input, len, ptr); }

    const uint32_t len_mask = _bzhi_u32(0xFFFFFFFFu, (unsigned)len);
    // Masked load: bytes [0,len) come from input, the rest are filled with '.'.
    // Reads exactly `len` bytes -- no over-read past the string. Filling with a
    // dot makes the virtual terminator at `len` a real one, so the marker mask
    // is just the dot compare: no or, and the compare stays in a k register
    // instead of being rebuilt from a GPR partition.
    const __m128i dot_v = _mm_set1_epi8('.');
    const __m128i v = _mm_mask_loadu_epi8(dot_v, (__mmask16)len_mask, (const __m128i *)input);

    // Markers: the three dots plus the '.'-filled tail. Only the first four
    // compressed positions are read, so the extra tail dots are harmless.
    const __mmask16 delim = _mm_cmpeq_epi8_mask(v, dot_v);
    const uint32_t dots = (uint32_t)delim & len_mask;   // the real dots
    const uint32_t keep = len_mask & ~dots;             // digit lanes

    const __m128i zero_digit = _mm_set1_epi8('0');
    // Digit values; dots and the tail land above 9.
    const __m128i digits = _mm_sub_epi8(v, zero_digit);
    const __mmask16 is_digit = _mm_cmple_epu8_mask(digits, _mm_set1_epi8(9));
    // Junk in a digit slot (e.g. "1.2.3.:"): neither a digit nor a dot.
    const __mmask16 hole = _kandn_mask16(is_digit, (__mmask16)keep);
    // Zero every non-digit lane: pad fetches of a previous-dot index return 0.
    const __m128i v0 = _mm_maskz_mov_epi8(is_digit, digits);

    // c = vpcompressb(iota, D): bytes 0..3 are (q₀,q₁,q₂,q₃), the marker
    // positions. Compress does the rank scan that tzcnt/pdep would.
    const __m128i iota =
        _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i c = _mm_maskz_compress_epi8(delim, iota);

    // Broadcast qᵢ to lane i, add [−4,−3,−2,−1]: digit bytes point at the
    // right sources; pad bytes point at junk (previous dot, earlier digits,
    // or negative for lane 0).
    const __m128i k_rep =
        _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    const __m128i qi = _mm_shuffle_epi8(c, k_rep);
    // prev = qᵢ₋₁ per lane, −1 for lane 0 (palignr prepends −1, then the
    // same broadcast).
    const __m128i prev =
        _mm_shuffle_epi8(_mm_alignr_epi8(c, _mm_set1_epi8(-1), 15), k_rep);
    // Kill the pads without a mask: digits have idx ≥ qᵢ₋₁+1 and are
    // untouched; pads have idx ≤ qᵢ₋₁ and clamp onto the previous dot.
    // Octet 0's pads clamp to −1, whose bit 7 makes pshufb emit zero.
    const __m128i idx = _mm_max_epi8(
        _mm_add_epi8(qi, _mm_setr_epi8(-4, -3, -2, -1, -4, -3, -2, -1, -4, -3,
                                       -2, -1, -4, -3, -2, -1)),
        prev);
    // pshufb, not permb: every index is either a real position in [0,15] or
    // negative, and a negative control byte zeroes the lane for free. One cycle
    // instead of three, and no AVX512-VBMI needed.
    const __m128i padded = _mm_shuffle_epi8(v0, idx);

    // One VNNI dot-product does hundreds*100 + tens*10 + ones*1 per octet, each
    // landing in its own 32-bit lane. Weight bytes (LE): 0, 100, 10, 1.
    const __m128i res = _mm_dpbusd_epi32(_mm_setzero_si128(), padded,
                                         _mm_set1_epi32(0x010a6400));
    const __mmask8 over = _mm_cmpgt_epu32_mask(res, _mm_set1_epi32(0xff));

    // Octet lengths from marker gaps: [q₀, q₁−q₀, q₂−q₁, q₃−q₂] must be
    // [1..3, 2..4, 2..4, 2..4] (first gap is digits only; the rest include the
    // preceding dot). Unsigned (gap − min) > 2 catches both too-small (wrap)
    // and too-large. Combined with exactly-three-dots, this is the 81 valid
    // layouts and nothing else.
    const __m128i gap = _mm_sub_epi8(c, _mm_slli_si128(c, 1));
    const __mmask16 bad_gap = _mm_mask_cmpgt_epu8_mask(
        0x000F,
        _mm_sub_epi8(gap, _mm_setr_epi8(1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0)),
        _mm_set1_epi8(2));

    // Leading zero: a '0' at the start of an octet whose next char is a digit
    // (e.g. "01"). Bare "0" is fine -- the next char is a dot or the end.
    const uint32_t zero_bits = (uint32_t)_mm_cmpeq_epi8_mask(v, zero_digit);
    const uint32_t start_bits = (dots << 1) | 1u;

    int error = (_mm_popcnt_u32(dots) != 3);
    error |= (bad_gap != 0);
    error |= (_kor_mask16(hole, (__mmask16)over) != 0);
    error |= ((zero_bits & start_bits & (keep >> 1)) != 0);

    if (!error) [[likely]] {
        // Gather the low byte of each 32-bit octet into bytes 0..3 (network
        // order) and store.
        _mm_storeu_si32(ptr, _mm_shuffle_epi8(res, _mm_setr_epi8(0, 4, 8, 12, 0, 0,
                                                                 0, 0, 0, 0, 0, 0,
                                                                 0, 0, 0, 0)));
        return 1;
    }

    // The SIMD path is strict: it rejects valid-but-unusual forms (octal, hex,
    // leading zeros, fewer than four parts). Before declaring failure, give the
    // permissive ada parser a chance to accept the address.
    return parse_ipv4_ada(input, len, ptr);
}

#endif
