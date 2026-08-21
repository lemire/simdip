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
 * This borrows the key trick from simdzone's branchless SSE parser
 * (simdzone_ipv4.h): instead of computing the digit placement dynamically (a
 * long dependency chain of compress / cvt / permutevar / movepi8 / compress /
 * expand), it recognises that the layout of an IPv4 address is fully determined
 * by the four octet lengths (each 1..3), giving only 81 possibilities. A perfect
 * hash of the delimiter bit-pattern selects a precomputed 16-byte shuffle that
 * positions the digits directly.
 *
 * Where it differs from (and improves on) simdzone:
 *  - Because the caller passes `len`, it skips simdzone's length-inference
 *    (saturate + clip_mask + popcnt): the delimiter partition that drives the
 *    hash is just `dots | (1 << len)`.
 *  - It arranges the shuffle so each octet lands as four little-endian bytes
 *    [0, hundreds, tens, ones], then uses a single `_mm_dpbusd_epi32` (VNNI) to
 *    multiply by {0,100,10,1} and horizontally sum into four 32-bit octet
 *    values -- replacing simdzone's maddubs + shuffle + adds + packus.
 *  - Validation (dot count/positions/length, all-digits, no leading zero, no
 *    overflow) is done with AVX-512 mask compares, and the two mask-domain
 *    checks are fused so the result crosses to a general register only once.
 *
 * Unlike simdzone (and Mula's SSE parser), which unconditionally read 16 bytes,
 * this uses an AVX-512 masked load and reads exactly `len` bytes -- no over-read
 * past the string, so it is safe on inputs that end near an unmapped page.
 *
 */

namespace avx512vl_ipv4 {

// Perfect-hash multiplier borrowed from simdzone: for the 81 valid delimiter
// bit-patterns, (partition * kHashMul) >> 24 yields 81 distinct keys in [0,256).
static constexpr uint32_t kHashMul = 0x00CF7800u;

struct Tables {
    // Both tables are indexed directly by the 8-bit hash key. Because the hash
    // is a perfect hash over the 81 valid layouts, each valid layout owns a
    // distinct key, so no id-compaction indirection (and no second dependent
    // load) is needed. Keys that no valid layout maps to hold a zeroed slot.

    // per-key 16-byte shuffle: octet i occupies bytes [4i..4i+3] as
    // [0x80 (->0), hundreds, tens, ones]; 0x80 lanes shuffle to zero.
    uint8_t pat[256][16];
    // per-key validation word, fetched in a single load: the low 16 bits are the
    // canonical delimiter partition (three dots + terminator); the high 16 bits
    // are a mask of the leading digit lane of each multi-digit octet.
    //
    //  - partition compare: the hash is only collision-free over the 81 valid
    //    layouts, so an invalid layout (e.g. a 4-digit octet) may hash onto a
    //    key a valid layout also uses. Comparing the actual partition against
    //    this pinned value rejects every impostor and, in one compare, also
    //    validates the dot count, dot positions, and length. An unused key holds
    //    0, which no real (always non-zero) partition matches.
    //  - leading lane mask: used to reject leading zeros (e.g. "01").
    uint32_t aux[256];
};

// Builds both tables at compile time. Enumerates the 81 octet-length
// combinations; a `throw` (which makes the function non-constant, i.e. a compile
// error) fires if the perfect hash ever collides.
constexpr Tables make_tables() {
    Tables t{};
    for (int k = 0; k < 256; k++) {
        for (int b = 0; b < 16; b++) { t.pat[k][b] = 0x80; }
    }
    for (int l0 = 1; l0 <= 3; l0++)
    for (int l1 = 1; l1 <= 3; l1++)
    for (int l2 = 1; l2 <= 3; l2++)
    for (int l3 = 1; l3 <= 3; l3++) {
        const int s0 = 0;
        const int s1 = l0 + 1;
        const int s2 = l0 + l1 + 2;
        const int s3 = l0 + l1 + l2 + 3;
        const int len = s3 + l3;
        // Delimiter positions: the three dots plus a virtual terminator at `len`.
        const uint32_t partition = (1u << (uint32_t)l0) |
                                   (1u << (uint32_t)(l0 + l1 + 1)) |
                                   (1u << (uint32_t)(l0 + l1 + l2 + 2)) |
                                   (1u << (uint32_t)len);
        const uint32_t hk = (uint32_t)(partition * kHashMul) >> 24;
        if (t.aux[hk] != 0) { throw "avx512vl_ipv4: perfect-hash collision"; }

        const int starts[4] = {s0, s1, s2, s3};
        const int lens[4] = {l0, l1, l2, l3};
        uint16_t lead = 0;
        for (int oc = 0; oc < 4; oc++) {
            const int base = 4 * oc;
            const int s = starts[oc];
            const int l = lens[oc];
            t.pat[hk][base + 3] = (uint8_t)(s + l - 1);              // ones
            if (l >= 2) { t.pat[hk][base + 2] = (uint8_t)(s + l - 2); }  // tens
            if (l == 3) { t.pat[hk][base + 1] = (uint8_t)(s); }         // hundreds
            if (l == 3) { lead |= (uint16_t)(1u << (uint32_t)(base + 1)); }
            else if (l == 2) { lead |= (uint16_t)(1u << (uint32_t)(base + 2)); }
        }
        t.aux[hk] = (uint32_t)partition | ((uint32_t)lead << 16);
    }
    return t;
}

static constexpr Tables T = make_tables();

}  // namespace avx512vl_ipv4

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
    // Reject len > 15 (a strict dotted-quad is at most "255.255.255.255"). This
    // upper bound is correctness-critical: for len >= 16 the terminator bit
    // (1<<len) leaves the 16-bit partition window -- and for len >= 32 _bzhi_u32
    // saturates so it wraps to 0 and can alias a canonical partition, which would
    // false-accept a long string. We only need the upper bound: for len < 7 the
    // terminator sits below bit 7, beneath every canonical's high bit, so the
    // fast path always declines and defers correctly (just one wasted pass).
    if (len > 15) [[unlikely]] { return parse_ipv4_ada(input, len, ptr); }

    // len_mask = (1<<len)-1 in a single BZHI (zero all bits from position len up),
    // shaving the shift+decrement off the head of the critical path.
    const uint32_t len_mask = _bzhi_u32(0xFFFFFFFFu, (unsigned)len);
    const __m128i zero_digit = _mm_set1_epi8('0');
    // Masked load: bytes [0,len) come from input, the rest are filled with '0'.
    // Reads exactly `len` bytes -- no over-read past the string. Filling with a
    // digit (rather than '.') means the padding lanes are neither dots nor
    // non-digits, so `dots` below needs no masking and the digit check stays clean.
    const __m128i str = _mm_mask_loadu_epi8(zero_digit, (__mmask16)len_mask, (const __m128i *)input);

    const __mmask16 dots = _mm_cmpeq_epi8_mask(str, _mm_set1_epi8('.'));  // the real dots

    // Delimiter pattern = the dots + a terminator bit at position `len`, then the
    // same perfect hash simdzone uses. (1u<<len) == len_mask + 1. Indexing the
    // tables by the 8-bit hash key directly (rather than a compacted id) keeps
    // the critical path to a single dependent table load.
    const uint32_t partition = (uint32_t)dots | (len_mask + 1u);
    const uint32_t hash_key = (uint32_t)(partition * avx512vl_ipv4::kHashMul) >> 24;
    const uint32_t aux = avx512vl_ipv4::T.aux[hash_key];  // low: partition, high: lead

    // The partition must be exactly the canonical one for this key. This rejects
    // any layout that is not four octets of 1..3 digits (wrong dot count/spacing,
    // 4-digit octets, empty octets, wrong length) in a single compare.
    int error = (partition != (aux & 0xFFFFu));

    // Digit values: '0'..'9' -> 0..9; every non-digit maps to a byte > 9. `hole`
    // marks lanes that are neither a digit nor a dot -- i.e. junk in a digit slot
    // (e.g. "1.2.3.:"); a valid address has none. It is `~dots & bad` (kandn):
    // padding lanes are '0' (a digit, not in `bad`), and dots are excluded, so
    // only stray non-digits survive. Kept in the mask domain to fuse below.
    const __m128i digits = _mm_sub_epi8(str, zero_digit);
    const __mmask16 bad = _mm_cmpgt_epu8_mask(digits, _mm_set1_epi8(9));
    const __mmask16 hole = _kandn_mask16(dots, bad);

    // Position the digits: each octet -> [0, hundreds, tens, ones] little-endian.
    const __m128i shuf = _mm_loadu_si128((const __m128i *)avx512vl_ipv4::T.pat[hash_key]);
    const __m128i padded = _mm_shuffle_epi8(digits, shuf);

    // One VNNI dot-product does hundreds*100 + tens*10 + ones*1 per octet, each
    // landing in its own 32-bit lane. Weight bytes (LE): 0, 100, 10, 1.
    const __m128i res = _mm_dpbusd_epi32(_mm_setzero_si128(), padded,
                                         _mm_set1_epi32(0x010a6400));
    const __mmask8 over = _mm_cmpgt_epu32_mask(res, _mm_set1_epi32(0xff));  // octet > 255

    // Fuse the two mask-domain checks (junk lanes + octet overflow) into one
    // k-register and cross to the GPR error accumulator exactly once. `over` is
    // the tail of the dpbusd critical chain, so folding it in with a 1-cycle
    // kor + kortest beats sinking it through a 3-cycle k->GPR move on its own.
    error |= (_kor_mask16(hole, (__mmask16)over) != 0);

    // Leading-zero check: `lead` (aux >> 16) marks only the most-significant
    // lane of each multi-digit octet, and those lanes always hold a real digit
    // (never padding), so a zero there is exactly a leading zero like "01".
    const __m128i zero_lane = _mm_cmpeq_epi8(_mm_setzero_si128(), padded);
    const uint32_t lz = (uint32_t)_mm_movemask_epi8(zero_lane);
    error |= ((lz & (aux >> 16)) != 0);

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

/**
 * Table-free AVX-512 IPv4 parse. Same contract as parse_ipv4_avx512vl: strict
 * dotted-quad on the fast path, ada fallback for unusual-but-valid forms.
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
 */
static int parse_ipv4_avx512vl_notab(const char *input, size_t len, uint32_t *ptr) {
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
        _mm_storeu_si32(ptr, _mm_shuffle_epi8(res, _mm_setr_epi8(0, 4, 8, 12, 0, 0,
                                                                 0, 0, 0, 0, 0, 0,
                                                                 0, 0, 0, 0)));
        return 1;
    }

    return parse_ipv4_ada(input, len, ptr);
}

/**
 * Table-free AVX-512 IPv4 parse, second cut. Same contract as
 * parse_ipv4_avx512vl_notab, but keeps validation in k-registers: q₃==len
 * instead of popcnt, junk/leading-zero/overflow fused with kor, one kortest
 * at the end.
 */
static int parse_ipv4_avx512vl_notab2(const char *input, size_t len, uint32_t *ptr) {
    if (len > 15) [[unlikely]] { return parse_ipv4_ada(input, len, ptr); }

    const __mmask16 len_k = (__mmask16)_bzhi_u32(0xFFFFFFFFu, (unsigned)len);
    const __m128i dot_v = _mm_set1_epi8('.');
    const __m128i v = _mm_mask_loadu_epi8(dot_v, len_k, (const __m128i *)input);

    const __mmask16 delim = _mm_cmpeq_epi8_mask(v, dot_v);

    const __m128i zero_digit = _mm_set1_epi8('0');
    const __m128i digits = _mm_sub_epi8(v, zero_digit);
    const __mmask16 is_digit = _mm_cmple_epu8_mask(digits, _mm_set1_epi8(9));
    const __m128i v0 = _mm_maskz_mov_epi8(is_digit, digits);

    const __m128i iota =
        _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i c = _mm_maskz_compress_epi8(delim, iota);

    const __m128i k_rep =
        _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    const __m128i qi = _mm_shuffle_epi8(c, k_rep);
    const __m128i prev =
        _mm_shuffle_epi8(_mm_alignr_epi8(c, _mm_set1_epi8(-1), 15), k_rep);
    const __m128i idx = _mm_max_epi8(
        _mm_add_epi8(qi, _mm_setr_epi8(-4, -3, -2, -1, -4, -3, -2, -1, -4, -3,
                                       -2, -1, -4, -3, -2, -1)),
        prev);
    const __m128i padded = _mm_shuffle_epi8(v0, idx);
    const __m128i res = _mm_dpbusd_epi32(_mm_setzero_si128(), padded,
                                         _mm_set1_epi32(0x010a6400));

    const __m128i gap = _mm_sub_epi8(c, _mm_slli_si128(c, 1));
    const __mmask16 bad_gap = _mm_mask_cmpgt_epu8_mask(
        0x000F,
        _mm_sub_epi8(gap, _mm_setr_epi8(1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0)),
        _mm_set1_epi8(2));

    // Exactly three dots: the rank-3 marker must be the terminator at `len`.
    // With fewer real dots, lane 3 holds a tail dot past len (or compress's
    // zero fill); with more, a real dot before len. The broadcast of `len`
    // issues at entry, well before `c` is ready.
    __mmask16 err =
        _mm_mask_cmpneq_epi8_mask(0x8, c, _mm_set1_epi8((char)len));
    err = _kor_mask16(err, bad_gap);
    // Junk in a digit slot: inside [0,len) yet neither digit nor dot.
    err = _kor_mask16(err, _kandn_mask16(_kor_mask16(delim, is_digit), len_k));
    // Leading zero: '0' at an octet start whose successor is a digit lane.
    // delim<<1 needs no len masking: shifted tail bits land where v is '.'.
    const __mmask16 zero_k = _mm_cmpeq_epi8_mask(v, zero_digit);
    const __mmask16 starts =
        _kor_mask16(_kshiftli_mask16(delim, 1), _cvtu32_mask16(1));
    err = _kor_mask16(
        err, _kand_mask16(zero_k,
                          _kand_mask16(starts, _kshiftri_mask16(is_digit, 1))));
    // Octet > 255 joins last: it waits on the VNNI result anyway.
    err = _kor_mask16(err, (__mmask16)_mm_cmpgt_epu32_mask(
                               res, _mm_set1_epi32(0xff)));

    if (_kortestz_mask16_u8(err, err)) [[likely]] {
        _mm_storeu_si32(ptr, _mm_shuffle_epi8(res, _mm_setr_epi8(0, 4, 8, 12, 0,
                                                                 0, 0, 0, 0, 0,
                                                                 0, 0, 0, 0, 0,
                                                                 0)));
        return 1;
    }
    return parse_ipv4_ada(input, len, ptr);
}

/**
 * Table-free AVX-512 IPv4 parse, third cut: _notab2's shape, but every check
 * that does not need the VNNI result is computed in general registers from the
 * dot bitmask, which is ready long before `c` is.
 *
 * The point is not that scalar ops are cheaper -- they are not, this needs more
 * of them -- but that they leave the compress/shuffle chain alone and issue on
 * ports the vector chain is not using. Both other variants put their layout
 * check downstream of `vpcompressb`; here nothing but the octet values is.
 */
static int parse_ipv4_avx512vl_notab3(const char *input, size_t len, uint32_t *ptr) {
    if (len > 15) [[unlikely]] { return parse_ipv4_ada(input, len, ptr); }

    const uint32_t len_mask = _bzhi_u32(0xFFFFFFFFu, (unsigned)len);
    const __m128i dot_v = _mm_set1_epi8('.');
    const __m128i v = _mm_mask_loadu_epi8(dot_v, (__mmask16)len_mask, (const __m128i *)input);

    // Everything below this line is ready ~6 cycles in, while the compress
    // chain still has ~10 to run.
    const __mmask16 delim = _mm_cmpeq_epi8_mask(v, dot_v);
    const uint32_t dots = (uint32_t)delim & len_mask;
    const uint32_t keep = len_mask & ~dots;

    const __m128i zero_digit = _mm_set1_epi8('0');
    const __m128i digits = _mm_sub_epi8(v, zero_digit);
    const __mmask16 is_digit = _mm_cmple_epu8_mask(digits, _mm_set1_epi8(9));
    const __m128i v0 = _mm_maskz_mov_epi8(is_digit, digits);

    // --- critical path: compress -> index -> pshufb -> dpbusd ---------------
    const __m128i iota =
        _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i c = _mm_maskz_compress_epi8(delim, iota);
    const __m128i k_rep =
        _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    const __m128i qi = _mm_shuffle_epi8(c, k_rep);
    const __m128i prev =
        _mm_shuffle_epi8(_mm_alignr_epi8(c, _mm_set1_epi8(-1), 15), k_rep);
    const __m128i idx = _mm_max_epi8(
        _mm_add_epi8(qi, _mm_setr_epi8(-4, -3, -2, -1, -4, -3, -2, -1, -4, -3,
                                       -2, -1, -4, -3, -2, -1)),
        prev);
    const __m128i padded = _mm_shuffle_epi8(v0, idx);
    const __m128i res = _mm_dpbusd_epi32(_mm_setzero_si128(), padded,
                                         _mm_set1_epi32(0x010a6400));

    // --- validation, scalar, entirely off the compress chain ---------------
    // Layout: exactly three dots, none leading, adjacent or trailing, and no
    // run of four digit lanes (a 4-digit octet). Together with len <= 15 that
    // is exactly the 81 valid layouts.
    uint32_t error = (uint32_t)(_mm_popcnt_u32(dots) ^ 3u);
    error |= dots & 1u;
    error |= dots & (dots << 1);
    error |= dots & (len_mask ^ (len_mask >> 1));
    error |= keep & (keep << 1) & (keep << 2) & (keep << 3);
    // Junk in a digit slot: in range, not a dot, not a digit.
    error |= keep & ~(uint32_t)is_digit;
    // Leading zero: '0' starting an octet whose successor is a digit lane.
    const uint32_t zero_bits = (uint32_t)_mm_cmpeq_epi8_mask(v, zero_digit);
    error |= zero_bits & ((dots << 1) | 1u) & (keep >> 1);
    // Octet > 255 is the only check that has to wait for the VNNI result.
    error |= (uint32_t)_mm_cmpgt_epu32_mask(res, _mm_set1_epi32(0xff));

    if (!error) [[likely]] {
        _mm_storeu_si32(ptr, _mm_shuffle_epi8(res, _mm_setr_epi8(0, 4, 8, 12, 0, 0,
                                                                 0, 0, 0, 0, 0, 0,
                                                                 0, 0, 0, 0)));
        return 1;
    }
    return parse_ipv4_ada(input, len, ptr);
}

namespace avx512vl_ipv4_k {
// Held in .rodata rather than built with mov+vpbroadcastd. When the parser is
// not inlined (as in the benchmark) GCC re-materialises every _mm_set1 through
// a general register -- two instructions each, seven times -- whereas a
// .rodata constant folds into the consuming instruction as a memory operand.
static const __m128i dot   = _mm_setr_epi8('.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.');
static const __m128i zero  = _mm_setr_epi8('0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0');
static const __m128i nine  = _mm_setr_epi8(9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9);
static const __m128i iota  = _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
static const __m128i k_rep = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
static const __m128i off   = _mm_setr_epi8(-4,-3,-2,-1,-4,-3,-2,-1,-4,-3,-2,-1,-4,-3,-2,-1);
static const __m128i ones  = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
static const __m128i gmin  = _mm_setr_epi8(1,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0);
// Lanes 4..15 hold 255, which an unsigned byte can never exceed, so the gap
// compare needs no 0x000F write-mask (and no mov+kmovw to build it).
static const __m128i gmax  = _mm_setr_epi8(2,2,2,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
static const __m128i wts   = _mm_setr_epi8(0,100,10,1,0,100,10,1,0,100,10,1,0,100,10,1);
static const __m128i ff    = _mm_setr_epi8(-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0);
static const __m128i pack  = _mm_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0);
// _notab5 reverses the digit order inside each octet group: lane 4i+j fetches
// q_i-(j+1), so the group's little-endian dword is ones | tens<<8 | hundreds<<16.
// That is the base-256 encoding of the zero-padded 3-digit string, which is
// monotone in the decimal value -- so "> 255" is one unsigned dword compare
// against 0x00020505, and it can be taken from `padded` instead of from the
// dot-product result.
static const __m128i offr  = _mm_setr_epi8(-1,-2,-3,-4,-1,-2,-3,-4,-1,-2,-3,-4,-1,-2,-3,-4);
static const __m128i wtsr  = _mm_setr_epi8(1,10,100,0,1,10,100,0,1,10,100,0,1,10,100,0);
static const __m128i lim   = _mm_setr_epi8(5,5,2,0,5,5,2,0,5,5,2,0,5,5,2,0);
}  // namespace avx512vl_ipv4_k

/**
 * Table-free AVX-512 IPv4 parse, fourth cut: _notab's algorithm unchanged, with
 * two codegen-level changes.
 *
 *  - Splat constants live in .rodata instead of being rebuilt through a general
 *    register on every call.
 *  - The gap compare uses a per-lane bound ({2,2,2,2,255,...}) instead of a
 *    0x000F write-mask, so lanes 4..15 can never trip and the constant k-mask
 *    (a mov plus a kmovw) disappears.
 */
static int parse_ipv4_avx512vl_notab4(const char *input, size_t len, uint32_t *ptr) {
    namespace K = avx512vl_ipv4_k;
    if (len > 15) [[unlikely]] { return parse_ipv4_ada(input, len, ptr); }

    const uint32_t len_mask = _bzhi_u32(0xFFFFFFFFu, (unsigned)len);
    const __m128i v = _mm_mask_loadu_epi8(K::dot, (__mmask16)len_mask, (const __m128i *)input);

    const __mmask16 delim = _mm_cmpeq_epi8_mask(v, K::dot);
    const uint32_t dots = (uint32_t)delim & len_mask;
    const uint32_t keep = len_mask & ~dots;

    const __m128i digits = _mm_sub_epi8(v, K::zero);
    const __mmask16 is_digit = _mm_cmple_epu8_mask(digits, K::nine);
    const __mmask16 hole = _kandn_mask16(is_digit, (__mmask16)keep);
    const __m128i v0 = _mm_maskz_mov_epi8(is_digit, digits);

    const __m128i c = _mm_maskz_compress_epi8(delim, K::iota);
    const __m128i qi = _mm_shuffle_epi8(c, K::k_rep);
    const __m128i prev = _mm_shuffle_epi8(_mm_alignr_epi8(c, K::ones, 15), K::k_rep);
    const __m128i idx = _mm_max_epi8(_mm_add_epi8(qi, K::off), prev);
    const __m128i padded = _mm_shuffle_epi8(v0, idx);

    const __m128i res = _mm_dpbusd_epi32(_mm_setzero_si128(), padded, K::wts);
    const __mmask8 over = _mm_cmpgt_epu32_mask(res, K::ff);

    // Octet lengths from marker gaps, with the lane restriction folded into the
    // bound rather than into a write-mask.
    const __m128i gap = _mm_sub_epi8(c, _mm_slli_si128(c, 1));
    const __mmask16 bad_gap =
        _mm_cmpgt_epu8_mask(_mm_sub_epi8(gap, K::gmin), K::gmax);

    const uint32_t zero_bits = (uint32_t)_mm_cmpeq_epi8_mask(v, K::zero);
    const uint32_t start_bits = (dots << 1) | 1u;

    int error = (_mm_popcnt_u32(dots) != 3);
    error |= (bad_gap != 0);
    error |= (_kor_mask16(hole, (__mmask16)over) != 0);
    error |= ((zero_bits & start_bits & (keep >> 1)) != 0);

    if (!error) [[likely]] {
        _mm_storeu_si32(ptr, _mm_shuffle_epi8(res, K::pack));
        return 1;
    }
    return parse_ipv4_ada(input, len, ptr);
}

/**
 * Table-free AVX-512 IPv4 parse, fifth cut: _notab4's algorithm with three
 * changes, all aimed at the branch rather than at the compress chain.
 *
 *  - The digit order inside each octet group is reversed, which makes the
 *    group's dword the base-256 encoding of the zero-padded digit string.
 *    That encoding is monotone in the decimal value, so `octet > 255` is a
 *    single unsigned compare against 0x00020505 taken straight off `padded` --
 *    in parallel with `vpdpbusd` instead of five cycles behind it. After this
 *    the dot product feeds nothing but the store.
 *  - The pack-and-store is one `vpmovdb` with a 4-lane write mask instead of
 *    `vpshufb` + `vmovd`, which also retires the `pack` constant.
 *  - The four error terms were four `setcc` plus three `or`. They now merge
 *    into one k-register tree and one general-purpose word, tested once.
 *
 * The junk check also moves fully into the mask domain (`~(delim | is_digit)`
 * inside the length), so the parse no longer needs `keep` before it can run it.
 *
 * Measured on a Xeon Gold 6548N (Emerald Rapids), gcc 14.3, random dotted
 * quads: 14.6 cycles/address versus 15.6 for _notab4.
 */
static int parse_ipv4_avx512vl_notab5(const char *input, size_t len, uint32_t *ptr) {
    namespace K = avx512vl_ipv4_k;
    if (len > 15) [[unlikely]] { return parse_ipv4_ada(input, len, ptr); }

    const uint32_t len_mask = _bzhi_u32(0xFFFFFFFFu, (unsigned)len);
    const __mmask16 len_k = (__mmask16)len_mask;
    const __m128i v = _mm_mask_loadu_epi8(K::dot, len_k, (const __m128i *)input);

    const __mmask16 delim = _mm_cmpeq_epi8_mask(v, K::dot);
    const uint32_t dots = (uint32_t)delim & len_mask;
    const uint32_t keep = len_mask & ~dots;

    const __m128i digits = _mm_sub_epi8(v, K::zero);
    const __mmask16 is_digit = _mm_cmple_epu8_mask(digits, K::nine);
    // Junk in a digit slot: inside [0,len) yet neither a digit nor a dot.
    const __mmask16 hole = _kandn_mask16(_kor_mask16(delim, is_digit), len_k);
    const __m128i v0 = _mm_maskz_mov_epi8(is_digit, digits);

    const __m128i c = _mm_maskz_compress_epi8(delim, K::iota);
    const __m128i qi = _mm_shuffle_epi8(c, K::k_rep);
    const __m128i prev = _mm_shuffle_epi8(_mm_alignr_epi8(c, K::ones, 15), K::k_rep);
    const __m128i idx = _mm_max_epi8(_mm_add_epi8(qi, K::offr), prev);
    const __m128i padded = _mm_shuffle_epi8(v0, idx);

    // Octet > 255, straight off `padded`: the reversed group order makes the
    // dword monotone in the value, and the pad byte is always the high one.
    const __mmask8 over = _mm_cmpgt_epu32_mask(padded, K::lim);
    const __m128i res = _mm_dpbusd_epi32(_mm_setzero_si128(), padded, K::wtsr);

    const __m128i gap = _mm_sub_epi8(c, _mm_slli_si128(c, 1));
    const __mmask16 bad_gap =
        _mm_cmpgt_epu8_mask(_mm_sub_epi8(gap, K::gmin), K::gmax);

    const uint32_t zero_bits = (uint32_t)_mm_cmpeq_epi8_mask(v, K::zero);
    const uint32_t start_bits = (dots << 1) | 1u;

    const __mmask16 kerr = _kor_mask16(_kor_mask16(hole, (__mmask16)over), bad_gap);
    const uint32_t gerr = (uint32_t)(_mm_popcnt_u32(dots) ^ 3u)
                        | (zero_bits & start_bits & (keep >> 1));

    if ((gerr | (uint32_t)kerr) == 0) [[likely]] {
        _mm_mask_cvtepi32_storeu_epi8(ptr, 0x0F, res);
        return 1;
    }
    return parse_ipv4_ada(input, len, ptr);
}

#endif
