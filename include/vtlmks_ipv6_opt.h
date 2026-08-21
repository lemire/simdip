#ifndef VTLMKS_IPV6_OPT_H
#define VTLMKS_IPV6_OPT_H
//
// An optimized derivative of vtlmks::parse_ipv6_avx512 (Peter Fors, MIT; see
// vtlmks_ipv6.h, which is kept verbatim). The algorithm is unchanged -- two
// SIMD arms over an 8-element (end, start) per group, sharing a gather tail --
// and the scalar reference is reused as-is for the fallback. What changed is
// how the work reaches the vector units.
//
// Three changes, measured independently on a Xeon Gold 6548N (Emerald Rapids),
// gcc 14.3, 200k random dotted IPv6 addresses (the clean 8-group arm):
//
//   1. Colon-filled tail. The masked load merges ':' instead of zero-filling,
//      so the synthetic terminator at `len` is a real colon and the compress
//      mask is `mcolon` itself. Upstream has to rebuild it in a general
//      register -- kmovq, bts, kmovq -- which puts about nine cycles of round
//      trip in front of `vpcompressb`, the longest op in the parse.
//      (+7.0%: 20.35 -> 18.93 cycles.)
//
//   2. Clamp instead of a write mask on the gather. Pad lanes are pointed at
//      the colon that precedes their group rather than masked off, so the
//      gather is a plain `vpermb`. This needs ':' to translate to 0x00 instead
//      of 0x80, which is invisible to validation because the non-hex check
//      already excludes colons (`& ~mcolon`); group 0's pads clamp to -1, i.e.
//      byte 63, which the colon fill also makes a zero. That retires an ymm
//      `vpcmpb`, a `knot`, the 0xfffe write mask, two splats, and -- the real
//      prize -- a `kmovd`/`kmovq` pair gcc emits to widen __mmask32 to
//      __mmask64 directly on the path to the gather.
//      (+6.8% more: 18.93 -> 17.53 cycles, 72 -> 59 instructions.)
//
//   3. One sign-bit scan and one error accumulator. `vpternlog` folds the
//      high-bit check into the nibble vector (nib | (v & 0x80)), and the
//      remaining flags stay in k-registers with a per-lane width bound instead
//      of a general-register `& 0xff`.
//      (+2.4% more: 17.53 -> 17.08 cycles.)
//
//   4. On the "::" arm, `fields` -- and with it a colon popcount, a shift and a
//      k -> general-register move -- is deleted from the chain that gates the
//      expand. It exists upstream because a zero-filled tail leaves garbage
//      widths in the lanes past the last marker; the colon fill makes those
//      lanes contiguous tail-colon positions whose width is exactly 0, so the
//      width test already drops them. Plus a branchless head and _bzhi, which
//      also removes a shift-count UB when nkept > 8.
//      ("::" arm 36.8 -> 29.5 cycles, IPC 2.81 -> 3.23.)
//
// Net: clean arm 20.34 -> 17.04 cycles, 72 -> 57 instructions (+16.2%);
// "::" arm 40.33 -> 29.45 cycles, 108 -> 95 instructions (+27.1%).
//
// Verified against the upstream parser on 2.1M inputs (structured forms,
// random addresses biased toward zero runs, and random junk): no mismatches.
//
#include <stdint.h>
#include <immintrin.h>
#include <x86intrin.h>

#include "vtlmks_ipv6.h"
#include "avx512ip.h"   // parse_ipv4_avx512vl_strict, for embedded IPv4

namespace vtlmks_opt {

namespace k {

// Upstream's hex translate, except ':' -> 0x00 instead of 0x80. Colons are
// excluded from the non-hex check by `& ~mcolon`, so validation cannot see the
// difference -- and it lets a clamped pad index read a zero nibble off the
// colon that ends the previous group.
alignas(64) static const uint8_t hex_lo[64] = {
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x00,0x80,0x80,0x80,0x80,0x80
};
alignas(64) static const uint8_t hex_hi[64] = {
	0x80,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
};
alignas(64) static const uint8_t iota64[64] = {
	0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
	48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
};

// byte i of the 8-element vector -> lanes 4i..4i+3
static const __m256i grp = _mm256_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,
                                            4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
// "- 4 + {0,1,2,3}" folded into one addend
static const __m256i off = _mm256_setr_epi8(-4,-3,-2,-1,-4,-3,-2,-1,-4,-3,-2,-1,-4,-3,-2,-1,
                                            -4,-3,-2,-1,-4,-3,-2,-1,-4,-3,-2,-1,-4,-3,-2,-1);
static const __m256i w0110  = _mm256_set1_epi16(0x0110);
// group 0 has no preceding colon; -1 sends its pads to byte 63, a tail colon
static const __m128i lane0ff = _mm_setr_epi8(-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
// per-lane width bound: lanes 8..15 compare against 255, which an unsigned byte
// can never exceed, so no k-constant and no general-register "& 0xff"
static const __m128i wmax8 = _mm_setr_epi8(3,3,3,3,3,3,3,3,-1,-1,-1,-1,-1,-1,-1,-1);
static const __m128i two   = _mm_set1_epi8(2);
static const __m128i one   = _mm_set1_epi8(1);
static const __m128i five  = _mm_set1_epi8(5);
static const __m128i minus1 = _mm_set1_epi8(-1);
static const __m512i hi_bit = _mm512_set1_epi8((char)0x80);

}  // namespace k

// Gather each group's up-to-four nibbles into [n0 n1 n2 n3] and pack the eight
// hextets. `prev` is the byte before each group -- the colon ending the
// previous group, or -1 for a group at offset 0 -- so a pad index clamps onto
// a zero nibble instead of needing a write mask.
__attribute__((always_inline))
static inline void gather_store(__m128i ends8, __m128i prev, __m512i nib, uint8_t *out) {
	__m256i endsb = _mm256_permutexvar_epi8(k::grp, _mm256_zextsi128_si256(ends8));
	__m256i prevb = _mm256_permutexvar_epi8(k::grp, _mm256_zextsi128_si256(prev));
	__m256i idx = _mm256_max_epi8(_mm256_add_epi8(endsb, k::off), prevb);
	__m256i gathered = _mm512_castsi512_si256(
		_mm512_permutexvar_epi8(_mm512_zextsi256_si512(idx), nib));
	_mm_storeu_si128((__m128i *)out,
	                 _mm256_cvtepi16_epi8(_mm256_maddubs_epi16(gathered, k::w0110)));
}

// The two SIMD arms, with no fallback of their own: 0 means "this parser
// declines", not "invalid". Callers decide what to do next. `out` may be
// written speculatively even when 0 is returned.
// Requires 2 <= len <= 45.
__attribute__((always_inline))
static inline int parse_ipv6_core(const char *src, size_t len, uint8_t *out) {
	const __mmask64 active = (__mmask64)_bzhi_u64(~0ull, (uint32_t)len);
	const __m512i colon = _mm512_set1_epi8(':');
	// Colon-filled tail: bytes at and past `len` read as ':', which makes the
	// terminator at `len` a real colon and keeps the compress mask in a
	// k-register. Tail colons land in compressed lanes 8+, which neither arm
	// reads.
	const __m512i v = _mm512_mask_loadu_epi8(colon, active, src);
	const __mmask64 mcolon = _mm512_cmpeq_epi8_mask(v, colon);

	const __m512i nib = _mm512_permutex2var_epi8(_mm512_load_si512(k::hex_lo), v,
	                                             _mm512_load_si512(k::hex_hi));
	// Compress issues straight off the colon compare now; both arms want it.
	const __m512i comp = _mm512_maskz_compress_epi8(mcolon, _mm512_load_si512(k::iota64));
	const __m128i comp128 = _mm512_castsi512_si128(comp);

	const uint64_t mc = (uint64_t)mcolon;
	const uint64_t act = (uint64_t)active;
	// A real "::" needs both colons inside [0,len).
	const uint64_t dc = mc & (mc >> 1) & (act >> 1);

	if(!dc) {
		// Clean 8-group form. Fields map 1:1 to groups.
		const __m128i ends8 = comp128;
		const __m128i prev = _mm_or_si128(_mm_bslli_si128(ends8, 1), k::lane0ff);
		const __m128i w1 = _mm_sub_epi8(ends8, prev);   // width + 1, so 2..5 is legal
		gather_store(ends8, prev, nib, out);

		// nib | (v & 0x80): one sign scan covers non-hex characters and bytes
		// whose high bit made vpermi2b alias a legal one.
		const __m512i fused = _mm512_ternarylogic_epi32(nib, v, k::hi_bit, 0xf8);
		__mmask64 kerr = _kandn_mask64(mcolon, _mm512_movepi8_mask(fused));
		kerr = _kor_mask64(kerr, (__mmask64)(uint16_t)_mm_cmpgt_epu8_mask(
			_mm_sub_epi8(w1, k::two), k::wmax8));
		uint64_t bad = (uint64_t)(_mm_popcnt_u64(mc & act) != 7) | (uint64_t)kerr;
		return bad ? 0 : 1;
	}

	// "::" form: per-field end/start on the contiguous fields, drop the
	// zero-width gap field(s), then expand the surviving head and tail fields
	// into 8 slots with all-zero groups inserted at the gap.
	//
	// `prev` is the same vector the clean arm builds, so upstream's masked add
	// -- with its 0xfffe write mask and its splat -- is not needed; the widths
	// come out one too large, which only shifts the two bounds below.
	const __m128i prev_full = _mm_or_si128(_mm_bslli_si128(comp128, 1), k::lane0ff);
	const __m128i w1_full = _mm_sub_epi8(comp128, prev_full);   // width + 1

	// Upstream masks the width test with `fields` (hence a popcount of the
	// colons and a shift) because with a zero-filled tail the compressed lanes
	// past the last real marker are 0 and their widths are garbage. The
	// colon-filled tail makes those lanes *contiguous* tail-colon positions
	// instead, so every field past the last real one has width exactly 0 and
	// the width test drops it already. comp128's sixteen lanes are always fully
	// populated: len is at most 45, so the tail contributes at least 19 colons.
	// Deleting `fields` takes a k -> general-register move, a popcount and a
	// shift out of the chain that gates the expand, and is worth 6 cycles.
	const uint32_t head = (uint32_t)_mm_popcnt_u64(mc & (dc - 1)) + (uint32_t)(dc > 1);
	const __mmask16 keepf = _mm_cmpgt_epi8_mask(w1_full, k::one);
	const uint32_t nkept = (uint32_t)_mm_popcnt_u32(keepf);
	const uint32_t zeros = 8 - nkept;
	const __mmask16 outm = (__mmask16)(~(_bzhi_u32(0xffffffffu, zeros) << head) & 0xff);
	const __m128i ends8 = _mm_maskz_expand_epi8(outm, _mm_maskz_compress_epi8(keepf, comp128));
	// Inserted all-zero groups need prev = -1 so their lanes clamp to byte 63,
	// a tail colon, and read zero. A merge-masked expand supplies it for free.
	const __m128i prev8 = _mm_mask_expand_epi8(k::minus1, outm,
	                                           _mm_maskz_compress_epi8(keepf, prev_full));
	gather_store(ends8, prev8, nib, out);

	uint64_t bad = _mm512_movepi8_mask(v);                        // high-bit bytes
	bad |= _mm512_movepi8_mask(nib) & ~mc;                        // non-hex
	bad |= _blsr_u64(dc);                                         // more than one "::"
	bad |= (uint64_t)(nkept >= 8);                                // "::" must compress >=1 group
	bad |= _mm_cmpgt_epu8_mask(w1_full, k::five) & keepf;         // a kept group too long
	bad |= (mc & 1ull) & ~dc;                                     // leading colon not part of "::"
	bad |= ((mc >> (len - 1)) & 1ull) & ~(dc >> (len - 2));       // trailing colon not part of "::"
	return bad ? 0 : 1;
}

// An IPv6 address may write its last 32 bits as a dotted quad (RFC 4291), and
// the SIMD arms reject the form on purpose: '.' translates to 0x80, so it trips
// the non-hex check. Upstream then re-walks the entire string byte at a time,
// which costs 203 cycles -- ten times the clean arm -- and on a corpus shaped
// like real traffic that is a quarter of total time for a twentieth of the
// input.
//
// "X:d.d.d.d" denotes exactly the address "X:AABB:CCDD". Rather than teach the
// group machinery a second element size, this parses the quad with the strict
// IPv4 fast path, rewrites those characters as two hextets, and runs the same
// core over the result: correct by construction, and it cannot recurse because
// the rewritten string has no dots left.
//
// Kept out of line so the common path never sees any of it.
__attribute__((noinline))
static int parse_ipv6_embedded_v4(const char *src, size_t len, uint8_t *out) {
	const __mmask64 active = (__mmask64)_bzhi_u64(~0ull, (uint32_t)len);
	const __m512i v = _mm512_maskz_loadu_epi8(active, src);
	const uint64_t act = (uint64_t)active;
	const uint64_t dots = (uint64_t)_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8('.')) & act;
	const uint64_t colons = (uint64_t)_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(':')) & act;
	if(dots == 0 || colons == 0) {
		return vtlmks::parse_ipv6_scalar(src, len, out);
	}
	// The quad must be the final field: every dot after the last colon.
	const uint32_t last_colon = 63u - (uint32_t)_lzcnt_u64(colons);
	if((uint32_t)_tzcnt_u64(dots) <= last_colon) {
		return vtlmks::parse_ipv6_scalar(src, len, out);
	}
	const size_t off = (size_t)last_colon + 1;
	const size_t n4 = len - off;
	uint32_t quad = 0;
	if(n4 < 7 || n4 > 15 || !parse_ipv4_avx512vl_strict(src + off, n4, &quad)) {
		return vtlmks::parse_ipv6_scalar(src, len, out);
	}

	// "d.d.d.d" (7..15 chars) becomes "AABB:CCDD" (9). A valid hextet prefix is
	// at most six groups plus their colons, so the rewrite fits; anything longer
	// is invalid anyway and goes to the reference.
	const size_t nlen = off + 9;
	if(nlen > 45) {
		return vtlmks::parse_ipv6_scalar(src, len, out);
	}
	alignas(64) char buf[64];
	_mm512_mask_storeu_epi8(buf, (__mmask64)_bzhi_u64(~0ull, (uint32_t)off), v);
	static const char hexd[16] = {'0','1','2','3','4','5','6','7',
	                              '8','9','a','b','c','d','e','f'};
	uint8_t q[4];
	__builtin_memcpy(q, &quad, 4);
	char *o = buf + off;
	o[0] = hexd[q[0] >> 4]; o[1] = hexd[q[0] & 15];
	o[2] = hexd[q[1] >> 4]; o[3] = hexd[q[1] & 15];
	o[4] = ':';
	o[5] = hexd[q[2] >> 4]; o[6] = hexd[q[2] & 15];
	o[7] = hexd[q[3] >> 4]; o[8] = hexd[q[3] & 15];

	if(parse_ipv6_core(buf, nlen, out)) {
		return 1;
	}
	return vtlmks::parse_ipv6_scalar(src, len, out);
}

__attribute__((always_inline))
static inline int parse_ipv6_avx512_opt(const char *src, size_t len, uint8_t *out) {
	if(len < 2 || len > 45) {
		return vtlmks::parse_ipv6_scalar(src, len, out);
	}
	if(parse_ipv6_core(src, len, out)) {
		return 1;
	}
	return parse_ipv6_embedded_v4(src, len, out);
}

}  // namespace vtlmks_opt

#endif
