// Vendored from https://github.com/vtlmks/ipv6-avx512 (parse_ipv6.h).
//
// AVX-512 IPv6 text-to-bytes parser by Peter Fors (vtlmks).
// Copyright (c) 2026 Peter Fors
// SPDX-License-Identifier: MIT
//
// The body below is reproduced verbatim from the upstream parse_ipv6.h, with
// the sole change of wrapping it in `namespace vtlmks` so it can coexist with
// the other parsers benchmarked in this project. All credit to Peter Fors.
#ifndef VTLMKS_IPV6_H
#define VTLMKS_IPV6_H

#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <x86intrin.h>

namespace vtlmks {

// iota for vpcompressb: gathers the byte offsets of set mask bits. Not const
// per house style; it is written once at load and only ever read.
static uint8_t iota64[64] = {
	0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
	48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
};

// vpermi2b hex translation: ASCII byte -> nibble value, invalid -> 0x80 (the
// sign bit lets one movepi8 flag every non-hex character at once). The two
// halves are the low/high 64 entries of the 128-entry table indexed by the
// low 7 bits of each byte: '0'-'9' -> 0-9, 'A'-'F'/'a'-'f' -> 10-15. This one
// op does translation and validation together, which is why it beats an
// arithmetic translate that needs separate range checks.
static uint8_t hex_lo[64] = {
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x80,0x80,0x80,0x80,0x80,0x80
};
static uint8_t hex_hi[64] = {
	0x80,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
	0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
};

// [=]===^=[ hex_value ]==================================================[=]
static int hex_value(uint8_t c) {
	if(c >= '0' && c <= '9') {
		return c - '0';
	}
	if(c >= 'a' && c <= 'f') {
		return c - 'a' + 10;
	}
	if(c >= 'A' && c <= 'F') {
		return c - 'A' + 10;
	}
	return -1;
}

// [=]===^=[ parse_ipv4_tail ]============================================[=]
// Parse a strict dotted-quad in [p,end) into 4 bytes. Leading zeros in an
// octet are rejected, matching inet_pton's embedded-IPv4 handling.
static int parse_ipv4_tail(uint8_t *p, uint8_t *end, uint8_t *out4) {
	uint32_t octet = 0;
	uint32_t val = 0;
	uint32_t digits = 0;
	while(p < end) {
		uint8_t ch = *p++;
		if(ch >= '0' && ch <= '9') {
			if(digits == 1 && val == 0) {
				return 0;
			}
			val = val * 10 + (uint32_t)(ch - '0');
			++digits;
			if(val > 255 || digits > 3) {
				return 0;
			}
			continue;
		}
		if(ch == '.') {
			if(digits == 0 || octet >= 3) {
				return 0;
			}
			out4[octet++] = (uint8_t)val;
			val = 0;
			digits = 0;
			continue;
		}
		return 0;
	}
	if(digits == 0 || octet != 3) {
		return 0;
	}
	out4[octet] = (uint8_t)val;
	return 1;
}

// [=]===^=[ parse_groups ]===============================================[=]
// Parse colon-separated hextets in [p,end) into out (cap bytes). The final
// token may be an embedded IPv4 dotted-quad when allow_v4 is set. Sets
// *produced to the number of bytes written. Empty range yields 0 bytes.
static int parse_groups(uint8_t *p, uint8_t *end, uint8_t *out, uint32_t cap, uint32_t *produced, int allow_v4) {
	uint32_t bytes = 0;
	if(p == end) {
		*produced = 0;
		return 1;
	}
	while(1) {
		uint8_t *q = p;
		int has_dot = 0;
		while(q < end && *q != ':') {
			if(*q == '.') {
				has_dot = 1;
			}
			++q;
		}
		if(p == q) {
			return 0;
		}
		if(has_dot) {
			if(!allow_v4 || bytes + 4 > cap) {
				return 0;
			}
			if(!parse_ipv4_tail(p, q, &out[bytes])) {
				return 0;
			}
			bytes += 4;
			if(q != end) {
				return 0;
			}
			break;
		}
		uint32_t ndig = 0;
		uint32_t val = 0;
		for(uint8_t *r = p; r < q; ++r) {
			int v = hex_value(*r);
			if(v < 0) {
				return 0;
			}
			val = (val << 4) | (uint32_t)v;
			if(++ndig > 4) {
				return 0;
			}
		}
		if(bytes + 2 > cap) {
			return 0;
		}
		out[bytes++] = (uint8_t)(val >> 8);
		out[bytes++] = (uint8_t)(val & 0xff);
		if(q == end) {
			break;
		}
		p = q + 1;
		if(p == end) {
			return 0;
		}
	}
	*produced = bytes;
	return 1;
}

// [=]===^=[ parse_ipv6_scalar ]==========================================[=]
static int parse_ipv6_scalar(const char *src, size_t len, uint8_t *out) {
	uint8_t *s = (uint8_t *)src;
	uint8_t *end = s + len;
	if(len == 0) {
		return 0;
	}

	uint8_t *dc = 0;
	for(uint8_t *p = s; p + 1 < end; ++p) {
		if(p[0] == ':' && p[1] == ':') {
			dc = p;
			break;
		}
	}

	uint8_t tmp[16];
	memset(tmp, 0, sizeof tmp);

	if(dc) {
		for(uint8_t *p = dc + 2; p + 1 < end; ++p) {
			if(p[0] == ':' && p[1] == ':') {
				return 0;
			}
		}
		uint8_t *head_end = dc;
		uint8_t *tail_beg = dc + 2;
		if(head_end > s && head_end[-1] == ':') {
			return 0;
		}
		if(tail_beg < end && tail_beg[0] == ':') {
			return 0;
		}
		uint8_t headbuf[16];
		uint8_t tailbuf[16];
		uint32_t hb = 0;
		uint32_t tb = 0;
		if(!parse_groups(s, head_end, headbuf, 16, &hb, 1)) {
			return 0;
		}
		if(!parse_groups(tail_beg, end, tailbuf, 16, &tb, 1)) {
			return 0;
		}
		if(hb + tb > 14) {
			return 0;
		}
		memcpy(tmp, headbuf, hb);
		memcpy(tmp + 16 - tb, tailbuf, tb);
	} else {
		if(s[0] == ':' || end[-1] == ':') {
			return 0;
		}
		uint32_t b = 0;
		if(!parse_groups(s, end, tmp, 16, &b, 1)) {
			return 0;
		}
		if(b != 16) {
			return 0;
		}
	}

	memcpy(out, tmp, 16);
	return 1;
}

// [=]===^=[ parse_ipv6_avx512 ]==========================================[=]
// Two fast SIMD paths share one gather tail: the clean 8-group form, and the
// "::"-compressed form. Both reduce to an 8-element (end, start) per group;
// the gather then reads each group's digits and packs them. Embedded IPv4 and
// anything malformed drop to the scalar reference, which matches inet_pton.
__attribute__((always_inline))
static inline int parse_ipv6_avx512(const char *src, size_t len, uint8_t *out) {
	if(len < 2 || len > 45) {
		return parse_ipv6_scalar(src, len, out);
	}

	__mmask64 active = _bzhi_u64(~0ull, (uint32_t)len);
	__m512i v = _mm512_maskz_loadu_epi8(active, src);

	__mmask64 mcolon = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(':'));
	__mmask64 dc     = mcolon & (mcolon >> 1);

	// One vpermi2b translates ASCII hex to nibble values; non-hex bytes (incl.
	// '.') land negative, so one movepi8 flags every invalid char and routes
	// IPv4-form addresses to the scalar reference through the normal bad gate.
	__m512i nib = _mm512_permutex2var_epi8(_mm512_loadu_si512(hex_lo), v, _mm512_loadu_si512(hex_hi));

	__m256i grp  = _mm256_setr_epi8(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7);
	__m256i kvec = _mm256_setr_epi8(0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3);

	if(!dc) {
		// Clean 8-group form. Fields map 1:1 to groups; a synthetic separator at
		// position len ends the last field. The "width in 1..4" test also rejects
		// stray "::"/leading/trailing colons (each forces a width-0 group).
		__m512i comp = _mm512_maskz_compress_epi8(mcolon | (1ull << len), _mm512_loadu_si512(iota64));
		__m128i ends8 = _mm512_castsi512_si128(comp);
		__m128i starts8 = _mm_mask_add_epi8(_mm_setzero_si128(), 0xfffe, _mm_bslli_si128(ends8, 1), _mm_set1_epi8(1));
		__m128i width = _mm_sub_epi8(ends8, starts8);

		__m256i endsb   = _mm256_permutexvar_epi8(grp, _mm256_castsi128_si256(ends8));
		__m256i startsb = _mm256_permutexvar_epi8(grp, _mm256_castsi128_si256(starts8));
		__m256i idx_src = _mm256_add_epi8(_mm256_sub_epi8(endsb, _mm256_set1_epi8(4)), kvec);
		__mmask32 keep = ~_mm256_cmplt_epi8_mask(idx_src, startsb);
		__m256i gathered = _mm512_castsi512_si256(_mm512_maskz_permutexvar_epi8((__mmask64)(uint32_t)keep, _mm512_castsi256_si512(idx_src), nib));
		_mm_storeu_si128((__m128i *)out, _mm256_cvtepi16_epi8(_mm256_maddubs_epi16(gathered, _mm256_set1_epi16(0x0110))));

		uint64_t bad = (uint64_t)(_mm_popcnt_u64(mcolon) != 7);
		bad |= _mm_cmpgt_epu8_mask(_mm_sub_epi8(width, _mm_set1_epi8(1)), _mm_set1_epi8(3)) & 0xff;
		bad |= _mm512_movepi8_mask(v) & active;
		bad |= _mm512_movepi8_mask(nib) & active & ~mcolon;
		if(bad) {
			return parse_ipv6_scalar(src, len, out);
		}
		return 1;
	}

	// "::" form: per-field end/start on the contiguous fields, drop the
	// zero-width gap field(s), then expand the surviving head and tail fields
	// into 8 slots with Z all-zero groups inserted at the gap.
	__m512i comp = _mm512_maskz_compress_epi8(mcolon | (1ull << len), _mm512_loadu_si512(iota64));
	__m128i comp128 = _mm512_castsi512_si128(comp);
	__m128i starts_full = _mm_mask_add_epi8(_mm_setzero_si128(), 0xfffe, _mm_bslli_si128(comp128, 1), _mm_set1_epi8(1));
	__m128i width_full = _mm_sub_epi8(comp128, starts_full);

	uint32_t p = (uint32_t)_tzcnt_u64(dc);
	uint32_t ncolon = (uint32_t)_mm_popcnt_u64(mcolon);
	uint32_t head = (p == 0) ? 0 : (uint32_t)_mm_popcnt_u64(mcolon & (((uint64_t)1 << p) - 1)) + 1;
	__mmask16 fields = (__mmask16)((1u << (ncolon + 1)) - 1);
	__mmask16 keepf = _mm_cmpgt_epi8_mask(width_full, _mm_setzero_si128()) & fields;
	uint32_t nkept = (uint32_t)_mm_popcnt_u32(keepf);
	uint32_t zeros = 8 - nkept;
	__mmask16 outm = (__mmask16)(~(((1u << zeros) - 1) << head) & 0xff);
	__m128i ends8 = _mm_maskz_expand_epi8(outm, _mm_maskz_compress_epi8(keepf, comp128));
	__m128i starts8 = _mm_maskz_expand_epi8(outm, _mm_maskz_compress_epi8(keepf, starts_full));

	__m256i endsb   = _mm256_permutexvar_epi8(grp, _mm256_castsi128_si256(ends8));
	__m256i startsb = _mm256_permutexvar_epi8(grp, _mm256_castsi128_si256(starts8));
	__m256i idx_src = _mm256_add_epi8(_mm256_sub_epi8(endsb, _mm256_set1_epi8(4)), kvec);
	__mmask32 keep = ~_mm256_cmplt_epi8_mask(idx_src, startsb);
	__m256i gathered = _mm512_castsi512_si256(_mm512_maskz_permutexvar_epi8((__mmask64)(uint32_t)keep, _mm512_castsi256_si512(idx_src), nib));
	_mm_storeu_si128((__m128i *)out, _mm256_cvtepi16_epi8(_mm256_maddubs_epi16(gathered, _mm256_set1_epi16(0x0110))));

	uint64_t bad = _mm512_movepi8_mask(v) & active;
	bad |= _mm512_movepi8_mask(nib) & active & ~mcolon;
	bad |= _blsr_u64(dc);                                          // more than one "::"
	bad |= (uint64_t)(nkept >= 8);                                 // "::" must compress >=1 group
	bad |= _mm_cmpgt_epu8_mask(width_full, _mm_set1_epi8(4)) & keepf;  // a kept group too long
	bad |= (mcolon & 1ull) & ~dc;                                 // leading colon not part of "::"
	bad |= ((mcolon >> (len - 1)) & 1ull) & ~(dc >> (len - 2));   // trailing colon not part of "::"
	if(bad) {
		return parse_ipv6_scalar(src, len, out);
	}
	return 1;
}

}  // namespace vtlmks

#endif
