#ifndef ADA_IP_H
#define ADA_IP_H

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <system_error>
#include <utility>

/**
 * Scalar fallback parsers ported from the ada URL library (WHATWG-conformant).
 * Source: https://github.com/ada-url/ada/blob/791fb5c0a570b69b598d71b98827ff9f18a6a6c3/src/url.cpp#L28
 *
 * Unlike the strict SIMD path, these accept the unusual-but-valid forms allowed
 * by the URL standard: octal (leading-zero) and hexadecimal (0x-prefixed)
 * segments, and IPv4 addresses with fewer than four parts (e.g. "192.168.1").
 */

static inline bool ada_has_hex_prefix(std::string_view input) {
    return input.size() >= 2 && input[0] == '0' &&
           (input[1] == 'x' || input[1] == 'X');
}

static inline bool ada_is_digit(char c) { return c >= '0' && c <= '9'; }

static inline bool ada_is_ascii_hex_digit(char c) {
    return ada_is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static inline uint8_t ada_convert_hex_to_binary(char c) {
    if (ada_is_digit(c)) { return uint8_t(c - '0'); }
    return uint8_t((c | 0x20) - 'a' + 10);
}

/**
 * Parses an IPv4 address using the permissive ada/WHATWG algorithm.
 * The result is written as four bytes in network order (first octet first),
 * matching the layout produced by parse_ipv4_avx512vl.
 *
 * @return 1 on success, 0 if the format is invalid.
 */
static int parse_ipv4_ada(const char *input_ptr, size_t input_len, uint32_t *ptr) {
    std::string_view input(input_ptr, input_len);
    if (input.empty()) { return 0; }
    if (input.back() == '.') { input.remove_suffix(1); }
    size_t digit_count = 0;
    uint64_t ipv4 = 0;
    for (; (digit_count < 4) && !(input.empty()); digit_count++) {
        uint32_t segment_result = 0;  // any value exceeding 32 bits is an error
        bool is_hex = ada_has_hex_prefix(input);
        if (is_hex && ((input.length() == 2) ||
                       ((input.length() > 2) && (input[2] == '.')))) {
            // special case: a bare "0x" segment is zero
            segment_result = 0;
            input.remove_prefix(2);
        } else {
            std::from_chars_result r{};
            if (is_hex) {
                r = std::from_chars(input.data() + 2, input.data() + input.size(),
                                    segment_result, 16);
            } else if ((input.length() >= 2) && input[0] == '0' &&
                       ada_is_digit(input[1])) {
                r = std::from_chars(input.data() + 1, input.data() + input.size(),
                                    segment_result, 8);
            } else {
                r = std::from_chars(input.data(), input.data() + input.size(),
                                    segment_result, 10);
            }
            if (r.ec != std::errc()) { return 0; }
            input.remove_prefix(r.ptr - input.data());
        }
        if (input.empty()) {
            // last value: at this stage ipv4 holds digit_count*8 bits, so the
            // final segment must fit in the remaining 32-digit_count*8 bits.
            if (segment_result >= (uint64_t(1) << (32 - digit_count * 8))) {
                return 0;
            }
            ipv4 <<= (32 - digit_count * 8);
            ipv4 |= segment_result;
            goto final;
        } else {
            // more to come: the value must be <= 255 and followed by a '.'
            if ((segment_result > 255) || (input[0] != '.')) { return 0; }
            ipv4 <<= 8;
            ipv4 |= segment_result;
            input.remove_prefix(1);  // remove '.'
        }
    }
    if ((digit_count != 4) || (!input.empty())) { return 0; }
final:
    uint8_t *out = (uint8_t *)ptr;
    out[0] = uint8_t(ipv4 >> 24);
    out[1] = uint8_t(ipv4 >> 16);
    out[2] = uint8_t(ipv4 >> 8);
    out[3] = uint8_t(ipv4);
    return 1;
}

/**
 * Parses an IPv6 address using the ada/WHATWG algorithm.
 * The validated address is written as 16 bytes (network order) to ptr.
 *
 * @return 1 on success, 0 if the format is invalid.
 */
static int parse_ipv6_ada(const char *input_ptr, size_t input_len, uint8_t *ptr) {
    std::string_view input(input_ptr, input_len);
    if (input.empty()) { return 0; }
    uint16_t address[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int piece_index = 0;
    int compress = -1;  // -1 stands for "null"
    const char *pointer = input.data();
    const char *end = input.data() + input.size();

    if (input[0] == ':') {
        if (input.size() == 1 || input[1] != ':') { return 0; }
        pointer += 2;
        compress = ++piece_index;
    }

    while (pointer != end) {
        if (piece_index == 8) { return 0; }
        if (*pointer == ':') {
            if (compress != -1) { return 0; }
            pointer++;
            compress = ++piece_index;
            continue;
        }

        uint16_t value = 0, length = 0;
        while (length < 4 && pointer != end && ada_is_ascii_hex_digit(*pointer)) {
            value = uint16_t(value * 0x10 + ada_convert_hex_to_binary(*pointer));
            pointer++;
            length++;
        }

        if (pointer != end && *pointer == '.') {
            if (length == 0) { return 0; }
            pointer -= length;
            if (piece_index > 6) { return 0; }
            int numbers_seen = 0;
            while (pointer != end) {
                int ipv4_piece = -1;  // -1 stands for "null"
                if (numbers_seen > 0) {
                    if (*pointer == '.' && numbers_seen < 4) {
                        pointer++;
                    } else {
                        return 0;
                    }
                }
                if (pointer == end || !ada_is_digit(*pointer)) { return 0; }
                while (pointer != end && ada_is_digit(*pointer)) {
                    int number = *pointer - '0';
                    if (ipv4_piece == -1) {
                        ipv4_piece = number;
                    } else if (ipv4_piece == 0) {
                        return 0;
                    } else {
                        ipv4_piece = ipv4_piece * 10 + number;
                    }
                    if (ipv4_piece > 255) { return 0; }
                    pointer++;
                }
                address[piece_index] =
                    uint16_t(address[piece_index] * 0x100 + ipv4_piece);
                numbers_seen++;
                if (numbers_seen == 2 || numbers_seen == 4) { piece_index++; }
            }
            if (numbers_seen != 4) { return 0; }
            break;
        } else if ((pointer != end) && (*pointer == ':')) {
            pointer++;
            if (pointer == end) { return 0; }
        } else if (pointer != end) {
            return 0;
        }

        address[piece_index] = value;
        piece_index++;
    }

    if (compress != -1) {
        int swaps = piece_index - compress;
        piece_index = 7;
        while (piece_index != 0 && swaps > 0) {
            std::swap(address[piece_index], address[compress + swaps - 1]);
            piece_index--;
            swaps--;
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

#endif  // ADA_IP_H
