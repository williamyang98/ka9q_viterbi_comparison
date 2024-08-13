#pragma once

#include <immintrin.h>
#include <stdint.h>
#include <stdalign.h>
#include <stddef.h>

// simd=256bit, branch=8bit
template <typename poly_t>
static inline __m256i mm256_get_branch_value_s8(size_t curr_state_offset, poly_t G, int8_t branch_low, int8_t branch_high) {
    const __m256i v_decision_low = _mm256_set1_epi8(branch_low);
    const __m256i v_decision_high = _mm256_set1_epi8(branch_high);

    __m256i v_p32[4];
    __m256i v_p16[2];
    __m256i v_p8;
   
    // setup initial values
    if constexpr(sizeof(poly_t) == 4) {
        // 32bit
        alignas(32) const int32_t state_offset[8] = { 0, 4, 8, 12, 16, 20, 24, 28 };
        const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
        const __m256i v_G = _mm256_set1_epi32(int(G) >> 1);
        for (size_t i = 0; i < 4; i++) {
            const __m256i v_state = _mm256_add_epi32(v_state_offset, _mm256_set1_epi32(curr_state_offset + i));
            const __m256i v_reg = _mm256_and_si256(v_state, v_G);
            v_p32[i] = v_reg;
        }
    } else if constexpr(sizeof(poly_t) == 2) {
        // 16bit
        alignas(32) const int16_t state_offset[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 };
        const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
        const __m256i v_G = _mm256_set1_epi16(short(G) >> 1);
        for (size_t i = 0; i < 2; i++) {
            const __m256i v_state = _mm256_add_epi16(v_state_offset, _mm256_set1_epi16(curr_state_offset + i));
            const __m256i v_reg = _mm256_and_si256(v_state, v_G);
            v_p16[i] = v_reg;
        }
    } else if constexpr(sizeof(poly_t) == 1) {
        // 8bit
        alignas(32) const int8_t state_offset[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
        const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
        const __m256i v_state = _mm256_add_epi8(v_state_offset, _mm256_set1_epi8(curr_state_offset));
        const __m256i v_G = _mm256_set1_epi8(char(G) >> 1);
        const __m256i v_reg = _mm256_and_si256(v_state, v_G);
        v_p8 = v_reg;
    }

    if constexpr(sizeof(poly_t) >= 4) {
        // 32bit to 16bit
        v_p32[0] = _mm256_xor_si256(v_p32[0], _mm256_srli_epi32(v_p32[0], 16));
        v_p32[1] = _mm256_xor_si256(v_p32[1], _mm256_srli_epi32(v_p32[1], 16));
        v_p32[2] = _mm256_xor_si256(v_p32[2], _mm256_slli_epi32(v_p32[2], 16));
        v_p32[3] = _mm256_xor_si256(v_p32[3], _mm256_slli_epi32(v_p32[3], 16));
        // zip 16bit
        const uint8_t blend_u32 = 0b1010'1010;
        v_p16[0] = _mm256_blend_epi16(v_p32[0], v_p32[2], blend_u32);
        v_p16[1] = _mm256_blend_epi16(v_p32[1], v_p32[3], blend_u32);
    }
    if constexpr(sizeof(poly_t) >= 2) {
        // 16bit to 8bit
        v_p16[0] = _mm256_xor_si256(v_p16[0], _mm256_srli_epi16(v_p16[0], 8));
        v_p16[1] = _mm256_xor_si256(v_p16[1], _mm256_slli_epi16(v_p16[1], 8));
        // zip 8bit
        alignas(32) const uint8_t blend_u16[32] = {
            0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        };
        const __m256i v_blend_u16 = _mm256_load_si256(reinterpret_cast<const __m256i*>(blend_u16));
        v_p8 = _mm256_blendv_epi8(v_p16[0], v_p16[1], v_blend_u16);
    }
    if constexpr(sizeof(poly_t) >= 1) {
        // 8bit to 1bit
        v_p8 = _mm256_xor_si256(v_p8, _mm256_slli_epi64(v_p8, 4));
        v_p8 = _mm256_xor_si256(v_p8, _mm256_slli_epi64(v_p8, 2));
        v_p8 = _mm256_xor_si256(v_p8, _mm256_slli_epi64(v_p8, 1));
    }
    const __m256i branch_value = _mm256_blendv_epi8(v_decision_low, v_decision_high, v_p8);
    return branch_value;
}

// simd=128bit, branch=8bit
template <typename poly_t>
static inline __m128i mm128_get_branch_value_s8(size_t curr_state_offset, poly_t G, int8_t branch_low, int8_t branch_high) {
    const __m128i v_decision_low = _mm_set1_epi8(branch_low);
    const __m128i v_decision_high = _mm_set1_epi8(branch_high);

    __m128i v_p32[4];
    __m128i v_p16[2];
    __m128i v_p8;

    // setup initial values
    if constexpr(sizeof(poly_t) == 4) {
        // 32bit
        alignas(16) const int32_t state_offset[4] = { 0, 4, 8, 12 };
        const __m128i v_state_offset = _mm_load_si128(reinterpret_cast<const __m128i*>(state_offset));
        const __m128i v_G = _mm_set1_epi32(int(G) >> 1);
        for (size_t i = 0; i < 4; i++) {
            const __m128i v_state = _mm_add_epi32(v_state_offset, _mm_set1_epi32(curr_state_offset + i));
            const __m128i v_reg = _mm_and_si128(v_state, v_G);
            v_p32[i] = v_reg;
        }
    } else if constexpr(sizeof(poly_t) == 2) {
        // 16bit
        alignas(16) const int16_t state_offset[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };
        const __m128i v_state_offset = _mm_load_si128(reinterpret_cast<const __m128i*>(state_offset));
        const __m128i v_G = _mm_set1_epi16(short(G) >> 1);
        for (size_t i = 0; i < 2; i++) {
            const __m128i v_state = _mm_add_epi16(v_state_offset, _mm_set1_epi16(curr_state_offset + i));
            const __m128i v_reg = _mm_and_si128(v_state, v_G);
            v_p16[i] = v_reg;
        }
    } else if constexpr(sizeof(poly_t) == 1) {
        // 8bit
        alignas(16) const int8_t state_offset[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        const __m128i v_state_offset = _mm_load_si128(reinterpret_cast<const __m128i*>(state_offset));
        const __m128i v_state = _mm_add_epi8(v_state_offset, _mm_set1_epi8(curr_state_offset));
        const __m128i v_G = _mm_set1_epi8(char(G) >> 1);
        const __m128i v_reg = _mm_and_si128(v_state, v_G);
        v_p8 = v_reg;
    }

    if constexpr(sizeof(poly_t) >= 4) {
        // 32bit to 16bit
        v_p32[0] = _mm_xor_si128(v_p32[0], _mm_srli_epi32(v_p32[0], 16));
        v_p32[1] = _mm_xor_si128(v_p32[1], _mm_srli_epi32(v_p32[1], 16));
        v_p32[2] = _mm_xor_si128(v_p32[2], _mm_slli_epi32(v_p32[2], 16));
        v_p32[3] = _mm_xor_si128(v_p32[3], _mm_slli_epi32(v_p32[3], 16));
        // zip 16bit
        const uint8_t blend_u32 = 0b1010'1010;
        v_p16[0] = _mm_blend_epi16(v_p32[0], v_p32[2], blend_u32);
        v_p16[1] = _mm_blend_epi16(v_p32[1], v_p32[3], blend_u32);
    }
    if constexpr(sizeof(poly_t) >= 2) {
        // 16bit to 8bit
        v_p16[0] = _mm_xor_si128(v_p16[0], _mm_srli_epi16(v_p16[0], 8));
        v_p16[1] = _mm_xor_si128(v_p16[1], _mm_slli_epi16(v_p16[1], 8));
        // zip 8bit
        alignas(32) const uint8_t blend_u16[32] = {
            0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        };
        const __m128i v_blend_u16 = _mm_load_si128(reinterpret_cast<const __m128i*>(blend_u16));
        v_p8 = _mm_blendv_epi8(v_p16[0], v_p16[1], v_blend_u16);
    }
    if constexpr(sizeof(poly_t) >= 1) {
        // 8bit to 1bit
        v_p8 = _mm_xor_si128(v_p8, _mm_slli_epi64(v_p8, 4));
        v_p8 = _mm_xor_si128(v_p8, _mm_slli_epi64(v_p8, 2));
        v_p8 = _mm_xor_si128(v_p8, _mm_slli_epi64(v_p8, 1));
    }
    const __m128i branch_value = _mm_blendv_epi8(v_decision_low, v_decision_high, v_p8);
    return branch_value;
}

// simd=256bit, branch=16bit
template <typename poly_t>
static inline __m256i mm256_get_branch_value_s16(size_t curr_state_offset, poly_t G, int16_t branch_low, int16_t branch_high) {
    const __m256i v_decision_low = _mm256_set1_epi16(branch_low);
    const __m256i v_decision_high = _mm256_set1_epi16(branch_high);

    __m256i v_p32[2];
    __m256i v_p16;
   
    // setup initial values
    if constexpr(sizeof(poly_t) == 4) {
        // 32bit
        alignas(32) const int32_t state_offset[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };
        const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
        const __m256i v_G = _mm256_set1_epi32(int(G) >> 1);
        for (size_t i = 0; i < 4; i++) {
            const __m256i v_state = _mm256_add_epi32(v_state_offset, _mm256_set1_epi32(curr_state_offset + i));
            const __m256i v_reg = _mm256_and_si256(v_state, v_G);
            v_p32[i] = v_reg;
        }
    } else if constexpr(sizeof(poly_t) == 2 || sizeof(poly_t) == 1) {
        // 16bit and 8bit
        alignas(32) const int16_t state_offset[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
        const __m256i v_G = _mm256_set1_epi16(short(G) >> 1);
        const __m256i v_state = _mm256_add_epi16(v_state_offset, _mm256_set1_epi16(curr_state_offset));
        const __m256i v_reg = _mm256_and_si256(v_state, v_G);
        v_p16 = v_reg;
    }

    if constexpr(sizeof(poly_t) >= 4) {
        // 32bit to 16bit
        v_p32[0] = _mm256_xor_si256(v_p32[0], _mm256_srli_epi32(v_p32[0], 16));
        v_p32[1] = _mm256_xor_si256(v_p32[1], _mm256_slli_epi32(v_p32[1], 16));
        // zip 16bit
        const uint8_t blend_u32 = 0b1010'1010;
        v_p16 = _mm256_blend_epi16(v_p32[0], v_p32[1], blend_u32);
    }
    if constexpr(sizeof(poly_t) >= 1) {
        // 16bit to 1bit
        const __m256i parity_mask = _mm256_set1_epi16(0x8000);
        v_p16 = _mm256_xor_si256(v_p16, _mm256_slli_epi64(v_p16, 8));
        v_p16 = _mm256_xor_si256(v_p16, _mm256_slli_epi64(v_p16, 4));
        v_p16 = _mm256_xor_si256(v_p16, _mm256_slli_epi64(v_p16, 2));
        v_p16 = _mm256_xor_si256(v_p16, _mm256_slli_epi64(v_p16, 1));
        v_p16 = _mm256_cmpeq_epi16(_mm256_and_si256(v_p16, parity_mask), parity_mask);
    }
    const __m256i branch_value = _mm256_blendv_epi8(v_decision_low, v_decision_high, v_p16);
    return branch_value;
}

// simd=128bit, branch=16bit
template <typename poly_t>
static inline __m128i mm128_get_branch_value_s16(size_t curr_state_offset, poly_t G, int16_t branch_low, int16_t branch_high) {
    const __m128i v_decision_low = _mm_set1_epi16(branch_low);
    const __m128i v_decision_high = _mm_set1_epi16(branch_high);

    __m128i v_p32[2];
    __m128i v_p16;
   
    // setup initial values
    if constexpr(sizeof(poly_t) == 4) {
        // 32bit
        alignas(32) const int32_t state_offset[4] = { 0, 2, 4, 6 };
        const __m128i v_state_offset = _mm_load_si128(reinterpret_cast<const __m128i*>(state_offset));
        const __m128i v_G = _mm_set1_epi32(int(G) >> 1);
        for (size_t i = 0; i < 4; i++) {
            const __m128i v_state = _mm_add_epi32(v_state_offset, _mm_set1_epi32(curr_state_offset + i));
            const __m128i v_reg = _mm_and_si128(v_state, v_G);
            v_p32[i] = v_reg;
        }
    } else if constexpr(sizeof(poly_t) == 2 || sizeof(poly_t) == 1) {
        // 16bit and 8bit
        alignas(32) const int16_t state_offset[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        const __m128i v_state_offset = _mm_load_si128(reinterpret_cast<const __m128i*>(state_offset));
        const __m128i v_G = _mm_set1_epi16(short(G) >> 1);
        const __m128i v_state = _mm_add_epi16(v_state_offset, _mm_set1_epi16(curr_state_offset));
        const __m128i v_reg = _mm_and_si128(v_state, v_G);
        v_p16 = v_reg;
    }

    if constexpr(sizeof(poly_t) >= 4) {
        // 32bit to 16bit
        v_p32[0] = _mm_xor_si128(v_p32[0], _mm_srli_epi32(v_p32[0], 16));
        v_p32[1] = _mm_xor_si128(v_p32[1], _mm_slli_epi32(v_p32[1], 16));
        // zip 16bit
        const uint8_t blend_u32 = 0b1010'1010;
        v_p16 = _mm_blend_epi16(v_p32[0], v_p32[1], blend_u32);
    }
    if constexpr(sizeof(poly_t) >= 1) {
        // 16bit to 1bit
        const __m128i parity_mask = _mm_set1_epi16(0x8000);
        v_p16 = _mm_xor_si128(v_p16, _mm_slli_epi64(v_p16, 8));
        v_p16 = _mm_xor_si128(v_p16, _mm_slli_epi64(v_p16, 4));
        v_p16 = _mm_xor_si128(v_p16, _mm_slli_epi64(v_p16, 2));
        v_p16 = _mm_xor_si128(v_p16, _mm_slli_epi64(v_p16, 1));
        v_p16 = _mm_cmpeq_epi16(_mm_and_si128(v_p16, parity_mask), parity_mask);
    }
    const __m128i branch_value = _mm_blendv_epi8(v_decision_low, v_decision_high, v_p16);
    return branch_value;
}
