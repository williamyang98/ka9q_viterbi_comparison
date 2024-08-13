#pragma once

#include <immintrin.h>
#include <stdint.h>
#include <stdalign.h>
#include <stddef.h>

static inline __m256i mm256_get_branch_value_s8(size_t curr_state_offset, uint32_t G, int8_t branch_low, int8_t branch_high) {
    // zip the parity bits together
    alignas(32) const int32_t state_offset[8] = { 0, 4, 8, 12, 16, 20, 24, 28 };
    const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
    const __m256i v_state_offset_shifted =  _mm256_slli_epi32(v_state_offset, 1);
    alignas(32) const uint8_t blend_mask[32] = {
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
    };
    const __m256i v_blend_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(blend_mask));
    const __m256i v_G = _mm256_set1_epi32(int(G));
    const __m256i v_decision_low = _mm256_set1_epi8(branch_low);
    const __m256i v_decision_high = _mm256_set1_epi8(branch_high);

    __m256i v_p0[4];
    for (size_t j = 0; j < 4; j++) {
        const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
        const __m256i v_state = _mm256_add_epi32(v_state_offset_shifted, _mm256_set1_epi32(state_offset_shifted));
        const __m256i v_reg = _mm256_and_si256(v_state, v_G);
        const __m256i p0 = _mm256_xor_si256(v_reg, _mm256_srli_epi32(v_reg, 16));
        v_p0[j] = p0;
    }

    __m256i v_p1[2];
    v_p1[0] = _mm256_blend_epi16(v_p0[0], _mm256_slli_epi32(v_p0[2], 16), 0b1010'1010);
    v_p1[1] = _mm256_blend_epi16(v_p0[1], _mm256_slli_epi32(v_p0[3], 16), 0b1010'1010);
    v_p1[0] = _mm256_xor_si256(v_p1[0], _mm256_srli_epi16(v_p1[0], 8));
    v_p1[1] = _mm256_xor_si256(v_p1[1], _mm256_srli_epi16(v_p1[1], 8));
    const __m256i p4 = _mm256_blendv_epi8(v_p1[0], _mm256_slli_epi16(v_p1[1], 8), v_blend_mask);
    const __m256i p5 = _mm256_xor_si256(p4, _mm256_slli_epi64(p4, 4));
    const __m256i p6 = _mm256_xor_si256(p5, _mm256_slli_epi64(p5, 2));
    const __m256i p7 = _mm256_xor_si256(p6, _mm256_slli_epi64(p6, 1));
    const __m256i branch_value = _mm256_blendv_epi8(v_decision_low, v_decision_high, p7);
    return branch_value;
}

static inline __m128i mm128_get_branch_value_s8(size_t curr_state_offset, uint32_t G, int8_t branch_low, int8_t branch_high) {
    // zip the parity bits together
    alignas(16) const int32_t state_offset[4] = { 0, 4, 8, 12 };
    const __m128i v_state_offset = _mm_load_si128(reinterpret_cast<const __m128i*>(state_offset));
    const __m128i v_state_offset_shifted =  _mm_slli_epi32(v_state_offset, 1);
    alignas(16) const uint8_t blend_mask[16] = {
        0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
    };
    const __m128i v_blend_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(blend_mask));
    const __m128i v_G = _mm_set1_epi32(int(G));
    const __m128i v_decision_low = _mm_set1_epi8(branch_low);
    const __m128i v_decision_high = _mm_set1_epi8(branch_high);

    __m128i v_p0[4];
    for (size_t j = 0; j < 4; j++) {
        const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
        const __m128i v_state = _mm_add_epi32(v_state_offset_shifted, _mm_set1_epi32(state_offset_shifted));
        const __m128i v_reg = _mm_and_si128(v_state, v_G);
        const __m128i p0 = _mm_xor_si128(v_reg, _mm_srli_epi32(v_reg, 16));
        v_p0[j] = p0;
    }

    __m128i v_p1[2];
    v_p1[0] = _mm_blend_epi16(v_p0[0], _mm_slli_epi32(v_p0[2], 16), 0b1010'1010);
    v_p1[1] = _mm_blend_epi16(v_p0[1], _mm_slli_epi32(v_p0[3], 16), 0b1010'1010);
    v_p1[0] = _mm_xor_si128(v_p1[0], _mm_srli_epi16(v_p1[0], 8));
    v_p1[1] = _mm_xor_si128(v_p1[1], _mm_srli_epi16(v_p1[1], 8));
    const __m128i p4 = _mm_blendv_epi8(v_p1[0], _mm_slli_epi16(v_p1[1], 8), v_blend_mask);
    const __m128i p5 = _mm_xor_si128(p4, _mm_slli_epi64(p4, 4));
    const __m128i p6 = _mm_xor_si128(p5, _mm_slli_epi64(p5, 2));
    const __m128i p7 = _mm_xor_si128(p6, _mm_slli_epi64(p6, 1));
    const __m128i branch_value = _mm_blendv_epi8(v_decision_low, v_decision_high, p7);
    return branch_value;
}

static inline __m256i mm256_get_branch_value_s16(size_t curr_state_offset, uint32_t G, int16_t branch_low, int16_t branch_high) {
    // zip the parity bits together
    alignas(32) const int32_t state_offset[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };
    const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
    const __m256i v_state_offset_shifted =  _mm256_slli_epi32(v_state_offset, 1);
    const __m256i v_G = _mm256_set1_epi32(int(G));
    const __m256i v_decision_low = _mm256_set1_epi16(branch_low);
    const __m256i v_decision_high = _mm256_set1_epi16(branch_high);

    __m256i v_p0[2];
    for (size_t j = 0; j < 2; j++) {
        const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
        const __m256i v_state = _mm256_add_epi32(v_state_offset_shifted, _mm256_set1_epi32(state_offset_shifted));
        const __m256i v_reg = _mm256_and_si256(v_state, v_G);
        const __m256i p0 = _mm256_xor_si256(v_reg, _mm256_srli_epi32(v_reg, 16));
        v_p0[j] = p0;
    }

    const __m256i p1 = _mm256_blend_epi16(v_p0[0], _mm256_slli_epi32(v_p0[1], 16), 0b1010'1010);
    const __m256i p4 = _mm256_xor_si256(p1, _mm256_slli_epi64(p1, 8));
    const __m256i p5 = _mm256_xor_si256(p4, _mm256_slli_epi64(p4, 4));
    const __m256i p6 = _mm256_xor_si256(p5, _mm256_slli_epi64(p5, 2));
    const __m256i p7 = _mm256_xor_si256(p6, _mm256_slli_epi64(p6, 1));

    const __m256i parity_mask = _mm256_set1_epi16(0x8000);
    const __m256i p8 = _mm256_cmpeq_epi16(_mm256_and_si256(p7, parity_mask), parity_mask);
    const __m256i branch_value = _mm256_blendv_epi8(v_decision_low, v_decision_high, p8);
    return branch_value;
}

static inline __m128i mm128_get_branch_value_s16(size_t curr_state_offset, uint32_t G, int16_t branch_low, int16_t branch_high) {
    // zip the parity bits together
    alignas(32) const int32_t state_offset[4] = { 0, 2, 4, 6 };
    const __m128i v_state_offset = _mm_load_si128(reinterpret_cast<const __m128i*>(state_offset));
    const __m128i v_state_offset_shifted =  _mm_slli_epi32(v_state_offset, 1);
    const __m128i v_G = _mm_set1_epi32(int(G));
    const __m128i v_decision_low = _mm_set1_epi16(branch_low);
    const __m128i v_decision_high = _mm_set1_epi16(branch_high);

    __m128i v_p0[2];
    for (size_t j = 0; j < 2; j++) {
        const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
        const __m128i v_state = _mm_add_epi32(v_state_offset_shifted, _mm_set1_epi32(state_offset_shifted));
        const __m128i v_reg = _mm_and_si128(v_state, v_G);
        const __m128i p0 = _mm_xor_si128(v_reg, _mm_srli_epi32(v_reg, 16));
        v_p0[j] = p0;
    }

    const __m128i p1 = _mm_blend_epi16(v_p0[0], _mm_slli_epi32(v_p0[1], 16), 0b1010'1010);
    const __m128i p4 = _mm_xor_si128(p1, _mm_slli_epi64(p1, 8));
    const __m128i p5 = _mm_xor_si128(p4, _mm_slli_epi64(p4, 4));
    const __m128i p6 = _mm_xor_si128(p5, _mm_slli_epi64(p5, 2));
    const __m128i p7 = _mm_xor_si128(p6, _mm_slli_epi64(p6, 1));

    const __m128i parity_mask = _mm_set1_epi16(0x8000);
    const __m128i p8 = _mm_cmpeq_epi16(_mm_and_si128(p7, parity_mask), parity_mask);
    const __m128i branch_value = _mm_blendv_epi8(v_decision_low, v_decision_high, p8);
    return branch_value;
}
