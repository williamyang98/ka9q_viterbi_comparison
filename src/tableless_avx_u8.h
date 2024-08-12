/* Copyright 2004-2014, Phil Karn, KA9Q
* Phil Karn's github repository: https://github.com/ka9q/libfec 
* May be used under the terms of the GNU Lesser General Public License (LGPL)
* 
* Modified by author, William Yang
* 07/2023 - Generalised decoder using AVX2 instructions for 8bit types giving 32 way speedup.
*/
#pragma once
#include "viterbi/viterbi_decoder_core.h"
#include <stdint.h>
#include <stddef.h>
#include <stdalign.h>
#include <assert.h>
#include <immintrin.h>

struct ViterbiDecoder_Tableless_Config_AVX_u8 {
    int8_t soft_decision_high;
    int8_t soft_decision_low;
    const uint32_t* poly;
};

/// @brief Vectorisation using AVX2.
//         8bit integers for errors, soft-decision values.
//         32 way vectorisation from 256bits/8bits.
template <size_t constraint_length, size_t code_rate>
class ViterbiDecoder_Tableless_AVX_u8
{
private:
    using Base = ViterbiDecoder_Core<constraint_length,code_rate,uint8_t,int8_t>;
    using Config = ViterbiDecoder_Tableless_Config_AVX_u8;
    using decision_bits_t = typename Base::Decisions::format_t;
private:
    // Calculate the minimum constraint length for vectorisation
    // We require: stride(metric)/2 = stride(branch_table) = stride(decision)
    // total_states = 2^(K-1)
    //
    // sizeof(metric)       = total_states   * sizeof(u8) = 2^(K-1)
    // sizeof(branch_table) = total_states/2 * sizeof(s8) = 2^(K-2)
    //
    // sizeof(__m256i)      = 32 = 2^5
    // stride(metric)       = sizeof(metric)       / sizeof(__m128i) = 2^(K-6)
    // stride(branch_table) = sizeof(branch_table) / sizeof(__m128i) = 2^(K-7)
    //
    // For stride(...) >= 1, then K >= 7
    static constexpr size_t SIMD_ALIGN = sizeof(__m256i);
    static constexpr size_t v_stride_metric = Base::Metrics::SIZE_IN_BYTES/SIMD_ALIGN;
    static constexpr size_t K_min = 7;
public:
    static constexpr bool is_valid = Base::K >= K_min;
    template <typename sum_error_t>
    static sum_error_t update(Base& base, const int8_t* symbols, const size_t N, const Config& config) {
        static_assert(is_valid, "Insufficient constraint length for vectorisation");
        static_assert(Base::Metrics::ALIGNMENT % SIMD_ALIGN == 0);
        // number of symbols must be a multiple of the code rate
        assert(N % Base::R == 0);
        const size_t total_decoded_bits = N / Base::R;
        const size_t max_decoded_bits = base.get_traceback_length() + Base::TOTAL_STATE_BITS;
        assert((total_decoded_bits + base.m_current_decoded_bit) <= max_decoded_bits);
        sum_error_t total_error = 0;
        for (size_t s = 0; s < N; s+=Base::R) {
            auto* decision = base.m_decisions[base.m_current_decoded_bit];
            auto* old_metric = base.m_metrics.get_old();
            auto* new_metric = base.m_metrics.get_new();
            bfly(base, &symbols[s], decision, old_metric, new_metric, config);
            if (new_metric[0] >= base.m_config.renormalisation_threshold) {
                total_error += sum_error_t(renormalise(new_metric));
            }
            base.m_metrics.swap();
            base.m_current_decoded_bit++;
        }
        return total_error;
   }
private:
    static void bfly(Base& base, const int8_t* symbols, decision_bits_t* decision, uint8_t* old_metric, uint8_t* new_metric, const Config& config) {
        __m256i* v_old_metrics = reinterpret_cast<__m256i*>(old_metric);
        __m256i* v_new_metrics = reinterpret_cast<__m256i*>(new_metric);
        uint64_t* v_decision = reinterpret_cast<uint64_t*>(decision);

        assert(uintptr_t(v_old_metrics)  % SIMD_ALIGN == 0);
        assert(uintptr_t(v_new_metrics)  % SIMD_ALIGN == 0);

        __m256i v_symbols[Base::R];
        __m256i v_poly[Base::R];
        for (size_t i = 0; i < Base::R; i++) {
            v_poly[i] = _mm256_set1_epi32(int(config.poly[i]));
        }
        alignas(32) static const int32_t state_offset[8] = { 0, 4, 8, 12, 16, 20, 24, 28 };
        static const __m256i v_state_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(state_offset));
        static const __m256i v_state_offset_shifted =  _mm256_slli_epi32(v_state_offset, 1);
        alignas(32) static const uint8_t blend_mask[32] = {
            0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
        };
        static const __m256i v_blend_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(blend_mask));

        // Vectorise constants
        for (size_t i = 0; i < Base::R; i++) {
            v_symbols[i] = _mm256_set1_epi8(symbols[i]);
        }
        const __m256i max_error = _mm256_set1_epi8(base.m_config.soft_decision_max_error);
        const __m256i v_decision_high = _mm256_set1_epi8(config.soft_decision_high);
        const __m256i v_decision_low = _mm256_set1_epi8(config.soft_decision_low);

        // We are using the butterfly optimisation trick so we only need to compute half the states
        constexpr size_t TOTAL_STATE_BITS = constraint_length-1;
        constexpr size_t NUMSTATES = (size_t(1) << TOTAL_STATE_BITS)/2;
        constexpr size_t SIZE_IN_BYTES = sizeof(int8_t)*NUMSTATES;
        constexpr size_t v_stride_branch_table = SIZE_IN_BYTES/SIMD_ALIGN;

        for (size_t curr_state = 0u; curr_state < v_stride_branch_table; curr_state++) {
            const size_t curr_state_offset = curr_state*32;
            // Total errors across R symbols
            __m256i total_error = _mm256_set1_epi8(0);
            for (size_t i = 0u; i < Base::R; i++) {
                const __m256i G = v_poly[i];
                __m256i v_p0_0;
                __m256i v_p0_1;
                __m256i v_p0_2;
                __m256i v_p0_3;
                {
                    constexpr size_t j = 0;
                    const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
                    const __m256i v_state = _mm256_add_epi32(v_state_offset_shifted, _mm256_set1_epi32(state_offset_shifted));
                    const __m256i v_reg = _mm256_and_si256(v_state, G);
                    const __m256i p0 = _mm256_xor_si256(v_reg, _mm256_srli_epi32(v_reg, 16));
                    v_p0_0 = p0;
                }
                {
                    constexpr size_t j = 1;
                    const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
                    const __m256i v_state = _mm256_add_epi32(v_state_offset_shifted, _mm256_set1_epi32(state_offset_shifted));
                    const __m256i v_reg = _mm256_and_si256(v_state, G);
                    const __m256i p0 = _mm256_xor_si256(v_reg, _mm256_srli_epi32(v_reg, 16));
                    v_p0_1 = p0;
                }
                {
                    constexpr size_t j = 2;
                    const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
                    const __m256i v_state = _mm256_add_epi32(v_state_offset_shifted, _mm256_set1_epi32(state_offset_shifted));
                    const __m256i v_reg = _mm256_and_si256(v_state, G);
                    const __m256i p0 = _mm256_xor_si256(v_reg, _mm256_srli_epi32(v_reg, 16));
                    v_p0_2 = p0;
                }
                {
                    constexpr size_t j = 3;
                    const uint32_t state_offset_shifted = uint32_t((curr_state_offset + j) << 1);
                    const __m256i v_state = _mm256_add_epi32(v_state_offset_shifted, _mm256_set1_epi32(state_offset_shifted));
                    const __m256i v_reg = _mm256_and_si256(v_state, G);
                    const __m256i p0 = _mm256_xor_si256(v_reg, _mm256_srli_epi32(v_reg, 16));
                    v_p0_3 = p0;
                }
                __m256i v_p1_0 = _mm256_blend_epi16(v_p0_0, _mm256_slli_epi32(v_p0_2, 16), 0b1010'1010);
                __m256i v_p1_1 = _mm256_blend_epi16(v_p0_1, _mm256_slli_epi32(v_p0_3, 16), 0b1010'1010);
                v_p1_0 = _mm256_xor_si256(v_p1_0, _mm256_srli_epi16(v_p1_0, 8));
                v_p1_1 = _mm256_xor_si256(v_p1_1, _mm256_srli_epi16(v_p1_1, 8));
                const __m256i p4 = _mm256_blendv_epi8(v_p1_0, _mm256_slli_epi16(v_p1_1, 8), v_blend_mask);
                const __m256i p5 = _mm256_xor_si256(p4, _mm256_slli_epi64(p4, 4));
                const __m256i p6 = _mm256_xor_si256(p5, _mm256_slli_epi64(p5, 2));
                const __m256i p7 = _mm256_xor_si256(p6, _mm256_slli_epi64(p6, 1));
                const __m256i branch_value = _mm256_blendv_epi8(v_decision_low, v_decision_high, p7);
                __m256i error = _mm256_subs_epi8(branch_value, v_symbols[i]);
                error = _mm256_abs_epi8(error);
                total_error = _mm256_adds_epu8(total_error, error);
            }
            // Butterfly algorithm
            const size_t curr_state_0 = curr_state;
            const size_t curr_state_1 = curr_state + v_stride_metric/2;
            const size_t next_state_0 = (curr_state << 1) | 0;
            const size_t next_state_1 = (curr_state << 1) | 1;
            const __m256i inverse_error = _mm256_subs_epu8(max_error, total_error);
            const __m256i next_error_0_0 = _mm256_adds_epu8(v_old_metrics[curr_state_0],   total_error);
            const __m256i next_error_1_0 = _mm256_adds_epu8(v_old_metrics[curr_state_1], inverse_error);
            const __m256i next_error_0_1 = _mm256_adds_epu8(v_old_metrics[curr_state_0], inverse_error);
            const __m256i next_error_1_1 = _mm256_adds_epu8(v_old_metrics[curr_state_1],   total_error);
            const __m256i min_next_error_0 = _mm256_min_epu8(next_error_0_0, next_error_1_0);
            const __m256i min_next_error_1 = _mm256_min_epu8(next_error_0_1, next_error_1_1);
            const __m256i decision_0 = _mm256_cmpeq_epi8(min_next_error_0, next_error_1_0);
            const __m256i decision_1 = _mm256_cmpeq_epi8(min_next_error_1, next_error_1_1);
            // Update metrics
            const __m256i new_metric_lo = _mm256_unpacklo_epi8(min_next_error_0, min_next_error_1);
            const __m256i new_metric_hi = _mm256_unpackhi_epi8(min_next_error_0, min_next_error_1);
            // Reshuffle into correct order along 128bit boundaries
            v_new_metrics[next_state_0] = _mm256_permute2x128_si256(new_metric_lo, new_metric_hi, 0b0010'0000);
            v_new_metrics[next_state_1] = _mm256_permute2x128_si256(new_metric_lo, new_metric_hi, 0b0011'0001);
            // Pack decision bits
            const __m256i shuffled_decision_lo = _mm256_unpacklo_epi8(decision_0, decision_1);
            const __m256i shuffled_decision_hi = _mm256_unpackhi_epi8(decision_0, decision_1);
            // Reshuffle into correct order along 128bit boundaries
            const __m256i packed_decision_lo = _mm256_permute2x128_si256(shuffled_decision_lo, shuffled_decision_hi, 0b0010'0000);
            const __m256i packed_decision_hi = _mm256_permute2x128_si256(shuffled_decision_lo, shuffled_decision_hi, 0b0011'0001);
            uint64_t decision_bits_lo = uint64_t(_mm256_movemask_epi8(packed_decision_lo));
            uint64_t decision_bits_hi = uint64_t(_mm256_movemask_epi8(packed_decision_hi));
            // NOTE: mm256_movemask doesn't zero out the upper 32bits
            decision_bits_lo &= uint64_t(0xFFFFFFFF);
            decision_bits_hi &= uint64_t(0xFFFFFFFF);
            v_decision[curr_state] = uint64_t(decision_bits_hi << 32u) | decision_bits_lo;
        }
    }

    static uint8_t renormalise(uint8_t* metric) {
        assert(uintptr_t(metric) % SIMD_ALIGN == 0);
        __m256i* v_metric = reinterpret_cast<__m256i*>(metric);
        union alignas(SIMD_ALIGN) {
            __m256i m256;
            __m128i m128[2];
            uint8_t u8[32]; 
        } reduce_buffer;
        // Find minimum 
        reduce_buffer.m256 = v_metric[0];
        for (size_t i = 1u; i < v_stride_metric; i++) {
            reduce_buffer.m256 = _mm256_min_epu8(reduce_buffer.m256, v_metric[i]);
        }
        // Shift half of the array onto the other half and get the minimum between them
        // Repeat this until we get the minimum value of all 16bit values
        // NOTE: srli performs shift on 128bit lanes
        __m128i adjustv = _mm_min_epu8(reduce_buffer.m128[0], reduce_buffer.m128[1]);
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 8));
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 4));
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 2));
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 1));
        reduce_buffer.m128[0] = adjustv;
        const uint8_t min = reduce_buffer.u8[0];
        const __m256i vmin = _mm256_set1_epi8(min);
        // Normalise to minimum
        for (size_t i = 0u; i < v_stride_metric; i++) {
            v_metric[i] = _mm256_subs_epu8(v_metric[i], vmin);
        }
        return min;
    }
};

