#pragma once
#include <random>
#include <stdint.h>
#include <assert.h>
#include "viterbi/convolutional_encoder.h"
#include "./bitcount.h"

static void generate_random_bytes(uint8_t* data, const size_t N) {
    for (size_t i = 0u; i < N; i++) {
        data[i] = uint8_t(std::rand() % 256);
    }
}

template <typename T>
static size_t encode_data(
    ConvolutionalEncoder* enc, 
    const uint8_t* input_bytes, const size_t total_input_bytes,
    T* output_symbols, const size_t max_output_symbols,
    const T soft_decision_high,
    const T soft_decision_low) 
{
    const size_t K = enc->K;
    const size_t R = enc->R;

    const size_t total_input_bits = total_input_bytes*8;
    const size_t total_tail_bits = K-1;
    const size_t total_output_symbols = (total_input_bits + total_tail_bits) * R;
    assert(total_output_symbols <= max_output_symbols);

    size_t curr_output_symbol = 0u;
    auto push_symbols = [&](const uint8_t* buf, const size_t total_bits) {
        for (size_t i = 0u; i < total_bits; i++) {
            const size_t curr_byte = i / 8;
            const size_t curr_bit = i % 8;
            const bool bit = (buf[curr_byte] >> curr_bit) & 0b1;
            output_symbols[curr_output_symbol] = bit ? soft_decision_high : soft_decision_low;
            curr_output_symbol++;                
        }
    };

    auto symbols = std::vector<uint8_t>(R);

    // encode input bytes
    for (size_t i = 0u; i < total_input_bytes; i++) {
        const uint8_t x = input_bytes[i];
        enc->consume_byte(x, symbols.data());
        push_symbols(symbols.data(), 8u*R);
    }

    // terminate tail at state 0
    for (size_t i = 0u; i < total_tail_bits; ) {
        const size_t remain_bits = total_tail_bits-i;
        // const size_t total_bits = min(remain_bits, size_t(8u));
        const size_t total_bits = (remain_bits > 8) ? 8 : remain_bits;
        enc->consume_byte(0x00, symbols.data());
        push_symbols(symbols.data(), total_bits*R);
        i += total_bits;
    }

    assert(curr_output_symbol == total_output_symbols);
    return total_output_symbols;
}

static size_t get_total_bit_errors(const uint8_t* x0, const uint8_t* x1, const size_t N) {
    auto& bitcount_table = BitcountTable::get();
    size_t total_errors = 0u;
    for (size_t i = 0u; i < N; i++) {
        const uint8_t error_mask = x0[i] ^ x1[i];
        const uint8_t bit_errors = bitcount_table.parse(error_mask);
        total_errors += size_t(bit_errors);
    }
    return total_errors;
}

struct SI_Notation {
    double value; 
    const char* prefix;
};

static SI_Notation get_si_notation(double x) {
    if (x > 1e12) return SI_Notation { x*1e-12, "T" };
    if (x > 1e9) return SI_Notation { x*1e-9, "G" };
    if (x > 1e6) return SI_Notation { x*1e-6, "M" };
    if (x > 1e3) return SI_Notation { x*1e-3, "k" };
    return SI_Notation { x, "" };
}

