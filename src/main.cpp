#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <memory>
#include "./util.h"
#include "./timer.h"
#include "./viterbi_configs.h"
#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_shift_register.h"
#include "viterbi/viterbi_decoder_core.h"
#include "viterbi/viterbi_branch_table.h"
#include "viterbi/viterbi_decoder_config.h"
#include "viterbi/x86/viterbi_decoder_sse_u8.h"
#include "viterbi/x86/viterbi_decoder_sse_u16.h"
#include "viterbi/x86/viterbi_decoder_avx_u8.h"
#include "viterbi/x86/viterbi_decoder_avx_u16.h"

#define CONFIG 0

#if CONFIG == 0
#include "viterbi27_sse2.h"
constexpr size_t K = 7;
constexpr size_t R = 2;
using reg_t = uint32_t;
auto POLY = V27_POLY;
auto kafq_create = create_viterbi27_sse2;
auto kafq_init = init_viterbi27_sse2;
auto kafq_update = update_viterbi27_blk_sse2;
auto kafq_chainback = chainback_viterbi27_sse2;
auto kafq_delete = delete_viterbi27_sse2;
constexpr size_t TOTAL_DECODE_BYTES = 1024;
constexpr size_t TOTAL_RESULTS = 4096*4;
#elif CONFIG == 1
#include "viterbi615_sse2.h"
constexpr size_t K = 15;
constexpr size_t R = 6;
using reg_t = uint32_t;
auto POLY = V615_POLY;
auto kafq_create = create_viterbi615_sse2;
auto kafq_init = init_viterbi615_sse2;
auto kafq_update = update_viterbi615_blk_sse2;
auto kafq_chainback = chainback_viterbi615_sse2;
auto kafq_delete = delete_viterbi615_sse2;
constexpr size_t TOTAL_DECODE_BYTES = 256;
constexpr size_t TOTAL_RESULTS = 256;
#elif CONFIG == 2
#include "viterbi224_sse2.h"
constexpr size_t K = 24;
constexpr size_t R = 2;
using reg_t = uint32_t;
auto POLY = V224_POLY;
auto kafq_create = create_viterbi224_sse2;
auto kafq_init = init_viterbi224_sse2;
auto kafq_update = update_viterbi224_blk_sse2;
auto kafq_chainback = chainback_viterbi224_sse2;
auto kafq_delete = delete_viterbi224_sse2;
constexpr size_t TOTAL_DECODE_BYTES = 8;
constexpr size_t TOTAL_RESULTS = 4;
#elif CONFIG == 3
#include "viterbi29_sse2.h"
constexpr size_t K = 9;
constexpr size_t R = 2;
using reg_t = uint32_t;
auto POLY = V29_POLY;
auto kafq_create = create_viterbi29_sse2;
auto kafq_init = init_viterbi29_sse2;
auto kafq_update = update_viterbi29_blk_sse2;
auto kafq_chainback = chainback_viterbi29_sse2;
auto kafq_delete = delete_viterbi29_sse2;
constexpr size_t TOTAL_DECODE_BYTES = 1024;
constexpr size_t TOTAL_RESULTS = 4096*4;
#endif

struct SI_Notation {
    double value; 
    const char* prefix;
};

SI_Notation get_si_notation(double x) {
    if (x > 1e12) return SI_Notation { x*1e-12, "T" };
    if (x > 1e9) return SI_Notation { x*1e-9, "G" };
    if (x > 1e6) return SI_Notation { x*1e-6, "M" };
    if (x > 1e3) return SI_Notation { x*1e-3, "k" };
    return SI_Notation { x, "" };
}

int main(int argc, char** argv) {
    printf("[parameters]\n");
    printf("K=%zu, R=%zu\n", K, R);
    // create encoder
    auto encoder = ConvolutionalEncoder_ShiftRegister<reg_t>(K, R, POLY);
    // create decoder
    constexpr size_t TOTAL_DECODE_BITS = TOTAL_DECODE_BYTES*8;
    constexpr size_t TOTAL_TAIL_BITS = K-1u;
    constexpr size_t TOTAL_TRANSMIT_BITS = TOTAL_DECODE_BITS + TOTAL_TAIL_BITS;
    constexpr size_t TOTAL_SYMBOLS = TOTAL_TRANSMIT_BITS*R;
    // generate data
    auto x_in = std::vector<uint8_t>(TOTAL_DECODE_BYTES);
    generate_random_bytes(x_in.data(), x_in.size());
    // run tests
    struct TestResult {
        uint64_t init_ns = 0;
        uint64_t update_symbols_ns = 0;
        uint64_t chainback_bits_ns = 0;
    };
    auto results = std::vector<TestResult>(TOTAL_RESULTS);
    auto x_out = std::vector<uint8_t>(TOTAL_DECODE_BYTES);
    const auto print_results = [&]() {
        double avg_update_ns = 0.0f;
        double avg_chainback_ns = 0.0f;
        for (size_t i = 0; i < results.size(); i++) {
            const auto& res = results[i];
            avg_update_ns += double(res.update_symbols_ns);
            avg_chainback_ns += double(res.chainback_bits_ns);
        }
        avg_update_ns /= double(results.size());
        avg_chainback_ns /= double(results.size());
        const double avg_update_rate = double(TOTAL_SYMBOLS) / (avg_update_ns*1e-9f);
        const double avg_chainback_rate = double(TOTAL_DECODE_BITS) / (avg_chainback_ns*1e-9f);
        const auto avg_si_update_rate = get_si_notation(avg_update_rate);
        const auto avg_si_chainback_rate = get_si_notation(avg_chainback_rate);
        printf("update_rate:    %.2f %ssym/s\n", avg_si_update_rate.value, avg_si_update_rate.prefix);
        printf("chainback_rate: %.2f %sb/s\n", avg_si_chainback_rate.value, avg_si_chainback_rate.prefix);
        const size_t total_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
        const float total_bit_error_rate = float(total_errors) / float(TOTAL_DECODE_BITS) * 100.0f;
        printf("bit_error_rate: %.2f%%\n", total_bit_error_rate);
        printf("bit_errors:     %zu/%zu\n", total_errors, TOTAL_DECODE_BITS);
    };
    // test kafq
    printf("\n[kafq decoder]\n");
    {
        auto* decoder = kafq_create(TOTAL_TRANSMIT_BITS);
        auto y_out = std::vector<uint8_t>(TOTAL_SYMBOLS);
        const uint8_t MARGIN = 30; // signed types dont use saturating arithmetic so we need this
        const uint8_t SOFT_DECISION_HIGH = 255 - MARGIN;
        const uint8_t SOFT_DECISION_LOW = 0 + MARGIN;
        encode_data<uint8_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            SOFT_DECISION_HIGH, SOFT_DECISION_LOW
        );
        for (auto& result: results) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                kafq_init(decoder, 0);
                result.init_ns = t.get_delta();
            }
            {
                Timer t;
                kafq_update(decoder, y_out.data(), TOTAL_TRANSMIT_BITS);
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                kafq_chainback(decoder, x_out.data(), TOTAL_DECODE_BITS, 0);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        kafq_delete(decoder);
        print_results();
    }
    // test ours
    {
        printf("\n[sse-u8]\n");
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_SSE_u8<K,R>;
        auto y_out = std::vector<soft_t>(TOTAL_SYMBOLS);
        auto config = get_soft8_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(TOTAL_DECODE_BITS);
        for (auto& result: results) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                result.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), TOTAL_DECODE_BITS);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    {
        printf("\n[avx-u8]\n");
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_AVX_u8<K,R>;
        auto y_out = std::vector<soft_t>(TOTAL_SYMBOLS);
        auto config = get_soft8_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(TOTAL_DECODE_BITS);
        for (auto& result: results) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                result.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), TOTAL_DECODE_BITS);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    {
        printf("\n[sse-u16]\n");
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_SSE_u16<K,R>;
        auto y_out = std::vector<soft_t>(TOTAL_SYMBOLS);
        auto config = get_soft16_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(TOTAL_DECODE_BITS);
        for (auto& result: results) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                result.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), TOTAL_DECODE_BITS);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    {
        printf("\n[avx-u16]\n");
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_AVX_u16<K,R>;
        auto y_out = std::vector<soft_t>(TOTAL_SYMBOLS);
        auto config = get_soft16_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(TOTAL_DECODE_BITS);
        for (auto& result: results) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                result.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), TOTAL_DECODE_BITS);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }

    return 0;
}
