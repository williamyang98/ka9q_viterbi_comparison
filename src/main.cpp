#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <memory>
#include "./util.h"
#include "./timer.h"
#include "./viterbi_configs.h"
#include "./ka9q_interface.h"
#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_shift_register.h"
#include "viterbi/viterbi_decoder_core.h"
#include "viterbi/viterbi_branch_table.h"
#include "viterbi/viterbi_decoder_config.h"
#include "viterbi/x86/viterbi_decoder_sse_u8.h"
#include "viterbi/x86/viterbi_decoder_sse_u16.h"
#include "viterbi/x86/viterbi_decoder_avx_u8.h"
#include "viterbi/x86/viterbi_decoder_avx_u16.h"

template <typename ka9q_t, typename reg_t>
void run_test(const size_t total_decode_bytes, const size_t total_results) {
    printf("[parameters]\n");
    constexpr size_t K = ka9q_t::K;
    constexpr size_t R = ka9q_t::R;
    printf("K=%zu, R=%zu\n", K, R);
    // create encoder
    auto encoder = ConvolutionalEncoder_ShiftRegister<reg_t>(K, R, ka9q_t::POLY);
    // create decoder
    const size_t total_decode_bits = total_decode_bytes*8;
    const size_t total_tail_bits = K-1u;
    const size_t total_transmit_bits = total_decode_bits + total_tail_bits;
    const size_t total_symbols = total_transmit_bits*R;
    // generate data
    auto x_in = std::vector<uint8_t>(total_decode_bytes);
    generate_random_bytes(x_in.data(), x_in.size());
    // run tests
    struct TestResult {
        uint64_t init_ns = 0;
        uint64_t update_symbols_ns = 0;
        uint64_t chainback_bits_ns = 0;
    };
    auto results = std::vector<TestResult>(total_results);
    auto x_out = std::vector<uint8_t>(total_decode_bytes);
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
        const double avg_update_rate = double(total_symbols) / (avg_update_ns*1e-9f);
        const double avg_chainback_rate = double(total_decode_bits) / (avg_chainback_ns*1e-9f);
        const auto avg_si_update_rate = get_si_notation(avg_update_rate);
        const auto avg_si_chainback_rate = get_si_notation(avg_chainback_rate);
        printf("update_rate:    %.2f %ssym/s\n", avg_si_update_rate.value, avg_si_update_rate.prefix);
        printf("chainback_rate: %.2f %sb/s\n", avg_si_chainback_rate.value, avg_si_chainback_rate.prefix);
        // const size_t total_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
        // const float total_bit_error_rate = float(total_errors) / float(total_decode_bits) * 100.0f;
        // printf("bit_error_rate: %.2f%%\n", total_bit_error_rate);
        // printf("bit_errors:     %zu/%zu\n", total_errors, total_decode_bits);
    };
    // test kafq
    printf("[kafq decoder]\n");
    {
        auto decoder = ka9q_t(total_transmit_bits);
        auto config = get_ka9q_offset_binary_config();
        auto y_out = std::vector<uint8_t>(total_symbols);
        encode_data<uint8_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        for (auto& result: results) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                decoder.reset();
                result.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder.update(y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                decoder.chainback(x_out.data(), total_decode_bits);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    // test ours
    {
        printf("[sse-u8]\n");
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_SSE_u8<K,R>;
        auto y_out = std::vector<soft_t>(total_symbols);
        auto config = get_soft8_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(ka9q_t::POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(total_decode_bits);
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
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    {
        printf("[avx-u8]\n");
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_AVX_u8<K,R>;
        auto y_out = std::vector<soft_t>(total_symbols);
        auto config = get_soft8_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(ka9q_t::POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(total_decode_bits);
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
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    {
        printf("[sse-u16]\n");
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_SSE_u16<K,R>;
        auto y_out = std::vector<soft_t>(total_symbols);
        auto config = get_soft16_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(ka9q_t::POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(total_decode_bits);
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
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    {
        printf("[avx-u16]\n");
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_AVX_u16<K,R>;
        auto y_out = std::vector<soft_t>(total_symbols);
        auto config = get_soft16_decoding_config(R);
        encode_data<soft_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(ka9q_t::POLY, config.soft_decision_high, config.soft_decision_low);
        auto core = std::make_unique<ViterbiDecoder_Core<K,R,error_t,soft_t>>(*branch_table, config.decoder_config);
        core->set_traceback_length(total_decode_bits);
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
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                result.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                result.chainback_bits_ns = t.get_delta();
            }
        }
        print_results();
    }
    printf("\n\n");
}

int main(int argc, char** argv) {
    run_test<ka9q_viterbi27, uint32_t>(1024, 4096*4);
    run_test<ka9q_viterbi29, uint32_t>(512, 4096);
    run_test<ka9q_viterbi615, uint32_t>(256, 256);
    run_test<ka9q_viterbi224, uint32_t>(8, 4);
    return 0;
}
