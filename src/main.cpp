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

struct TestSample {
    uint64_t init_ns = 0;
    uint64_t update_symbols_ns = 0;
    uint64_t chainback_bits_ns = 0;
};

struct ResultAverage {
    double avg_update_ns = 0.0f;
    double avg_chainback_ns = 0.0f;
};

struct TestResults {
    size_t K;
    size_t R;
    size_t total_samples;
    size_t total_input_bytes;
    size_t total_output_symbols;
    ResultAverage ka9q;
    ResultAverage sse_u8;
    ResultAverage sse_u16;
    ResultAverage avx_u8;
    ResultAverage avx_u16;
};

ResultAverage get_results_average(const TestSample* samples, const size_t N) {
    double avg_update_ns = 0.0f;
    double avg_chainback_ns = 0.0f;
    for (size_t i = 0; i < N; i++) {
        const auto& res = samples[i];
        avg_update_ns += double(res.update_symbols_ns);
        avg_chainback_ns += double(res.chainback_bits_ns);
    }
    avg_update_ns /= double(N);
    avg_chainback_ns /= double(N);
    return { avg_update_ns, avg_chainback_ns };
}

template <typename ka9q_t, typename reg_t>
TestResults run_test(const size_t total_decode_bytes, const size_t total_samples) {
    printf("[test_run]\n");
    constexpr size_t K = ka9q_t::K;
    constexpr size_t R = ka9q_t::R;
    printf("K=%zu, R=%zu\n", K, R);
    printf("total_input_bytes = %zu\n", total_decode_bytes);
    printf("total_samples = %zu\n", total_samples);
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
    auto results = TestResults();
    auto samples = std::vector<TestSample>(total_samples);
    results.K = K;
    results.R = R;
    results.total_samples = total_samples;
    results.total_input_bytes = total_decode_bytes;
    results.total_output_symbols = total_symbols;
    auto x_out = std::vector<uint8_t>(total_decode_bytes);
    // test kafq
    {
        printf("- kafq decoder\r");
        auto decoder = ka9q_t(total_transmit_bits);
        auto config = get_ka9q_offset_binary_config();
        auto y_out = std::vector<uint8_t>(total_symbols);
        encode_data<uint8_t>(
            &encoder,
            x_in.data(), x_in.size(), y_out.data(), y_out.size(),
            config.soft_decision_high, config.soft_decision_low
        );
        for (auto& sample: samples) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                decoder.reset();
                sample.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder.update(y_out.data(), y_out.size());
                sample.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                decoder.chainback(x_out.data(), total_decode_bits);
                sample.chainback_bits_ns = t.get_delta();
            }
        }
        results.ka9q = get_results_average(samples.data(), samples.size());
        const size_t total_bit_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
        const double bit_error_rate = double(total_bit_errors) / double(x_in.size()*8);
        printf("o kafq decoder (%.3f)\n", bit_error_rate);
    }
    // test ours
    {
        printf("- sse_u8\r");
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
        for (auto& sample: samples) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                sample.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                sample.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                sample.chainback_bits_ns = t.get_delta();
            }
        }
        results.sse_u8 = get_results_average(samples.data(), samples.size());
        const size_t total_bit_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
        const double bit_error_rate = double(total_bit_errors) / double(x_in.size()*8);
        printf("o sse_u8 (%.3f)\n", bit_error_rate);
    }
    {
        printf("- avx_u8\r");
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
        for (auto& sample: samples) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                sample.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                sample.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                sample.chainback_bits_ns = t.get_delta();
            }
        }
        results.avx_u8 = get_results_average(samples.data(), samples.size());
        const size_t total_bit_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
        const double bit_error_rate = double(total_bit_errors) / double(x_in.size()*8);
        printf("o avx_u8 (%.3f)\n", bit_error_rate);
    }
    {
        printf("- sse_u16\r");
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
        for (auto& sample: samples) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                sample.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                sample.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                sample.chainback_bits_ns = t.get_delta();
            }
        }
        results.sse_u16 = get_results_average(samples.data(), samples.size());
        const size_t total_bit_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
        const double bit_error_rate = double(total_bit_errors) / double(x_in.size()*8);
        printf("o sse_u16 (%.3f)\n", bit_error_rate);
    }
    {
        printf("- avx_u16\r");
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
        for (auto& sample: samples) {
            {
                for (auto& x: x_out) x = 0x00;
            }
            {
                Timer t;
                core->reset();
                sample.init_ns = t.get_delta();
            }
            {
                Timer t;
                decoder::template update<uint64_t>(*core, y_out.data(), y_out.size());
                sample.update_symbols_ns = t.get_delta();
            }
            {
                Timer t;
                core->chainback(x_out.data(), total_decode_bits);
                sample.chainback_bits_ns = t.get_delta();
            }
        }
        results.avx_u16 = get_results_average(samples.data(), samples.size());
        const size_t total_bit_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
        const double bit_error_rate = double(total_bit_errors) / double(x_in.size()*8);
        printf("o avx_u16 (%.3f)\n", bit_error_rate);
    }
    printf("\n");
    return results;
}

int main(int argc, char** argv) {
    std::vector<TestResults> results;
    results.push_back(run_test<ka9q_viterbi27, uint32_t>(1024, 4096*4));
    results.push_back(run_test<ka9q_viterbi29, uint32_t>(512, 4096));
    results.push_back(run_test<ka9q_viterbi615, uint32_t>(256, 256));
    results.push_back(run_test<ka9q_viterbi224, uint32_t>(8, 4));

    printf("[update (symbols/s)]\n");
    printf("| K    | R    | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |\n");
    printf("| ---- | ---- | ---- | ------ | ------ | ------- | ------- |\n");
    for (const auto& result: results) {
        SI_Notation si[5];
        const double K = result.total_output_symbols * 1e9;
        si[0] = get_si_notation(K/result.ka9q.avg_update_ns);
        si[1] = get_si_notation(K/result.sse_u8.avg_update_ns);
        si[2] = get_si_notation(K/result.avx_u8.avg_update_ns);
        si[3] = get_si_notation(K/result.sse_u16.avg_update_ns);
        si[4] = get_si_notation(K/result.avx_u16.avg_update_ns);
        printf(
            "| %zu | %zu | %.3g%s | %.3g%s | %.3g%s | %.3g%s | %.3g%s |\n",
            result.K, result.R,
            si[0].value, si[0].prefix,
            si[1].value, si[1].prefix,
            si[2].value, si[2].prefix,
            si[3].value, si[3].prefix,
            si[4].value, si[4].prefix
        );
    }
    printf("\n");

    printf("[chainback (bits/s)]\n");
    printf("| K    | R    | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |\n");
    printf("| ---- | ---- | ---- | ------ | ------ | ------- | ------- |\n");
    for (const auto& result: results) {
        SI_Notation si[5];
        const double K = result.total_input_bytes * 8 * 1e9;
        si[0] = get_si_notation(K/result.ka9q.avg_chainback_ns);
        si[1] = get_si_notation(K/result.sse_u8.avg_chainback_ns);
        si[2] = get_si_notation(K/result.avx_u8.avg_chainback_ns);
        si[3] = get_si_notation(K/result.sse_u16.avg_chainback_ns);
        si[4] = get_si_notation(K/result.avx_u16.avg_chainback_ns);
        printf(
            "| %zu | %zu | %.3g%s | %.3g%s | %.3g%s | %.3g%s | %.3g%s |\n",
            result.K, result.R,
            si[0].value, si[0].prefix,
            si[1].value, si[1].prefix,
            si[2].value, si[2].prefix,
            si[3].value, si[3].prefix,
            si[4].value, si[4].prefix
        );
    }
    printf("\n");

    return 0;
}
