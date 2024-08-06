#include <memory>
#include <optional>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "./span.h"
#include "./util.h"
#include "./timer.h"
#include "./viterbi_configs.h"
#include "./ka9q_interface.h"
#include "./spiral_interface.h"
#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_shift_register.h"
#include "viterbi/viterbi_decoder_core.h"
#include "viterbi/viterbi_branch_table.h"
#include "viterbi/viterbi_decoder_config.h"
#include "viterbi/x86/viterbi_decoder_sse_u8.h"
#include "viterbi/x86/viterbi_decoder_sse_u16.h"
#include "viterbi/x86/viterbi_decoder_avx_u8.h"
#include "viterbi/x86/viterbi_decoder_avx_u16.h"

#if _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

static const int V27_POLY[2] = { 0x6d, 0x4f };
static const int V29_POLY[2] = { 0x1af, 0x11d };
static const int V47_POLY[4] = { 121, 117, 91, 111 };
static const int V49_POLY[4] = { 501, 441, 331, 315 };
static const int V615_POLY[6] = { 042631, 047245, 056507, 073363, 077267, 064537 };
static const int V224_POLY[2] = { 062650457, 062650455 };

struct TestSample {
    uint64_t init_ns = 0;
    uint64_t update_symbols_ns = 0;
    uint64_t chainback_bits_ns = 0;
    double update_symbols_rate = 0.0;
    double chainback_bits_rate = 0.0;
};

struct TestResult {
    double avg_update_ns = 0.0f;
    double std_update_ns = 0.0f;
    double avg_chainback_ns = 0.0f;
    double std_chainback_ns = 0.0f;
    double avg_update_rate = 0.0f;
    double std_update_rate = 0.0f;
    double avg_chainback_rate = 0.0f;
    double std_chainback_rate = 0.0f;
    double bit_error_rate = 0.0f;
};

struct Test {
    size_t K;
    size_t R;
    size_t total_samples;
    size_t total_transmit_bits;
    size_t total_input_bytes;
    size_t total_output_symbols;
    const int* poly;
    std::vector<uint8_t> x_in;
    std::vector<uint8_t> x_out;
    std::vector<TestSample> samples;
};

TestResult get_test_result(tcb::span<const TestSample> samples) {
    const size_t N = samples.size();
    // calculate mean
    double avg_update_ns = 0.0f;
    double avg_chainback_ns = 0.0f;
    double avg_update_rate = 0.0f;
    double avg_chainback_rate = 0.0f;
    for (size_t i = 0; i < N; i++) {
        const auto& res = samples[i];
        avg_update_ns += double(res.update_symbols_ns);
        avg_chainback_ns += double(res.chainback_bits_ns);
        avg_update_rate += res.update_symbols_rate;
        avg_chainback_rate += res.chainback_bits_rate;
    }
    avg_update_ns /= double(N);
    avg_chainback_ns /= double(N);
    avg_update_rate /= double(N);
    avg_chainback_rate /= double(N);

    // calculate std
    double std_update_ns = 0.0f;
    double std_chainback_ns = 0.0f;
    double std_update_rate = 0.0f;
    double std_chainback_rate = 0.0f;
    const auto square = [](double x) { return x*x; };
    for (size_t i = 0; i < N; i++) {
        const auto& res = samples[i];
        std_update_ns += square(double(res.update_symbols_ns) - avg_update_ns);
        std_chainback_ns += square(double(res.chainback_bits_ns) - avg_chainback_ns);
        std_update_rate += square(res.update_symbols_rate - avg_update_rate);
        std_chainback_rate += square(res.chainback_bits_rate - avg_chainback_rate);
    }
    std_update_ns = sqrt(std_update_ns/double(N));
    std_chainback_ns = sqrt(std_chainback_ns/double(N));
    std_update_rate = sqrt(std_update_rate/double(N));
    std_chainback_rate = sqrt(std_chainback_rate/double(N));
    return { 
        avg_update_ns, std_update_ns,
        avg_chainback_ns, std_chainback_ns,
        avg_update_rate, std_update_rate,
        avg_chainback_rate, std_chainback_rate,
    };
}

using test_results_t = std::vector<std::optional<TestResult>>;

template <size_t K, size_t R>
Test init_test(const int* poly, const size_t total_decode_bytes, const size_t total_samples) {
    printf("[test_run]\n");
    printf("K=%zu, R=%zu\n", K, R);
    printf("total_input_bytes = %zu\n", total_decode_bytes);
    printf("total_samples = %zu\n", total_samples);
    // create decoder
    const size_t total_decode_bits = total_decode_bytes*8;
    const size_t total_tail_bits = K-1u;
    const size_t total_transmit_bits = total_decode_bits + total_tail_bits;
    const size_t total_symbols = total_transmit_bits*R;
    // generate data
    auto x_in = std::vector<uint8_t>(total_decode_bytes);
    generate_random_bytes(x_in.data(), x_in.size());
    // run tests
    auto test = Test();
    test.K = K;
    test.R = R;
    test.poly = poly;
    test.total_samples = total_samples;
    test.total_input_bytes = total_decode_bytes;
    test.total_output_symbols = total_symbols;
    test.total_transmit_bits = total_transmit_bits;
    test.samples.resize(total_samples);
    test.x_in = x_in;
    test.x_out.resize(total_decode_bytes);
    return test;
}

// test ours
template <size_t K, size_t R, typename soft_t, typename error_t, class decoder_t>
TestResult test_ours_single(Test& test, Decoder_Config<soft_t, error_t> config) {
    const size_t total_decode_bits = test.total_input_bytes*8;
    const int* poly = test.poly;
    const auto& x_in = test.x_in;
    auto& x_out = test.x_out;
    auto& samples = test.samples;
    using reg_t = uint32_t;
    auto encoder = ConvolutionalEncoder_ShiftRegister<reg_t>(K, R, poly);
    auto y_out = std::vector<soft_t>(test.total_output_symbols);
    encode_data<soft_t>(
        &encoder,
        x_in.data(), x_in.size(), y_out.data(), y_out.size(),
        config.soft_decision_high, config.soft_decision_low
    );
    auto branch_table = std::make_unique<ViterbiBranchTable<K,R,soft_t>>(poly, config.soft_decision_high, config.soft_decision_low);
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
            decoder_t::template update<uint64_t>(*core, y_out.data(), y_out.size());
            sample.update_symbols_ns = t.get_delta();
            sample.update_symbols_rate = y_out.size() / double(sample.update_symbols_ns) * 1e9;
        }
        {
            Timer t;
            core->chainback(x_out.data(), total_decode_bits);
            sample.chainback_bits_ns = t.get_delta();
            sample.chainback_bits_rate = total_decode_bits / double(sample.chainback_bits_ns) * 1e9;
        }
    }
    auto result = get_test_result(samples);
    const size_t total_bit_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
    const double bit_error_rate = double(total_bit_errors) / double(x_in.size()*8);
    result.bit_error_rate = bit_error_rate;
    return result;
}

template <size_t K, size_t R>
void test_ours(Test& test, test_results_t& results) {
    {
        printf("- sse_u8\r");
        fflush(stdout);
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_SSE_u8<K,R>;
        auto config = get_soft8_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>(test, config);
        printf("o sse_u8 (%.3f)\n", result.bit_error_rate);
        results.push_back(result);
    }
    {
        printf("- avx_u8\r");
        fflush(stdout);
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_AVX_u8<K,R>;
        auto config = get_soft8_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>(test, config);
        printf("o avx_u8 (%.3f)\n", result.bit_error_rate);
        results.push_back(result);
    }
    {
        printf("- sse_u16\r");
        fflush(stdout);
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_SSE_u16<K,R>;
        auto config = get_soft16_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>(test, config);
        printf("o sse_u16 (%.3f)\n", result.bit_error_rate);
        results.push_back(result);
    }
    {
        printf("- avx_u16\r");
        fflush(stdout);
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_AVX_u16<K,R>;
        auto config = get_soft16_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>(test, config);
        printf("o avx_u16 (%.3f)\n", result.bit_error_rate);
        results.push_back(result);
    }
}

template <size_t K, size_t R, typename decoder_t>
TestResult test_third_party(Test& test) {
    const size_t total_decode_bits = test.total_input_bytes*8;
    const int* poly = test.poly;
    const auto& x_in = test.x_in;
    auto& x_out = test.x_out;
    auto& samples = test.samples;
    using reg_t = uint32_t;
    auto encoder = ConvolutionalEncoder_ShiftRegister<reg_t>(K, R, poly);
    auto decoder = decoder_t(poly, test.total_transmit_bits);
    auto config = get_ka9q_offset_binary_config();
    auto y_out = std::vector<uint8_t>(test.total_output_symbols);
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
            sample.update_symbols_rate = y_out.size() / double(sample.update_symbols_ns) * 1e9;
        }
        {
            Timer t;
            decoder.chainback(x_out.data(), total_decode_bits);
            sample.chainback_bits_ns = t.get_delta();
            sample.chainback_bits_rate = total_decode_bits / double(sample.chainback_bits_ns) * 1e9;
        }
    }
    auto result = get_test_result(samples);
    const size_t total_bit_errors = get_total_bit_errors(x_in.data(), x_out.data(), x_in.size());
    const double bit_error_rate = double(total_bit_errors) / double(x_in.size()*8);
    result.bit_error_rate = bit_error_rate;
    return result;
}

template <size_t K, size_t R, typename decoder_t>
TestResult test_ka9q(Test& test) {
    printf("- kafq\r");
    fflush(stdout);
    const auto result = test_third_party<K,R,decoder_t>(test);
    printf("o kafq (%.3f)\n", result.bit_error_rate);
    return result;
}

template <size_t K, size_t R, typename decoder_t>
TestResult test_spiral(Test& test) {
    printf("- spiral\r");
    fflush(stdout);
    const auto result = test_third_party<K,R,decoder_t>(test);
    printf("o spiral (%.3f)\n", result.bit_error_rate);
    return result;
}

int main(int argc, char** argv) {
    #if _WIN32
    // get windows to print utf8 correctly to terminal
    SetConsoleOutputCP(CP_UTF8); 
    #endif

    std::vector<Test> tests;
    std::vector<test_results_t> test_results;
    if (1) {
        constexpr size_t K = 7;
        constexpr size_t R = 2;
        constexpr size_t total_input_bytes = 1024;
        constexpr size_t total_samples = 4096*4;
        auto test = init_test<K,R>(V27_POLY, total_input_bytes, total_samples);
        auto results = test_results_t();
        results.push_back(test_ka9q<K,R,ka9q_viterbi27>(test));
        results.push_back(test_spiral<K,R,spiral27_i>(test));
        test_ours<K,R>(test, results);
        tests.push_back(test);
        test_results.push_back(results);
        printf("\n");
    }
    if (1) {
        constexpr size_t K = 7;
        constexpr size_t R = 4;
        constexpr size_t total_input_bytes = 1024;
        constexpr size_t total_samples = 4096;
        auto test = init_test<K,R>(V47_POLY, total_input_bytes, total_samples);
        auto results = test_results_t();
        results.push_back(std::nullopt);
        results.push_back(test_spiral<K,R,spiral47_i>(test));
        test_ours<K,R>(test, results);
        tests.push_back(test);
        test_results.push_back(results);
        printf("\n");
    }
    if (1) {
        constexpr size_t K = 9;
        constexpr size_t R = 2;
        constexpr size_t total_input_bytes = 512;
        constexpr size_t total_samples = 4096;
        auto test = init_test<K,R>(V29_POLY, total_input_bytes, total_samples);
        auto results = test_results_t();
        results.push_back(test_ka9q<K,R,ka9q_viterbi29>(test));
        results.push_back(test_spiral<K,R,spiral29_i>(test));
        test_ours<K,R>(test, results);
        tests.push_back(test);
        test_results.push_back(results);
        printf("\n");
    }
    if (1) {
        constexpr size_t K = 9;
        constexpr size_t R = 4;
        constexpr size_t total_input_bytes = 512;
        constexpr size_t total_samples = 4096;
        auto test = init_test<K,R>(V49_POLY, total_input_bytes, total_samples);
        auto results = test_results_t();
        results.push_back(std::nullopt);
        results.push_back(test_spiral<K,R,spiral49_i>(test));
        test_ours<K,R>(test, results);
        tests.push_back(test);
        test_results.push_back(results);
        printf("\n");
    }
    if (1) {
        constexpr size_t K = 15;
        constexpr size_t R = 6;
        constexpr size_t total_input_bytes = 256;
        constexpr size_t total_samples = 256;
        auto test = init_test<K,R>(V615_POLY, total_input_bytes, total_samples);
        auto results = test_results_t();
        results.push_back(test_ka9q<K,R,ka9q_viterbi615>(test));
        results.push_back(test_spiral<K,R,spiral615_i>(test));
        test_ours<K,R>(test, results);
        tests.push_back(test);
        test_results.push_back(results);
        printf("\n");
    }
    if (1) {
        constexpr size_t K = 24;
        constexpr size_t R = 2;
        constexpr size_t total_input_bytes = 8;
        constexpr size_t total_samples = 8;
        auto test = init_test<K,R>(V224_POLY, total_input_bytes, total_samples);
        auto results = test_results_t();
        results.push_back(test_ka9q<K,R,ka9q_viterbi224>(test));
        results.push_back(std::nullopt);
        test_ours<K,R>(test, results);
        tests.push_back(test);
        test_results.push_back(results);
        printf("\n");
    }

    const size_t total_tests = tests.size();

    printf("[update (symbols/s)]\n");
    printf("| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |\n");
    printf("| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |\n");
    for (size_t i = 0; i < total_tests; i++) {
        const auto& test = tests[i];
        const auto& results = test_results[i];
        printf("| %zu | %zu |", test.K, test.R);
        for (const auto result: results) {
            if (result == std::nullopt) {
                printf(" --- |");
            } else {
                const double avg_update_rate = result->avg_update_rate;
                const double std_update_rate = result->std_update_rate;
                const auto si = get_si_notation(avg_update_rate);
                printf(" %.3g±%.2g%s |", avg_update_rate/si.scale, std_update_rate/si.scale, si.prefix);
            }
        }
        printf("\n");
    }
    printf("\n");

    printf("[chainback (bits/s)]\n");
    printf("| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |\n");
    printf("| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |\n");
    for (size_t i = 0; i < total_tests; i++) {
        const auto& test = tests[i];
        const auto& results = test_results[i];
        printf("| %zu | %zu |", test.K, test.R);
        for (const auto result: results) {
            if (result == std::nullopt) {
                printf(" --- |");
            } else {
                const double avg_chainback_rate = result->avg_chainback_rate;
                const double std_chainback_rate = result->std_chainback_rate;
                const auto si = get_si_notation(avg_chainback_rate);
                printf(" %.3g±%.2g%s |", avg_chainback_rate/si.scale, std_chainback_rate/si.scale, si.prefix);
            }
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
