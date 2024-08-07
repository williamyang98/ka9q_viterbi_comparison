#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include "./argparse.hpp"
#include "./ka9q_interface.h"
#include "./span.h"
#include "./spiral_interface.h"
#include "./timer.h"
#include "./util.h"
#include "./viterbi_configs.h"
#include "viterbi/convolutional_encoder_shift_register.h"
#include "viterbi/viterbi_branch_table.h"
#include "viterbi/viterbi_decoder_config.h"
#include "viterbi/viterbi_decoder_core.h"
#include "viterbi/x86/viterbi_decoder_avx_u16.h"
#include "viterbi/x86/viterbi_decoder_avx_u8.h"
#include "viterbi/x86/viterbi_decoder_sse_u16.h"
#include "viterbi/x86/viterbi_decoder_sse_u8.h"

static FILE* const fp_log = stderr;
static FILE* fp_out = stdout;

struct TestSample {
    uint64_t init_ns = 0;
    uint64_t update_symbols_ns = 0;
    uint64_t chainback_bits_ns = 0;
};

static std::vector<TestSample> samples;

struct Test {
    size_t K;
    size_t R;
    const int* poly;
    size_t total_transmit_bits;
    size_t total_input_bytes;
    size_t total_output_symbols;
    float sampling_time;
    size_t minimum_samples;
    std::vector<uint8_t> x_in;
    std::vector<uint8_t> x_out;
};

struct TestResult {
    double bit_error_rate = 0.0f;
};

template <typename T>
void print_array(tcb::span<const T> data, const char* fmt) {
    const size_t N = data.size();
    fprintf(fp_out, "[");
    for (size_t i = 0; i < N; i++) {
        fprintf(fp_out, fmt, data[i]);
        if (i != (N-1)) fprintf(fp_out, ",");
    }
    fprintf(fp_out, "]");
}

template <typename T, typename U>
void print_array(tcb::span<const T> data, const std::function<U(const T&)> transform, const char* fmt) {
    const size_t N = data.size();
    fprintf(fp_out, "[");
    for (size_t i = 0; i < N; i++) {
        fprintf(fp_out, fmt, transform(data[i]));
        if (i != (N-1)) fprintf(fp_out, ",");
    }
    fprintf(fp_out, "]");
}

TestResult print_test(const char* name, const Test& test) {
    fprintf(fp_out, "{\n");
    fprintf(fp_out, "  \"name\": \"%s\",\n", name);
    fprintf(fp_out, "  \"K\": %zu,\n", test.K);
    fprintf(fp_out, "  \"R\": %zu,\n", test.R);
    fprintf(fp_out, "  \"poly\": ");
    print_array<int>({ test.poly, test.R }, "%d");
    fprintf(fp_out, ",\n");

    fprintf(fp_out, "  \"total_input_bytes\": %zu,\n", test.total_input_bytes);
    fprintf(fp_out, "  \"total_transmit_bits\": %zu,\n", test.total_transmit_bits);
    fprintf(fp_out, "  \"total_output_symbols\": %zu,\n", test.total_output_symbols);
    fprintf(fp_out, "  \"sampling_time\": %f,\n", test.sampling_time);
    fprintf(fp_out, "  \"minimum_samples\": %zu,\n", test.minimum_samples);

    fprintf(fp_out, "  \"total_samples\": %zu,\n", samples.size());
    fprintf(fp_out, "  \"init_ns\": ");
    print_array<TestSample, uint64_t>(samples, [](const TestSample& sample) { return sample.init_ns; }, "%zu");
    fprintf(fp_out, ",\n");
    fprintf(fp_out, "  \"update_ns\": ");
    print_array<TestSample, uint64_t>(samples, [](const TestSample& sample) { return sample.update_symbols_ns; }, "%zu");
    fprintf(fp_out, ",\n");
    fprintf(fp_out, "  \"chainback_ns\": ");
    print_array<TestSample, uint64_t>(samples, [](const TestSample& sample) { return sample.chainback_bits_ns; }, "%zu");
    fprintf(fp_out, ",\n");

    const size_t total_bits = test.x_out.size()*8;
    const size_t total_bit_errors = get_total_bit_errors(test.x_in.data(), test.x_out.data(), test.x_in.size());
    const float bit_error_rate = float(total_bit_errors) / float(total_bits);
    fprintf(fp_out, "  \"total_bits\": %zu,\n", total_bits);
    fprintf(fp_out, "  \"total_bit_errors\": %zu,\n", total_bit_errors);
    fprintf(fp_out, "  \"bit_error_rate\": %f,\n", bit_error_rate);
    fprintf(fp_out, "},\n");
    return { bit_error_rate };
}

template <size_t K, size_t R>
Test init_test(const int* poly, const size_t total_decode_bytes, const float sampling_time, const size_t minimum_samples) {
    fprintf(fp_log, "[test_run]\n");
    fprintf(fp_log, "K=%zu, R=%zu\n", K, R);
    fprintf(fp_log, "total_input_bytes = %zu\n", total_decode_bytes);
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
    test.total_input_bytes = total_decode_bytes;
    test.total_output_symbols = total_symbols;
    test.total_transmit_bits = total_transmit_bits;
    test.sampling_time = sampling_time;
    test.minimum_samples = minimum_samples;
    test.x_in = x_in;
    test.x_out.resize(total_decode_bytes);
    return test;
}

// test ours
template <size_t K, size_t R, typename soft_t, typename error_t, class decoder_t>
TestResult test_ours_single(const char* name, Test& test, Decoder_Config<soft_t, error_t> config) {
    const size_t total_decode_bits = test.total_input_bytes*8;
    const int* poly = test.poly;
    const auto& x_in = test.x_in;
    auto& x_out = test.x_out;
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
    Timer total_time;
    samples.clear();
    for (size_t i = 0; ; i++) {
        const float elapsed_seconds = float(total_time.get_delta<std::chrono::milliseconds>())*1e-3f;
        if ((elapsed_seconds > test.sampling_time) && (i > test.minimum_samples)) break;
        TestSample sample;
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
        }
        {
            Timer t;
            core->chainback(x_out.data(), total_decode_bits);
            sample.chainback_bits_ns = t.get_delta();
        }
        samples.push_back(sample);
    }
    return print_test(name, test);
}

template <size_t K, size_t R>
void test_ours(Test& test) {
    {
        fprintf(fp_log, "- sse_u8\r");
        fflush(fp_log);
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_SSE_u8<K,R>;
        auto config = get_soft8_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>("sse_u8", test, config);
        fprintf(fp_log, "o sse_u8 (%.3f)\n", result.bit_error_rate);
    }
    {
        fprintf(fp_log, "- avx_u8\r");
        fflush(fp_log);
        using soft_t = int8_t;
        using error_t = uint8_t;
        using decoder = ViterbiDecoder_AVX_u8<K,R>;
        auto config = get_soft8_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>("avx_u8", test, config);
        fprintf(fp_log, "o avx_u8 (%.3f)\n", result.bit_error_rate);
    }
    {
        fprintf(fp_log, "- sse_u16\r");
        fflush(fp_log);
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_SSE_u16<K,R>;
        auto config = get_soft16_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>("sse_u16", test, config);
        fprintf(fp_log, "o sse_u16 (%.3f)\n", result.bit_error_rate);
    }
    {
        fprintf(fp_log, "- avx_u16\r");
        fflush(fp_log);
        using soft_t = int16_t;
        using error_t = uint16_t;
        using decoder = ViterbiDecoder_AVX_u16<K,R>;
        auto config = get_soft16_decoding_config(R);
        const auto result = test_ours_single<K,R,soft_t,error_t,decoder>("avx_u16", test, config);
        fprintf(fp_log, "o avx_u16 (%.3f)\n", result.bit_error_rate);
    }
}

template <size_t K, size_t R, typename decoder_t>
TestResult test_third_party(const char* name, Test& test) {
    const size_t total_decode_bits = test.total_input_bytes*8;
    const int* poly = test.poly;
    const auto& x_in = test.x_in;
    auto& x_out = test.x_out;
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
    Timer total_time;
    samples.clear();
    for (size_t i = 0; ; i++) {
        const float elapsed_seconds = float(total_time.get_delta<std::chrono::milliseconds>())*1e-3f;
        if ((elapsed_seconds > test.sampling_time) && (i > test.minimum_samples)) break;
        TestSample sample;
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
        samples.push_back(sample);
    }
    return print_test(name, test);
}

template <size_t K, size_t R, typename decoder_t>
void test_ka9q(Test& test) {
    fprintf(fp_log, "- kafq\r");
    fflush(fp_log);
    const auto result = test_third_party<K,R,decoder_t>("ka9q", test);
    fprintf(fp_log, "o kafq (%.3f)\n", result.bit_error_rate);
}

template <size_t K, size_t R, typename decoder_t>
void test_spiral(Test& test) {
    fprintf(fp_log, "- spiral\r");
    fflush(fp_log);
    const auto result = test_third_party<K,R,decoder_t>("spiral", test);
    fprintf(fp_log, "o spiral (%.3f)\n", result.bit_error_rate);
}

void init_parser(argparse::ArgumentParser& parser) {
    parser.add_argument("-t", "--sampling-time")
        .default_value(float(1.0f)).scan<'g', float>()
        .metavar("SAMPLING_TIME")
        .nargs(1).required()
        .help("Amount of time to run decoder");
    parser.add_argument("-n", "--minimum-samples")
        .default_value(size_t(8)).scan<'u', size_t>()
        .metavar("MINIMUM_SAMPLES")
        .nargs(1).required()
        .help("Minimum number of samples to accumulate");
    parser.add_argument("-o", "--output")
        .default_value(std::string("./data/benchmark.json"))
        .metavar("OUTPUT_FILENAME")
        .nargs(1).required()
        .help("Filename to output sample data (defaults to stdout)");
}

struct Args {
    float sampling_time;
    size_t minimum_samples;
    std::string output_filename;
};

Args get_args_from_parser(const argparse::ArgumentParser& parser) {
    Args args;
    args.sampling_time = parser.get<float>("--sampling-time");
    args.minimum_samples = parser.get<size_t>("--minimum-samples");
    args.output_filename = parser.get<std::string>("--output");
    return args;
}

int main(int argc, char** argv) {
    auto parser = argparse::ArgumentParser("run_benchmark", "0.1.0");
    parser.add_description("Run benchmark to compare ka9q, spiral and our Viterbi decoders");
    init_parser(parser);
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    const auto args = get_args_from_parser(parser);
    if (args.sampling_time <= 0.0f) {
        fprintf(stderr, "Sampling time must be positive (%.3f)\n", args.sampling_time);
        return 1;
    }
    if (args.minimum_samples == 0) {
        fprintf(stderr, "Minimum number of samples must be non-zero\n");
        return 1;
    }
    if (!args.output_filename.empty()) {
        fp_out = fopen(args.output_filename.c_str(), "w+");
        if (fp_out == nullptr) {
            fprintf(stderr, "Failed to open output file: '%s'\n", args.output_filename.c_str());
            return 1;
        }
    }
    samples.reserve(4096);
    samples.clear();

    fprintf(fp_out, "[\n");
    if (1) {
        constexpr size_t K = 7;
        constexpr size_t R = 2;
        constexpr size_t total_input_bytes = 1024;
        const int poly[2] = { 0x6d, 0x4f };
        auto test = init_test<K,R>(poly, total_input_bytes, args.sampling_time, args.minimum_samples);
        test_ka9q<K,R,ka9q_viterbi27>(test);
        test_spiral<K,R,spiral27_i>(test);
        test_ours<K,R>(test);
        fprintf(fp_log, "\n");
    }
    if (1) {
        constexpr size_t K = 7;
        constexpr size_t R = 4;
        constexpr size_t total_input_bytes = 1024;
        const int poly[4] = { 121, 117, 91, 111 };
        auto test = init_test<K,R>(poly, total_input_bytes, args.sampling_time, args.minimum_samples);
        test_spiral<K,R,spiral47_i>(test);
        test_ours<K,R>(test);
        fprintf(fp_log, "\n");
    }
    if (1) {
        constexpr size_t K = 9;
        constexpr size_t R = 2;
        constexpr size_t total_input_bytes = 512;
        const int poly[2] = { 0x1af, 0x11d };
        auto test = init_test<K,R>(poly, total_input_bytes, args.sampling_time, args.minimum_samples);
        test_ka9q<K,R,ka9q_viterbi29>(test);
        test_spiral<K,R,spiral29_i>(test);
        test_ours<K,R>(test);
        fprintf(fp_log, "\n");
    }
    if (1) {
        constexpr size_t K = 9;
        constexpr size_t R = 4;
        constexpr size_t total_input_bytes = 512;
        const int poly[4] = { 501, 441, 331, 315 };
        auto test = init_test<K,R>(poly, total_input_bytes, args.sampling_time, args.minimum_samples);
        test_spiral<K,R,spiral49_i>(test);
        test_ours<K,R>(test);
        fprintf(fp_log, "\n");
    }
    if (1) {
        constexpr size_t K = 15;
        constexpr size_t R = 6;
        constexpr size_t total_input_bytes = 256;
        const int poly[6] = { 042631, 047245, 056507, 073363, 077267, 064537 };
        auto test = init_test<K,R>(poly, total_input_bytes, args.sampling_time, args.minimum_samples);
        test_ka9q<K,R,ka9q_viterbi615>(test);
        test_spiral<K,R,spiral615_i>(test);
        test_ours<K,R>(test);
        fprintf(fp_log, "\n");
    }
    if (1) {
        constexpr size_t K = 24;
        constexpr size_t R = 2;
        constexpr size_t total_input_bytes = 8;
        const int poly[2] = { 062650457, 062650455 };
        auto test = init_test<K,R>(poly, total_input_bytes, args.sampling_time, args.minimum_samples);
        test_ka9q<K,R,ka9q_viterbi224>(test);
        test_ours<K,R>(test);
        fprintf(fp_log, "\n");
    }
    fprintf(fp_out, "]\n");
    return 0;
}
