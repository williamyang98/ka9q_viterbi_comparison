#pragma once
#include <stdint.h>
#include <stddef.h>
#include <type_traits>

template <typename soft_t, typename poly_t>
struct Tableless_Config {
    soft_t soft_decision_high;
    soft_t soft_decision_low;
    const poly_t* poly;
};


template <size_t K, typename E = void> struct GetRegisterType;
template <size_t K> struct GetRegisterType<K, typename std::enable_if<0 < K && K <= 8>::type> { using type = uint8_t; };
template <size_t K> struct GetRegisterType<K, typename std::enable_if<8 < K && K <= 16>::type> { using type = uint16_t; };
template <size_t K> struct GetRegisterType<K, typename std::enable_if<16 < K && K <= 32>::type> { using type = uint32_t; };
// template <size_t K> struct GetRegisterType<K, typename std::enable_if<K <= 64>::type> { using type = uint64_t; };
template <size_t K> using get_register_type = typename GetRegisterType<K>::type;
