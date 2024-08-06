#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include "spiral27.h"
#include "spiral29.h"
#include "spiral47.h"
#include "spiral49.h"
#include "spiral615.h"

// Match the interface for ka9q decoders to be like ours for testing
template <
    size_t _K, size_t _R,
    typename vRK,
    vRK* (*vRK_create)(const int*, int),
    int (*vRK_init)(vRK*,int),
    void (*vRK_update)(vRK*, uint8_t*,int),
    int (*vRK_chainback)(vRK*,uint8_t*,uint32_t, uint32_t),
    void (*vRK_delete)(vRK*)
>
class spiral_viterbi_interface {
public:
    static constexpr size_t K = _K;
    static constexpr size_t R = _R;
private:
    vRK* m_inner;
public:
    spiral_viterbi_interface(const int* poly, size_t transmit_bits): m_inner(vRK_create(poly, int(transmit_bits))) {}
    spiral_viterbi_interface(spiral_viterbi_interface& other) {
        m_inner = other.m_inner;
        other.m_inner = nullptr;
    }
    spiral_viterbi_interface& operator=(spiral_viterbi_interface& other) {
        if (m_inner != nullptr) vRK_delete(m_inner);
        m_inner = other.m_inner;
        other.m_inner = nullptr;
        return *this;
    }
    spiral_viterbi_interface(const spiral_viterbi_interface& other) = delete;
    spiral_viterbi_interface& operator=(const spiral_viterbi_interface& other) = delete;
    ~spiral_viterbi_interface() {
        if (m_inner != nullptr) vRK_delete(m_inner);
        m_inner = nullptr;
    }
    void reset() {
        vRK_init(m_inner, 0);
    }
    void update(uint8_t* sym, size_t total_syms) {
        assert(total_syms % _R == 0);
        const size_t total_bits = total_syms / _R;
        vRK_update(m_inner, sym, int(total_bits));
    }
    void chainback(uint8_t* data, size_t total_bits) {
        vRK_chainback(m_inner, data, uint32_t(total_bits), 0);
    }
};

using spiral27_i = spiral_viterbi_interface<7,2,spiral27,create_spiral27,init_spiral27,update_spiral27,chainback_spiral27, delete_spiral27>;
using spiral29_i = spiral_viterbi_interface<9,2,spiral29,create_spiral29,init_spiral29,update_spiral29,chainback_spiral29, delete_spiral29>;
using spiral47_i = spiral_viterbi_interface<7,4,spiral47,create_spiral47,init_spiral47,update_spiral47,chainback_spiral47, delete_spiral47>;
using spiral49_i = spiral_viterbi_interface<9,4,spiral49,create_spiral49,init_spiral49,update_spiral49,chainback_spiral49, delete_spiral49>;
using spiral615_i = spiral_viterbi_interface<15,6,spiral615,create_spiral615,init_spiral615,update_spiral615,chainback_spiral615, delete_spiral615>;
