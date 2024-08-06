#pragma once

#include "viterbi27_sse2.h"
#include "viterbi29_sse2.h"
#include "viterbi615_sse2.h"
#include "viterbi224_sse2.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

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
class ka9q_viterbi_interface {
public:
    static constexpr size_t K = _K;
    static constexpr size_t R = _R;
private:
    vRK* m_inner;
public:
    ka9q_viterbi_interface(const int* poly, size_t transmit_bits): m_inner(vRK_create(poly, int(transmit_bits))) {}
    ka9q_viterbi_interface(ka9q_viterbi_interface& other) {
        m_inner = other.m_inner;
        other.m_inner = nullptr;
    }
    ka9q_viterbi_interface& operator=(ka9q_viterbi_interface& other) {
        if (m_inner != nullptr) vRK_delete(m_inner);
        m_inner = other.m_inner;
        other.m_inner = nullptr;
        return *this;
    }
    ka9q_viterbi_interface(const ka9q_viterbi_interface& other) = delete;
    ka9q_viterbi_interface& operator=(const ka9q_viterbi_interface& other) = delete;
    ~ka9q_viterbi_interface() {
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

using ka9q_viterbi27 = ka9q_viterbi_interface<7,2,v27,create_viterbi27_sse2,init_viterbi27_sse2,update_viterbi27_blk_sse2,chainback_viterbi27_sse2,delete_viterbi27_sse2>;
using ka9q_viterbi29 = ka9q_viterbi_interface<9,2,v29,create_viterbi29_sse2,init_viterbi29_sse2,update_viterbi29_blk_sse2,chainback_viterbi29_sse2,delete_viterbi29_sse2>;
using ka9q_viterbi615 = ka9q_viterbi_interface<15,6,v615,create_viterbi615_sse2,init_viterbi615_sse2,update_viterbi615_blk_sse2,chainback_viterbi615_sse2,delete_viterbi615_sse2>;
using ka9q_viterbi224 = ka9q_viterbi_interface<24,2,v224,create_viterbi224_sse2,init_viterbi224_sse2,update_viterbi224_blk_sse2,chainback_viterbi224_sse2,delete_viterbi224_sse2>;
