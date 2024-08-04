#pragma once

/* r=1/2 k=7 convolutional encoder polynomials */
static const int V27_POLY[2] = { 0x6d, 0x4f };

struct v27;
struct v27 *create_viterbi27_sse2(int len);
int init_viterbi27_sse2(struct v27 *p, int starting_state);
int chainback_viterbi27_sse2(struct v27 *p, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_viterbi27_sse2(struct v27 *p);
void update_viterbi27_blk_sse2(struct v27 *p, unsigned char *syms, int nbits);
