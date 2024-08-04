#pragma once

/* r=1/2 k=9 convolutional encoder polynomials */
static const int V29_POLY[2] = { 0x1af, 0x11d };

struct v29;
struct v29 *create_viterbi29_sse2(int len);
int init_viterbi29_sse2(struct v29 *p, int starting_state);
int chainback_viterbi29_sse2(struct v29 *p, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_viterbi29_sse2(struct v29 *p);
void update_viterbi29_blk_sse2(struct v29 *p, unsigned char *syms, int nbits);
