#pragma once

// Massey r=1/2 k=24 optimally truncatable code
static const int V224_POLY[2] = { 062650457, 062650455 };

struct v224;
struct v224 *create_viterbi224_sse2(int len);
int init_viterbi224_sse2(struct v224 *p, int starting_state);
int chainback_viterbi224_sse2(struct v224 *p, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_viterbi224_sse2(struct v224 *p);
void update_viterbi224_blk_sse2(struct v224 *p, unsigned char *syms, int nbits);
