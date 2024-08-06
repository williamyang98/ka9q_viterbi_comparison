#pragma once

struct v224;
struct v224 *create_viterbi224_sse2(const int *poly, int len);
int init_viterbi224_sse2(struct v224 *p, int starting_state);
int chainback_viterbi224_sse2(struct v224 *p, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_viterbi224_sse2(struct v224 *p);
void update_viterbi224_blk_sse2(struct v224 *p, unsigned char *syms, int nbits);
