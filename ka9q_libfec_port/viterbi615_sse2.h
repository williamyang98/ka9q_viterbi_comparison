#pragma once

struct v615;
struct v615 *create_viterbi615_sse2(const int *poly, int len);
int init_viterbi615_sse2(struct v615 *p, int starting_state);
int chainback_viterbi615_sse2(struct v615 *p, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_viterbi615_sse2(struct v615 *p);
void update_viterbi615_blk_sse2(struct v615 *p, unsigned char *syms, int nbits);
