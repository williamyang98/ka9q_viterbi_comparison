#pragma once

/* r=1/6 k=15 convolutional encoder polynomials */
static const int V615_POLY[6] = {
  042631,
  047245,
  056507,
  073363,
  077267,
  064537,
};

struct v615;
struct v615 *create_viterbi615_sse2(int len);
int init_viterbi615_sse2(struct v615 *p, int starting_state);
int chainback_viterbi615_sse2(struct v615 *p, unsigned char *data, unsigned int nbits, unsigned int endstate);
void delete_viterbi615_sse2(struct v615 *p);
void update_viterbi615_blk_sse2(struct v615 *p, unsigned char *syms, int nbits);
