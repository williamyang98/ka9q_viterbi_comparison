#include <stdlib.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <mmintrin.h>
#include "./spiral615.h"
#include "../src/parity.h"

#define K 15
#define RATE 6
#define NUMSTATES 16384
#define DECISIONTYPE unsigned char
#define DECISIONTYPE_BITSIZE 8
#define COMPUTETYPE unsigned char
#define METRICSHIFT 2
#define PRECISIONSHIFT 2
#define RENORMALIZE_THRESHOLD 166

//decision_t is a BIT vector
typedef union  {
  DECISIONTYPE t[NUMSTATES/DECISIONTYPE_BITSIZE];
  unsigned int w[NUMSTATES/32];
  unsigned short s[NUMSTATES/16];
  unsigned char c[NUMSTATES/8];
} decision_t;

typedef union  {
  COMPUTETYPE t[NUMSTATES];
} metric_t;

inline void renormalize(COMPUTETYPE* X, COMPUTETYPE threshold){
    if (X[0]>threshold){
        COMPUTETYPE min=X[0];
        for(int i=0;i<NUMSTATES;i++) {
            if (min>X[i]) min=X[i];
        }
        for(int i=0;i<NUMSTATES;i++)
            X[i]-=min;
    }
}

 static COMPUTETYPE Branchtab[NUMSTATES/2*RATE];

/* State info for instance of Viterbi decoder
 */
struct spiral615 {
   metric_t metrics1; /* path metric buffer 1 */
   metric_t metrics2; /* path metric buffer 2 */
  metric_t *old_metrics,*new_metrics; /* Pointers to path metrics, swapped on every bit */
  decision_t *decisions;   /* decisions */
};

/* Initialize Viterbi decoder for start of new frame */
int init_spiral615(spiral615 *vp,int starting_state){
  int i;
  for(i=0;i<NUMSTATES;i++) vp->metrics1.t[i] = 63;
  vp->old_metrics = &vp->metrics1;
  vp->new_metrics = &vp->metrics2;
  vp->old_metrics->t[starting_state & (NUMSTATES-1)] = 0; /* Bias known start state */
  return 0;
}

/* Create a new instance of a Viterbi decoder */
spiral615 *create_spiral615(const int* poly, int len){
  static int Init = 0;

  if(!Init){
    int state, i;
    const auto& parity = ParityTable::get();
    for(state=0;state < NUMSTATES/2;state++){
      for (i=0; i<RATE; i++){
        Branchtab[i*NUMSTATES/2+state] = (poly[i] < 0) ^ parity.parse((2*state) & abs(poly[i])) ? 255 : 0;
      }
    }
    Init++;
  }

  spiral615 *vp = (spiral615*)malloc(sizeof(struct spiral615));
  vp->decisions = (decision_t*)malloc((len+(K-1))*sizeof(decision_t));
  init_spiral615(vp,0);
  return vp;
}

/* Viterbi chainback */
int chainback_spiral615(
      spiral615 *vp,
      unsigned char *data, /* Decoded output data */
      unsigned int nbits, /* Number of data bits */
      unsigned int endstate){ /* Terminal encoder state */
  decision_t *d;

  /* ADDSHIFT and SUBSHIFT make sure that the thing returned is a byte. */
#if (K-1<8)
#define ADDSHIFT (8-(K-1))
#define SUBSHIFT 0
#elif (K-1>8)
#define ADDSHIFT 0
#define SUBSHIFT ((K-1)-8)
#else
#define ADDSHIFT 0
#define SUBSHIFT 0
#endif
  d = vp->decisions;
  /* Make room beyond the end of the encoder register so we can
   * accumulate a full byte of decoded data
   */

  endstate = (endstate%NUMSTATES) << ADDSHIFT;

  /* The store into data[] only needs to be done every 8 bits.
   * But this avoids a conditional branch, and the writes will
   * combine in the cache anyway
   */
  d += (K-1); /* Look past tail */
  while(nbits-- != 0){
    int k;
    k = (d[nbits].w[(endstate>>ADDSHIFT)/32] >> ((endstate>>ADDSHIFT)%32)) & 1;
    endstate = (endstate >> 1) | (k << (K-2+ADDSHIFT));
    data[nbits>>3] = endstate>>SUBSHIFT;
  }
  return 0;
}

/* Delete instance of a Viterbi decoder */
void delete_spiral615(spiral615 *vp){
    free(vp->decisions);
    free(vp);
}

static void FULL_SPIRAL(unsigned char  *Y, unsigned char  *X, unsigned char  *syms, unsigned char  *dec, unsigned char  *Branchtab, int N) {
    for(int i9 = 0; i9 < N; i9++) {
        for(int i1 = 0; i1 <= 511; i1++) {
            unsigned char a101, a112, a122, a132, a142, a152;
            int a167, a168, a99;
            short int s12, s13;
            unsigned char  *a100, *a111, *a121, *a131, *a141, *a151, *b20;
            short int  *a166, *a169, *a170, *b25;
            __m128i  *a103, *a104, *a114, *a124, *a134, *a144, *a154
                    , *a171, *a172, *a173, *a96, *a97, *a98;
            __m128i a107, a108, a117, a118, a127, a128, a137
                    , a138, a147, a148, a157, a158, a161, a162;
            __m128i a102, a105, a106, a109, a110, a113, a115
                    , a116, a119, a120, a123, a125, a126, a129, a130
                    , a133, a135, a136, a139, a140, a143, a145, a146
                    , a149, a150, a153, a155, a156, a159, a160, a163
                    , a164, a165, b21, b22, b23, b24, d7, d8
                    , m13, m14, m15, m16, s10, s11, s14, s15
                    , t10, t11, t12;
            a96 = ((__m128i  *) X);
            a97 = (a96 + i1);
            s10 = *(a97);
            a98 = (a97 + 512);
            s11 = *(a98);
            a99 = (12 * i9);
            a100 = (syms + a99);
            a101 = *(a100);
            a102 = _mm_set1_epi8(a101);
            a103 = ((__m128i  *) Branchtab);
            a104 = (a103 + i1);
            a105 = *(a104);
            a106 = _mm_xor_si128(a102, a105);
            a107 = ((__m128i ) a106);
            a108 = _mm_srli_epi16(a107, 2);
            a109 = ((__m128i ) a108);
            a110 = _mm_and_si128(a109, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            b20 = (a99 + syms);
            a111 = (b20 + 1);
            a112 = *(a111);
            a113 = _mm_set1_epi8(a112);
            a114 = (a104 + 512);
            a115 = *(a114);
            a116 = _mm_xor_si128(a113, a115);
            a117 = ((__m128i ) a116);
            a118 = _mm_srli_epi16(a117, 2);
            a119 = ((__m128i ) a118);
            a120 = _mm_and_si128(a119, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a121 = (b20 + 2);
            a122 = *(a121);
            a123 = _mm_set1_epi8(a122);
            a124 = (a104 + 1024);
            a125 = *(a124);
            a126 = _mm_xor_si128(a123, a125);
            a127 = ((__m128i ) a126);
            a128 = _mm_srli_epi16(a127, 2);
            a129 = ((__m128i ) a128);
            a130 = _mm_and_si128(a129, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a131 = (b20 + 3);
            a132 = *(a131);
            a133 = _mm_set1_epi8(a132);
            a134 = (a104 + 1536);
            a135 = *(a134);
            a136 = _mm_xor_si128(a133, a135);
            a137 = ((__m128i ) a136);
            a138 = _mm_srli_epi16(a137, 2);
            a139 = ((__m128i ) a138);
            a140 = _mm_and_si128(a139, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a141 = (b20 + 4);
            a142 = *(a141);
            a143 = _mm_set1_epi8(a142);
            a144 = (a104 + 2048);
            a145 = *(a144);
            a146 = _mm_xor_si128(a143, a145);
            a147 = ((__m128i ) a146);
            a148 = _mm_srli_epi16(a147, 2);
            a149 = ((__m128i ) a148);
            a150 = _mm_and_si128(a149, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a151 = (b20 + 5);
            a152 = *(a151);
            a153 = _mm_set1_epi8(a152);
            a154 = (a104 + 2560);
            a155 = *(a154);
            a156 = _mm_xor_si128(a153, a155);
            a157 = ((__m128i ) a156);
            a158 = _mm_srli_epi16(a157, 2);
            a159 = ((__m128i ) a158);
            a160 = _mm_and_si128(a159, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            b21 = _mm_adds_epu8(a110, a120);
            b22 = _mm_adds_epu8(b21, a130);
            b23 = _mm_adds_epu8(b22, a140);
            b24 = _mm_adds_epu8(b23, a150);
            t10 = _mm_adds_epu8(b24, a160);
            a161 = ((__m128i ) t10);
            a162 = _mm_srli_epi16(a161, 2);
            a163 = ((__m128i ) a162);
            t11 = _mm_and_si128(a163, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            t12 = _mm_subs_epu8(_mm_set_epi8(94, 94, 94, 94, 94, 94, 94
    , 94, 94, 94, 94, 94, 94, 94, 94
    , 94), t11);
            m13 = _mm_adds_epu8(s10, t11);
            m14 = _mm_adds_epu8(s11, t12);
            m15 = _mm_adds_epu8(s10, t12);
            m16 = _mm_adds_epu8(s11, t11);
            a164 = _mm_min_epu8(m14, m13);
            d7 = _mm_cmpeq_epi8(a164, m14);
            a165 = _mm_min_epu8(m16, m15);
            d8 = _mm_cmpeq_epi8(a165, m16);
            s12 = _mm_movemask_epi8(_mm_unpacklo_epi8(d7,d8));
            a166 = ((short int  *) dec);
            a167 = (2 * i1);
            a168 = (2048 * i9);
            b25 = (a166 + a168);
            a169 = (b25 + a167);
            *(a169) = s12;
            s13 = _mm_movemask_epi8(_mm_unpackhi_epi8(d7,d8));
            a170 = (a169 + 1);
            *(a170) = s13;
            s14 = _mm_unpacklo_epi8(a164, a165);
            s15 = _mm_unpackhi_epi8(a164, a165);
            a171 = ((__m128i  *) Y);
            a172 = (a171 + a167);
            *(a172) = s14;
            a173 = (a172 + 1);
            *(a173) = s15;
        }
        renormalize(Y, 74);
        for(int i1 = 0; i1 <= 511; i1++) {
            unsigned char a274, a285, a295, a305, a315, a325;
            int a272, a340, a341;
            short int s26, s27;
            unsigned char  *a273, *a284, *a294, *a304, *a314, *a324, *b47;
            short int  *a339, *a342, *a343, *b52, *b53;
            __m128i  *a269, *a270, *a271, *a276, *a277, *a287, *a297
                    , *a307, *a317, *a327, *a344, *a345, *a346;
            __m128i a280, a281, a290, a291, a300, a301, a310
                    , a311, a320, a321, a330, a331, a334, a335;
            __m128i a275, a278, a279, a282, a283, a286, a288
                    , a289, a292, a293, a296, a298, a299, a302, a303
                    , a306, a308, a309, a312, a313, a316, a318, a319
                    , a322, a323, a326, a328, a329, a332, a333, a336
                    , a337, a338, b48, b49, b50, b51, d11, d12
                    , m21, m22, m23, m24, s24, s25, s28, s29
                    , t16, t17, t18;
            a269 = ((__m128i  *) Y);
            a270 = (a269 + i1);
            s24 = *(a270);
            a271 = (a270 + 512);
            s25 = *(a271);
            a272 = (12 * i9);
            b47 = (a272 + syms);
            a273 = (b47 + 6);
            a274 = *(a273);
            a275 = _mm_set1_epi8(a274);
            a276 = ((__m128i  *) Branchtab);
            a277 = (a276 + i1);
            a278 = *(a277);
            a279 = _mm_xor_si128(a275, a278);
            a280 = ((__m128i ) a279);
            a281 = _mm_srli_epi16(a280, 2);
            a282 = ((__m128i ) a281);
            a283 = _mm_and_si128(a282, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a284 = (b47 + 7);
            a285 = *(a284);
            a286 = _mm_set1_epi8(a285);
            a287 = (a277 + 512);
            a288 = *(a287);
            a289 = _mm_xor_si128(a286, a288);
            a290 = ((__m128i ) a289);
            a291 = _mm_srli_epi16(a290, 2);
            a292 = ((__m128i ) a291);
            a293 = _mm_and_si128(a292, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a294 = (b47 + 8);
            a295 = *(a294);
            a296 = _mm_set1_epi8(a295);
            a297 = (a277 + 1024);
            a298 = *(a297);
            a299 = _mm_xor_si128(a296, a298);
            a300 = ((__m128i ) a299);
            a301 = _mm_srli_epi16(a300, 2);
            a302 = ((__m128i ) a301);
            a303 = _mm_and_si128(a302, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a304 = (b47 + 9);
            a305 = *(a304);
            a306 = _mm_set1_epi8(a305);
            a307 = (a277 + 1536);
            a308 = *(a307);
            a309 = _mm_xor_si128(a306, a308);
            a310 = ((__m128i ) a309);
            a311 = _mm_srli_epi16(a310, 2);
            a312 = ((__m128i ) a311);
            a313 = _mm_and_si128(a312, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a314 = (b47 + 10);
            a315 = *(a314);
            a316 = _mm_set1_epi8(a315);
            a317 = (a277 + 2048);
            a318 = *(a317);
            a319 = _mm_xor_si128(a316, a318);
            a320 = ((__m128i ) a319);
            a321 = _mm_srli_epi16(a320, 2);
            a322 = ((__m128i ) a321);
            a323 = _mm_and_si128(a322, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            a324 = (b47 + 11);
            a325 = *(a324);
            a326 = _mm_set1_epi8(a325);
            a327 = (a277 + 2560);
            a328 = *(a327);
            a329 = _mm_xor_si128(a326, a328);
            a330 = ((__m128i ) a329);
            a331 = _mm_srli_epi16(a330, 2);
            a332 = ((__m128i ) a331);
            a333 = _mm_and_si128(a332, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            b48 = _mm_adds_epu8(a283, a293);
            b49 = _mm_adds_epu8(b48, a303);
            b50 = _mm_adds_epu8(b49, a313);
            b51 = _mm_adds_epu8(b50, a323);
            t16 = _mm_adds_epu8(b51, a333);
            a334 = ((__m128i ) t16);
            a335 = _mm_srli_epi16(a334, 2);
            a336 = ((__m128i ) a335);
            t17 = _mm_and_si128(a336, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
            t18 = _mm_subs_epu8(_mm_set_epi8(94, 94, 94, 94, 94, 94, 94
    , 94, 94, 94, 94, 94, 94, 94, 94
    , 94), t17);
            m21 = _mm_adds_epu8(s24, t17);
            m22 = _mm_adds_epu8(s25, t18);
            m23 = _mm_adds_epu8(s24, t18);
            m24 = _mm_adds_epu8(s25, t17);
            a337 = _mm_min_epu8(m22, m21);
            d11 = _mm_cmpeq_epi8(a337, m22);
            a338 = _mm_min_epu8(m24, m23);
            d12 = _mm_cmpeq_epi8(a338, m24);
            s26 = _mm_movemask_epi8(_mm_unpacklo_epi8(d11,d12));
            a339 = ((short int  *) dec);
            a340 = (2 * i1);
            a341 = (2048 * i9);
            b52 = (a339 + a341);
            b53 = (b52 + a340);
            a342 = (b53 + 1024);
            *(a342) = s26;
            s27 = _mm_movemask_epi8(_mm_unpackhi_epi8(d11,d12));
            a343 = (b53 + 1025);
            *(a343) = s27;
            s28 = _mm_unpacklo_epi8(a337, a338);
            s29 = _mm_unpackhi_epi8(a337, a338);
            a344 = ((__m128i  *) X);
            a345 = (a344 + a340);
            *(a345) = s28;
            a346 = (a345 + 1);
            *(a346) = s29;
        }
        renormalize(X, 74);
    }
    /* skip */
}

void update_spiral615(spiral615 *vp, COMPUTETYPE *syms, int nbits) {
  FULL_SPIRAL(vp->new_metrics->t, vp->old_metrics->t, syms, vp->decisions->c, Branchtab, nbits/2);
}
