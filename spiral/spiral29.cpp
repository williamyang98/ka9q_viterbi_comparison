#include <stdlib.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <mmintrin.h>
#include "./spiral29.h"
#include "../src/parity.h"

#define K 9
#define RATE 2
#define NUMSTATES 256
#define DECISIONTYPE unsigned char
#define DECISIONTYPE_BITSIZE 8
#define COMPUTETYPE unsigned char
#define METRICSHIFT 1
#define PRECISIONSHIFT 2
#define RENORMALIZE_THRESHOLD 168

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
    if (X[0]>threshold) {
        COMPUTETYPE min=X[0];
        for(int i=0;i<NUMSTATES;i++) {
            if (min>X[i]) min=X[i];
        }
        for(int i=0;i<NUMSTATES;i++) {
          X[i]-=min;
        }
    }
}

  static COMPUTETYPE Branchtab[NUMSTATES/2*RATE];

/* State info for instance of Viterbi decoder
 */
struct spiral29 {
   metric_t metrics1; /* path metric buffer 1 */
   metric_t metrics2; /* path metric buffer 2 */
  metric_t *old_metrics,*new_metrics; /* Pointers to path metrics, swapped on every bit */
  decision_t *decisions;   /* decisions */
};

/* Initialize Viterbi decoder for start of new frame */
int init_spiral29(spiral29 *vp,int starting_state){
  int i;
  for(i=0;i<NUMSTATES;i++)
      vp->metrics1.t[i] = 63;

  vp->old_metrics = &vp->metrics1;
  vp->new_metrics = &vp->metrics2;
  vp->old_metrics->t[starting_state & (NUMSTATES-1)] = 0; /* Bias known start state */
  return 0;
}

/* Create a new instance of a Viterbi decoder */
spiral29 *create_spiral29(const int *poly, int len){
  struct spiral29 *vp;
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
  vp = (spiral29*)malloc(sizeof(struct spiral29));
  vp->decisions = (decision_t*)malloc((len+(K-1))*sizeof(decision_t));
  init_spiral29(vp,0);
  return vp;
}

/* Viterbi chainback */
int chainback_spiral29(
      spiral29 *vp,
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
void delete_spiral29(spiral29 *vp){
  if(vp != NULL){
    free(vp->decisions);
    free(vp);
  }
}

static void FULL_SPIRAL(unsigned char  *Y, unsigned char  *X, unsigned char  *syms, unsigned char  *dec, unsigned char  *Branchtab, int N) {
    for(int i9 = 0; i9 < N; i9++) {
        unsigned char a285, a291;
        int a283, a302;
        short int s104, s105, s110, s111, s68, s69, s74
    , s75, s80, s81, s86, s87, s92, s93, s98
    , s99;
        unsigned char  *a284, *a290, *b24;
        short int  *a301, *a303, *a304, *a320, *a321, *a337, *a338
                , *a354, *a355, *a371, *a372, *a389, *a390, *a406, *a407
                , *a423, *a424;
        __m128i  *a281, *a282, *a287, *a293, *a305, *a306, *a307
                , *a308, *a309, *a312, *a322, *a323, *a324, *a325, *a326
                , *a329, *a339, *a340, *a341, *a342, *a343, *a346, *a356
                , *a357, *a358, *a359, *a360, *a363, *a373, *a374, *a375
                , *a376, *a377, *a380, *a391, *a392, *a393, *a394, *a395
                , *a398, *a408, *a409, *a410, *a411, *a412, *a415, *a425
                , *a426;
        __m128i a296, a297, a315, a316, a332, a333, a349
                , a350, a366, a367, a383, a384, a401, a402, a418
                , a419;
        __m128i a286, a288, a289, a292, a294, a295, a298
                , a299, a300, a310, a311, a313, a314, a317, a318
                , a319, a327, a328, a330, a331, a334, a335, a336
                , a344, a345, a347, a348, a351, a352, a353, a361
                , a362, a364, a365, a368, a369, a370, a378, a379
                , a381, a382, a385, a387, a388, a396, a397, a399
                , a400, a403, a404, a405, a413, a414, a416, a417
                , a420, a421, a422, d21, d22, d23, d24, d25
                , d26, d27, d28, d29, d30, d31, d32, d33
                , d34, d35, d36, m47, m48, m49, m50, m51
                , m52, m53, m54, m55, m56, m57, m58, m59
                , m60, m61, m62, m63, m64, m65, m66, m67
                , m68, m69, m70, m71, m72, m73, m74, m75
                , m76, m77, m78, s100, s101, s102, s103, s106
                , s107, s108, s109, s112, s113, s66, s67, s70
                , s71, s72, s73, s76, s77, s78, s79, s82
                , s83, s84, s85, s88, s89, s90, s91, s94
                , s95, s96, s97, t31, t32, t33, t34, t35
                , t36, t37, t38, t39, t40, t41, t42, t43
                , t44, t45, t46, t47, t48, t49, t50, t51
                , t52, t53, t54;
        a281 = ((__m128i  *) X);
        s66 = *(a281);
        a282 = (a281 + 8);
        s67 = *(a282);
        a283 = (4 * i9);
        a284 = (syms + a283);
        a285 = *(a284);
        a286 = _mm_set1_epi8(a285);
        a287 = ((__m128i  *) Branchtab);
        a288 = *(a287);
        a289 = _mm_xor_si128(a286, a288);
        b24 = (a283 + syms);
        a290 = (b24 + 1);
        a291 = *(a290);
        a292 = _mm_set1_epi8(a291);
        a293 = (a287 + 8);
        a294 = *(a293);
        a295 = _mm_xor_si128(a292, a294);
        t31 = _mm_avg_epu8(a289,a295);
        a296 = ((__m128i ) t31);
        a297 = _mm_srli_epi16(a296, 2);
        a298 = ((__m128i ) a297);
        t32 = _mm_and_si128(a298, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t33 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t32);
        m47 = _mm_adds_epu8(s66, t32);
        m48 = _mm_adds_epu8(s67, t33);
        m49 = _mm_adds_epu8(s66, t33);
        m50 = _mm_adds_epu8(s67, t32);
        a299 = _mm_min_epu8(m48, m47);
        d21 = _mm_cmpeq_epi8(a299, m48);
        a300 = _mm_min_epu8(m50, m49);
        d22 = _mm_cmpeq_epi8(a300, m50);
        s68 = _mm_movemask_epi8(_mm_unpacklo_epi8(d21,d22));
        a301 = ((short int  *) dec);
        a302 = (32 * i9);
        a303 = (a301 + a302);
        *(a303) = s68;
        s69 = _mm_movemask_epi8(_mm_unpackhi_epi8(d21,d22));
        a304 = (a303 + 1);
        *(a304) = s69;
        s70 = _mm_unpacklo_epi8(a299, a300);
        s71 = _mm_unpackhi_epi8(a299, a300);
        a305 = ((__m128i  *) Y);
        *(a305) = s70;
        a306 = (a305 + 1);
        *(a306) = s71;
        a307 = (a281 + 1);
        s72 = *(a307);
        a308 = (a281 + 9);
        s73 = *(a308);
        a309 = (a287 + 1);
        a310 = *(a309);
        a311 = _mm_xor_si128(a286, a310);
        a312 = (a287 + 9);
        a313 = *(a312);
        a314 = _mm_xor_si128(a292, a313);
        t34 = _mm_avg_epu8(a311,a314);
        a315 = ((__m128i ) t34);
        a316 = _mm_srli_epi16(a315, 2);
        a317 = ((__m128i ) a316);
        t35 = _mm_and_si128(a317, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t36 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t35);
        m51 = _mm_adds_epu8(s72, t35);
        m52 = _mm_adds_epu8(s73, t36);
        m53 = _mm_adds_epu8(s72, t36);
        m54 = _mm_adds_epu8(s73, t35);
        a318 = _mm_min_epu8(m52, m51);
        d23 = _mm_cmpeq_epi8(a318, m52);
        a319 = _mm_min_epu8(m54, m53);
        d24 = _mm_cmpeq_epi8(a319, m54);
        s74 = _mm_movemask_epi8(_mm_unpacklo_epi8(d23,d24));
        a320 = (a303 + 2);
        *(a320) = s74;
        s75 = _mm_movemask_epi8(_mm_unpackhi_epi8(d23,d24));
        a321 = (a303 + 3);
        *(a321) = s75;
        s76 = _mm_unpacklo_epi8(a318, a319);
        s77 = _mm_unpackhi_epi8(a318, a319);
        a322 = (a305 + 2);
        *(a322) = s76;
        a323 = (a305 + 3);
        *(a323) = s77;
        a324 = (a281 + 2);
        s78 = *(a324);
        a325 = (a281 + 10);
        s79 = *(a325);
        a326 = (a287 + 2);
        a327 = *(a326);
        a328 = _mm_xor_si128(a286, a327);
        a329 = (a287 + 10);
        a330 = *(a329);
        a331 = _mm_xor_si128(a292, a330);
        t37 = _mm_avg_epu8(a328,a331);
        a332 = ((__m128i ) t37);
        a333 = _mm_srli_epi16(a332, 2);
        a334 = ((__m128i ) a333);
        t38 = _mm_and_si128(a334, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t39 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t38);
        m55 = _mm_adds_epu8(s78, t38);
        m56 = _mm_adds_epu8(s79, t39);
        m57 = _mm_adds_epu8(s78, t39);
        m58 = _mm_adds_epu8(s79, t38);
        a335 = _mm_min_epu8(m56, m55);
        d25 = _mm_cmpeq_epi8(a335, m56);
        a336 = _mm_min_epu8(m58, m57);
        d26 = _mm_cmpeq_epi8(a336, m58);
        s80 = _mm_movemask_epi8(_mm_unpacklo_epi8(d25,d26));
        a337 = (a303 + 4);
        *(a337) = s80;
        s81 = _mm_movemask_epi8(_mm_unpackhi_epi8(d25,d26));
        a338 = (a303 + 5);
        *(a338) = s81;
        s82 = _mm_unpacklo_epi8(a335, a336);
        s83 = _mm_unpackhi_epi8(a335, a336);
        a339 = (a305 + 4);
        *(a339) = s82;
        a340 = (a305 + 5);
        *(a340) = s83;
        a341 = (a281 + 3);
        s84 = *(a341);
        a342 = (a281 + 11);
        s85 = *(a342);
        a343 = (a287 + 3);
        a344 = *(a343);
        a345 = _mm_xor_si128(a286, a344);
        a346 = (a287 + 11);
        a347 = *(a346);
        a348 = _mm_xor_si128(a292, a347);
        t40 = _mm_avg_epu8(a345,a348);
        a349 = ((__m128i ) t40);
        a350 = _mm_srli_epi16(a349, 2);
        a351 = ((__m128i ) a350);
        t41 = _mm_and_si128(a351, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t42 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t41);
        m59 = _mm_adds_epu8(s84, t41);
        m60 = _mm_adds_epu8(s85, t42);
        m61 = _mm_adds_epu8(s84, t42);
        m62 = _mm_adds_epu8(s85, t41);
        a352 = _mm_min_epu8(m60, m59);
        d27 = _mm_cmpeq_epi8(a352, m60);
        a353 = _mm_min_epu8(m62, m61);
        d28 = _mm_cmpeq_epi8(a353, m62);
        s86 = _mm_movemask_epi8(_mm_unpacklo_epi8(d27,d28));
        a354 = (a303 + 6);
        *(a354) = s86;
        s87 = _mm_movemask_epi8(_mm_unpackhi_epi8(d27,d28));
        a355 = (a303 + 7);
        *(a355) = s87;
        s88 = _mm_unpacklo_epi8(a352, a353);
        s89 = _mm_unpackhi_epi8(a352, a353);
        a356 = (a305 + 6);
        *(a356) = s88;
        a357 = (a305 + 7);
        *(a357) = s89;
        a358 = (a281 + 4);
        s90 = *(a358);
        a359 = (a281 + 12);
        s91 = *(a359);
        a360 = (a287 + 4);
        a361 = *(a360);
        a362 = _mm_xor_si128(a286, a361);
        a363 = (a287 + 12);
        a364 = *(a363);
        a365 = _mm_xor_si128(a292, a364);
        t43 = _mm_avg_epu8(a362,a365);
        a366 = ((__m128i ) t43);
        a367 = _mm_srli_epi16(a366, 2);
        a368 = ((__m128i ) a367);
        t44 = _mm_and_si128(a368, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t45 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t44);
        m63 = _mm_adds_epu8(s90, t44);
        m64 = _mm_adds_epu8(s91, t45);
        m65 = _mm_adds_epu8(s90, t45);
        m66 = _mm_adds_epu8(s91, t44);
        a369 = _mm_min_epu8(m64, m63);
        d29 = _mm_cmpeq_epi8(a369, m64);
        a370 = _mm_min_epu8(m66, m65);
        d30 = _mm_cmpeq_epi8(a370, m66);
        s92 = _mm_movemask_epi8(_mm_unpacklo_epi8(d29,d30));
        a371 = (a303 + 8);
        *(a371) = s92;
        s93 = _mm_movemask_epi8(_mm_unpackhi_epi8(d29,d30));
        a372 = (a303 + 9);
        *(a372) = s93;
        s94 = _mm_unpacklo_epi8(a369, a370);
        s95 = _mm_unpackhi_epi8(a369, a370);
        a373 = (a305 + 8);
        *(a373) = s94;
        a374 = (a305 + 9);
        *(a374) = s95;
        a375 = (a281 + 5);
        s96 = *(a375);
        a376 = (a281 + 13);
        s97 = *(a376);
        a377 = (a287 + 5);
        a378 = *(a377);
        a379 = _mm_xor_si128(a286, a378);
        a380 = (a287 + 13);
        a381 = *(a380);
        a382 = _mm_xor_si128(a292, a381);
        t46 = _mm_avg_epu8(a379,a382);
        a383 = ((__m128i ) t46);
        a384 = _mm_srli_epi16(a383, 2);
        a385 = ((__m128i ) a384);
        t47 = _mm_and_si128(a385, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t48 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t47);
        m67 = _mm_adds_epu8(s96, t47);
        m68 = _mm_adds_epu8(s97, t48);
        m69 = _mm_adds_epu8(s96, t48);
        m70 = _mm_adds_epu8(s97, t47);
        a387 = _mm_min_epu8(m68, m67);
        d31 = _mm_cmpeq_epi8(a387, m68);
        a388 = _mm_min_epu8(m70, m69);
        d32 = _mm_cmpeq_epi8(a388, m70);
        s98 = _mm_movemask_epi8(_mm_unpacklo_epi8(d31,d32));
        a389 = (a303 + 10);
        *(a389) = s98;
        s99 = _mm_movemask_epi8(_mm_unpackhi_epi8(d31,d32));
        a390 = (a303 + 11);
        *(a390) = s99;
        s100 = _mm_unpacklo_epi8(a387, a388);
        s101 = _mm_unpackhi_epi8(a387, a388);
        a391 = (a305 + 10);
        *(a391) = s100;
        a392 = (a305 + 11);
        *(a392) = s101;
        a393 = (a281 + 6);
        s102 = *(a393);
        a394 = (a281 + 14);
        s103 = *(a394);
        a395 = (a287 + 6);
        a396 = *(a395);
        a397 = _mm_xor_si128(a286, a396);
        a398 = (a287 + 14);
        a399 = *(a398);
        a400 = _mm_xor_si128(a292, a399);
        t49 = _mm_avg_epu8(a397,a400);
        a401 = ((__m128i ) t49);
        a402 = _mm_srli_epi16(a401, 2);
        a403 = ((__m128i ) a402);
        t50 = _mm_and_si128(a403, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t51 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t50);
        m71 = _mm_adds_epu8(s102, t50);
        m72 = _mm_adds_epu8(s103, t51);
        m73 = _mm_adds_epu8(s102, t51);
        m74 = _mm_adds_epu8(s103, t50);
        a404 = _mm_min_epu8(m72, m71);
        d33 = _mm_cmpeq_epi8(a404, m72);
        a405 = _mm_min_epu8(m74, m73);
        d34 = _mm_cmpeq_epi8(a405, m74);
        s104 = _mm_movemask_epi8(_mm_unpacklo_epi8(d33,d34));
        a406 = (a303 + 12);
        *(a406) = s104;
        s105 = _mm_movemask_epi8(_mm_unpackhi_epi8(d33,d34));
        a407 = (a303 + 13);
        *(a407) = s105;
        s106 = _mm_unpacklo_epi8(a404, a405);
        s107 = _mm_unpackhi_epi8(a404, a405);
        a408 = (a305 + 12);
        *(a408) = s106;
        a409 = (a305 + 13);
        *(a409) = s107;
        a410 = (a281 + 7);
        s108 = *(a410);
        a411 = (a281 + 15);
        s109 = *(a411);
        a412 = (a287 + 7);
        a413 = *(a412);
        a414 = _mm_xor_si128(a286, a413);
        a415 = (a287 + 15);
        a416 = *(a415);
        a417 = _mm_xor_si128(a292, a416);
        t52 = _mm_avg_epu8(a414,a417);
        a418 = ((__m128i ) t52);
        a419 = _mm_srli_epi16(a418, 2);
        a420 = ((__m128i ) a419);
        t53 = _mm_and_si128(a420, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t54 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t53);
        m75 = _mm_adds_epu8(s108, t53);
        m76 = _mm_adds_epu8(s109, t54);
        m77 = _mm_adds_epu8(s108, t54);
        m78 = _mm_adds_epu8(s109, t53);
        a421 = _mm_min_epu8(m76, m75);
        d35 = _mm_cmpeq_epi8(a421, m76);
        a422 = _mm_min_epu8(m78, m77);
        d36 = _mm_cmpeq_epi8(a422, m78);
        s110 = _mm_movemask_epi8(_mm_unpacklo_epi8(d35,d36));
        a423 = (a303 + 14);
        *(a423) = s110;
        s111 = _mm_movemask_epi8(_mm_unpackhi_epi8(d35,d36));
        a424 = (a303 + 15);
        *(a424) = s111;
        s112 = _mm_unpacklo_epi8(a421, a422);
        s113 = _mm_unpackhi_epi8(a421, a422);
        a425 = (a305 + 14);
        *(a425) = s112;
        a426 = (a305 + 15);
        *(a426) = s113;
        if ((((unsigned char  *) Y)[0]>210)) {
            __m128i m5, m6;
            m5 = ((__m128i  *) Y)[0];
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[1]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[2]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[3]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[4]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[5]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[6]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[7]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[8]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[9]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[10]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[11]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[12]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[13]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[14]);
            m5 = _mm_min_epu8(m5, ((__m128i  *) Y)[15]);
            __m128i m7;
            m7 = _mm_min_epu8(_mm_srli_si128(m5, 8), m5);
            m7 = ((__m128i ) _mm_min_epu8(((__m128i ) _mm_srli_epi64(m7, 32)), ((__m128i ) m7)));
            m7 = ((__m128i ) _mm_min_epu8(((__m128i ) _mm_srli_epi64(m7, 16)), ((__m128i ) m7)));
            m7 = ((__m128i ) _mm_min_epu8(((__m128i ) _mm_srli_epi64(m7, 8)), ((__m128i ) m7)));
            m7 = _mm_unpacklo_epi8(m7, m7);
            m7 = _mm_shufflelo_epi16(m7, _MM_SHUFFLE(0, 0, 0, 0));
            m6 = _mm_unpacklo_epi64(m7, m7);
            ((__m128i  *) Y)[0] = _mm_subs_epu8(((__m128i  *) Y)[0], m6);
            ((__m128i  *) Y)[1] = _mm_subs_epu8(((__m128i  *) Y)[1], m6);
            ((__m128i  *) Y)[2] = _mm_subs_epu8(((__m128i  *) Y)[2], m6);
            ((__m128i  *) Y)[3] = _mm_subs_epu8(((__m128i  *) Y)[3], m6);
            ((__m128i  *) Y)[4] = _mm_subs_epu8(((__m128i  *) Y)[4], m6);
            ((__m128i  *) Y)[5] = _mm_subs_epu8(((__m128i  *) Y)[5], m6);
            ((__m128i  *) Y)[6] = _mm_subs_epu8(((__m128i  *) Y)[6], m6);
            ((__m128i  *) Y)[7] = _mm_subs_epu8(((__m128i  *) Y)[7], m6);
            ((__m128i  *) Y)[8] = _mm_subs_epu8(((__m128i  *) Y)[8], m6);
            ((__m128i  *) Y)[9] = _mm_subs_epu8(((__m128i  *) Y)[9], m6);
            ((__m128i  *) Y)[10] = _mm_subs_epu8(((__m128i  *) Y)[10], m6);
            ((__m128i  *) Y)[11] = _mm_subs_epu8(((__m128i  *) Y)[11], m6);
            ((__m128i  *) Y)[12] = _mm_subs_epu8(((__m128i  *) Y)[12], m6);
            ((__m128i  *) Y)[13] = _mm_subs_epu8(((__m128i  *) Y)[13], m6);
            ((__m128i  *) Y)[14] = _mm_subs_epu8(((__m128i  *) Y)[14], m6);
            ((__m128i  *) Y)[15] = _mm_subs_epu8(((__m128i  *) Y)[15], m6);
        }
        unsigned char a711, a717;
        int a709, a728;
        short int s180, s181, s186, s187, s192, s193, s198
    , s199, s204, s205, s210, s211, s216, s217, s222
    , s223;
        unsigned char  *a710, *a716, *b57;
        short int  *a727, *a729, *a730, *a746, *a747, *a763, *a764
                , *a780, *a781, *a797, *a798, *a814, *a815, *a831, *a832
                , *a848, *a849, *b58;
        __m128i  *a707, *a708, *a713, *a719, *a731, *a732, *a733
                , *a734, *a735, *a738, *a748, *a749, *a750, *a751, *a752
                , *a755, *a765, *a766, *a767, *a768, *a769, *a772, *a782
                , *a783, *a784, *a785, *a786, *a789, *a799, *a800, *a801
                , *a802, *a803, *a806, *a816, *a817, *a818, *a819, *a820
                , *a823, *a833, *a834, *a835, *a836, *a837, *a840, *a850
                , *a851;
        __m128i a722, a723, a741, a742, a758, a759, a775
                , a776, a792, a793, a809, a810, a826, a827, a843
                , a844;
        __m128i a712, a714, a715, a718, a720, a721, a724
                , a725, a726, a736, a737, a739, a740, a743, a744
                , a745, a753, a754, a756, a757, a760, a761, a762
                , a770, a771, a773, a774, a777, a778, a779, a787
                , a788, a790, a791, a794, a795, a796, a804, a805
                , a807, a808, a811, a812, a813, a821, a822, a824
                , a825, a828, a829, a830, a838, a839, a841, a842
                , a845, a846, a847, d53, d54, d55, d56, d57
                , d58, d59, d60, d61, d62, d63, d64, d65
                , d66, d67, d68, m111, m112, m113, m114, m115
                , m116, m117, m118, m119, m120, m121, m122, m123
                , m124, m125, m126, m127, m128, m129, m130, m131
                , m132, m133, m134, m135, m136, m137, m138, m139
                , m140, m141, m142, s178, s179, s182, s183, s184
                , s185, s188, s189, s190, s191, s194, s195, s196
                , s197, s200, s201, s202, s203, s206, s207, s208
                , s209, s212, s213, s214, s215, s218, s219, s220
                , s221, s224, s225, t100, t101, t102, t79, t80
                , t81, t82, t83, t84, t85, t86, t87, t88
                , t89, t90, t91, t92, t93, t94, t95, t96
                , t97, t98, t99;
        a707 = ((__m128i  *) Y);
        s178 = *(a707);
        a708 = (a707 + 8);
        s179 = *(a708);
        a709 = (4 * i9);
        b57 = (a709 + syms);
        a710 = (b57 + 2);
        a711 = *(a710);
        a712 = _mm_set1_epi8(a711);
        a713 = ((__m128i  *) Branchtab);
        a714 = *(a713);
        a715 = _mm_xor_si128(a712, a714);
        a716 = (b57 + 3);
        a717 = *(a716);
        a718 = _mm_set1_epi8(a717);
        a719 = (a713 + 8);
        a720 = *(a719);
        a721 = _mm_xor_si128(a718, a720);
        t79 = _mm_avg_epu8(a715,a721);
        a722 = ((__m128i ) t79);
        a723 = _mm_srli_epi16(a722, 2);
        a724 = ((__m128i ) a723);
        t80 = _mm_and_si128(a724, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t81 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t80);
        m111 = _mm_adds_epu8(s178, t80);
        m112 = _mm_adds_epu8(s179, t81);
        m113 = _mm_adds_epu8(s178, t81);
        m114 = _mm_adds_epu8(s179, t80);
        a725 = _mm_min_epu8(m112, m111);
        d53 = _mm_cmpeq_epi8(a725, m112);
        a726 = _mm_min_epu8(m114, m113);
        d54 = _mm_cmpeq_epi8(a726, m114);
        s180 = _mm_movemask_epi8(_mm_unpacklo_epi8(d53,d54));
        a727 = ((short int  *) dec);
        a728 = (32 * i9);
        b58 = (a727 + a728);
        a729 = (b58 + 16);
        *(a729) = s180;
        s181 = _mm_movemask_epi8(_mm_unpackhi_epi8(d53,d54));
        a730 = (b58 + 17);
        *(a730) = s181;
        s182 = _mm_unpacklo_epi8(a725, a726);
        s183 = _mm_unpackhi_epi8(a725, a726);
        a731 = ((__m128i  *) X);
        *(a731) = s182;
        a732 = (a731 + 1);
        *(a732) = s183;
        a733 = (a707 + 1);
        s184 = *(a733);
        a734 = (a707 + 9);
        s185 = *(a734);
        a735 = (a713 + 1);
        a736 = *(a735);
        a737 = _mm_xor_si128(a712, a736);
        a738 = (a713 + 9);
        a739 = *(a738);
        a740 = _mm_xor_si128(a718, a739);
        t82 = _mm_avg_epu8(a737,a740);
        a741 = ((__m128i ) t82);
        a742 = _mm_srli_epi16(a741, 2);
        a743 = ((__m128i ) a742);
        t83 = _mm_and_si128(a743, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t84 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t83);
        m115 = _mm_adds_epu8(s184, t83);
        m116 = _mm_adds_epu8(s185, t84);
        m117 = _mm_adds_epu8(s184, t84);
        m118 = _mm_adds_epu8(s185, t83);
        a744 = _mm_min_epu8(m116, m115);
        d55 = _mm_cmpeq_epi8(a744, m116);
        a745 = _mm_min_epu8(m118, m117);
        d56 = _mm_cmpeq_epi8(a745, m118);
        s186 = _mm_movemask_epi8(_mm_unpacklo_epi8(d55,d56));
        a746 = (b58 + 18);
        *(a746) = s186;
        s187 = _mm_movemask_epi8(_mm_unpackhi_epi8(d55,d56));
        a747 = (b58 + 19);
        *(a747) = s187;
        s188 = _mm_unpacklo_epi8(a744, a745);
        s189 = _mm_unpackhi_epi8(a744, a745);
        a748 = (a731 + 2);
        *(a748) = s188;
        a749 = (a731 + 3);
        *(a749) = s189;
        a750 = (a707 + 2);
        s190 = *(a750);
        a751 = (a707 + 10);
        s191 = *(a751);
        a752 = (a713 + 2);
        a753 = *(a752);
        a754 = _mm_xor_si128(a712, a753);
        a755 = (a713 + 10);
        a756 = *(a755);
        a757 = _mm_xor_si128(a718, a756);
        t85 = _mm_avg_epu8(a754,a757);
        a758 = ((__m128i ) t85);
        a759 = _mm_srli_epi16(a758, 2);
        a760 = ((__m128i ) a759);
        t86 = _mm_and_si128(a760, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t87 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t86);
        m119 = _mm_adds_epu8(s190, t86);
        m120 = _mm_adds_epu8(s191, t87);
        m121 = _mm_adds_epu8(s190, t87);
        m122 = _mm_adds_epu8(s191, t86);
        a761 = _mm_min_epu8(m120, m119);
        d57 = _mm_cmpeq_epi8(a761, m120);
        a762 = _mm_min_epu8(m122, m121);
        d58 = _mm_cmpeq_epi8(a762, m122);
        s192 = _mm_movemask_epi8(_mm_unpacklo_epi8(d57,d58));
        a763 = (b58 + 20);
        *(a763) = s192;
        s193 = _mm_movemask_epi8(_mm_unpackhi_epi8(d57,d58));
        a764 = (b58 + 21);
        *(a764) = s193;
        s194 = _mm_unpacklo_epi8(a761, a762);
        s195 = _mm_unpackhi_epi8(a761, a762);
        a765 = (a731 + 4);
        *(a765) = s194;
        a766 = (a731 + 5);
        *(a766) = s195;
        a767 = (a707 + 3);
        s196 = *(a767);
        a768 = (a707 + 11);
        s197 = *(a768);
        a769 = (a713 + 3);
        a770 = *(a769);
        a771 = _mm_xor_si128(a712, a770);
        a772 = (a713 + 11);
        a773 = *(a772);
        a774 = _mm_xor_si128(a718, a773);
        t88 = _mm_avg_epu8(a771,a774);
        a775 = ((__m128i ) t88);
        a776 = _mm_srli_epi16(a775, 2);
        a777 = ((__m128i ) a776);
        t89 = _mm_and_si128(a777, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t90 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t89);
        m123 = _mm_adds_epu8(s196, t89);
        m124 = _mm_adds_epu8(s197, t90);
        m125 = _mm_adds_epu8(s196, t90);
        m126 = _mm_adds_epu8(s197, t89);
        a778 = _mm_min_epu8(m124, m123);
        d59 = _mm_cmpeq_epi8(a778, m124);
        a779 = _mm_min_epu8(m126, m125);
        d60 = _mm_cmpeq_epi8(a779, m126);
        s198 = _mm_movemask_epi8(_mm_unpacklo_epi8(d59,d60));
        a780 = (b58 + 22);
        *(a780) = s198;
        s199 = _mm_movemask_epi8(_mm_unpackhi_epi8(d59,d60));
        a781 = (b58 + 23);
        *(a781) = s199;
        s200 = _mm_unpacklo_epi8(a778, a779);
        s201 = _mm_unpackhi_epi8(a778, a779);
        a782 = (a731 + 6);
        *(a782) = s200;
        a783 = (a731 + 7);
        *(a783) = s201;
        a784 = (a707 + 4);
        s202 = *(a784);
        a785 = (a707 + 12);
        s203 = *(a785);
        a786 = (a713 + 4);
        a787 = *(a786);
        a788 = _mm_xor_si128(a712, a787);
        a789 = (a713 + 12);
        a790 = *(a789);
        a791 = _mm_xor_si128(a718, a790);
        t91 = _mm_avg_epu8(a788,a791);
        a792 = ((__m128i ) t91);
        a793 = _mm_srli_epi16(a792, 2);
        a794 = ((__m128i ) a793);
        t92 = _mm_and_si128(a794, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t93 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t92);
        m127 = _mm_adds_epu8(s202, t92);
        m128 = _mm_adds_epu8(s203, t93);
        m129 = _mm_adds_epu8(s202, t93);
        m130 = _mm_adds_epu8(s203, t92);
        a795 = _mm_min_epu8(m128, m127);
        d61 = _mm_cmpeq_epi8(a795, m128);
        a796 = _mm_min_epu8(m130, m129);
        d62 = _mm_cmpeq_epi8(a796, m130);
        s204 = _mm_movemask_epi8(_mm_unpacklo_epi8(d61,d62));
        a797 = (b58 + 24);
        *(a797) = s204;
        s205 = _mm_movemask_epi8(_mm_unpackhi_epi8(d61,d62));
        a798 = (b58 + 25);
        *(a798) = s205;
        s206 = _mm_unpacklo_epi8(a795, a796);
        s207 = _mm_unpackhi_epi8(a795, a796);
        a799 = (a731 + 8);
        *(a799) = s206;
        a800 = (a731 + 9);
        *(a800) = s207;
        a801 = (a707 + 5);
        s208 = *(a801);
        a802 = (a707 + 13);
        s209 = *(a802);
        a803 = (a713 + 5);
        a804 = *(a803);
        a805 = _mm_xor_si128(a712, a804);
        a806 = (a713 + 13);
        a807 = *(a806);
        a808 = _mm_xor_si128(a718, a807);
        t94 = _mm_avg_epu8(a805,a808);
        a809 = ((__m128i ) t94);
        a810 = _mm_srli_epi16(a809, 2);
        a811 = ((__m128i ) a810);
        t95 = _mm_and_si128(a811, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t96 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t95);
        m131 = _mm_adds_epu8(s208, t95);
        m132 = _mm_adds_epu8(s209, t96);
        m133 = _mm_adds_epu8(s208, t96);
        m134 = _mm_adds_epu8(s209, t95);
        a812 = _mm_min_epu8(m132, m131);
        d63 = _mm_cmpeq_epi8(a812, m132);
        a813 = _mm_min_epu8(m134, m133);
        d64 = _mm_cmpeq_epi8(a813, m134);
        s210 = _mm_movemask_epi8(_mm_unpacklo_epi8(d63,d64));
        a814 = (b58 + 26);
        *(a814) = s210;
        s211 = _mm_movemask_epi8(_mm_unpackhi_epi8(d63,d64));
        a815 = (b58 + 27);
        *(a815) = s211;
        s212 = _mm_unpacklo_epi8(a812, a813);
        s213 = _mm_unpackhi_epi8(a812, a813);
        a816 = (a731 + 10);
        *(a816) = s212;
        a817 = (a731 + 11);
        *(a817) = s213;
        a818 = (a707 + 6);
        s214 = *(a818);
        a819 = (a707 + 14);
        s215 = *(a819);
        a820 = (a713 + 6);
        a821 = *(a820);
        a822 = _mm_xor_si128(a712, a821);
        a823 = (a713 + 14);
        a824 = *(a823);
        a825 = _mm_xor_si128(a718, a824);
        t97 = _mm_avg_epu8(a822,a825);
        a826 = ((__m128i ) t97);
        a827 = _mm_srli_epi16(a826, 2);
        a828 = ((__m128i ) a827);
        t98 = _mm_and_si128(a828, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t99 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t98);
        m135 = _mm_adds_epu8(s214, t98);
        m136 = _mm_adds_epu8(s215, t99);
        m137 = _mm_adds_epu8(s214, t99);
        m138 = _mm_adds_epu8(s215, t98);
        a829 = _mm_min_epu8(m136, m135);
        d65 = _mm_cmpeq_epi8(a829, m136);
        a830 = _mm_min_epu8(m138, m137);
        d66 = _mm_cmpeq_epi8(a830, m138);
        s216 = _mm_movemask_epi8(_mm_unpacklo_epi8(d65,d66));
        a831 = (b58 + 28);
        *(a831) = s216;
        s217 = _mm_movemask_epi8(_mm_unpackhi_epi8(d65,d66));
        a832 = (b58 + 29);
        *(a832) = s217;
        s218 = _mm_unpacklo_epi8(a829, a830);
        s219 = _mm_unpackhi_epi8(a829, a830);
        a833 = (a731 + 12);
        *(a833) = s218;
        a834 = (a731 + 13);
        *(a834) = s219;
        a835 = (a707 + 7);
        s220 = *(a835);
        a836 = (a707 + 15);
        s221 = *(a836);
        a837 = (a713 + 7);
        a838 = *(a837);
        a839 = _mm_xor_si128(a712, a838);
        a840 = (a713 + 15);
        a841 = *(a840);
        a842 = _mm_xor_si128(a718, a841);
        t100 = _mm_avg_epu8(a839,a842);
        a843 = ((__m128i ) t100);
        a844 = _mm_srli_epi16(a843, 2);
        a845 = ((__m128i ) a844);
        t101 = _mm_and_si128(a845, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t102 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t101);
        m139 = _mm_adds_epu8(s220, t101);
        m140 = _mm_adds_epu8(s221, t102);
        m141 = _mm_adds_epu8(s220, t102);
        m142 = _mm_adds_epu8(s221, t101);
        a846 = _mm_min_epu8(m140, m139);
        d67 = _mm_cmpeq_epi8(a846, m140);
        a847 = _mm_min_epu8(m142, m141);
        d68 = _mm_cmpeq_epi8(a847, m142);
        s222 = _mm_movemask_epi8(_mm_unpacklo_epi8(d67,d68));
        a848 = (b58 + 30);
        *(a848) = s222;
        s223 = _mm_movemask_epi8(_mm_unpackhi_epi8(d67,d68));
        a849 = (b58 + 31);
        *(a849) = s223;
        s224 = _mm_unpacklo_epi8(a846, a847);
        s225 = _mm_unpackhi_epi8(a846, a847);
        a850 = (a731 + 14);
        *(a850) = s224;
        a851 = (a731 + 15);
        *(a851) = s225;
        if ((((unsigned char  *) X)[0]>210)) {
            __m128i m12, m13;
            m12 = ((__m128i  *) X)[0];
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[1]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[2]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[3]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[4]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[5]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[6]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[7]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[8]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[9]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[10]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[11]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[12]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[13]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[14]);
            m12 = _mm_min_epu8(m12, ((__m128i  *) X)[15]);
            __m128i m14;
            m14 = _mm_min_epu8(_mm_srli_si128(m12, 8), m12);
            m14 = ((__m128i ) _mm_min_epu8(((__m128i ) _mm_srli_epi64(m14, 32)), ((__m128i ) m14)));
            m14 = ((__m128i ) _mm_min_epu8(((__m128i ) _mm_srli_epi64(m14, 16)), ((__m128i ) m14)));
            m14 = ((__m128i ) _mm_min_epu8(((__m128i ) _mm_srli_epi64(m14, 8)), ((__m128i ) m14)));
            m14 = _mm_unpacklo_epi8(m14, m14);
            m14 = _mm_shufflelo_epi16(m14, _MM_SHUFFLE(0, 0, 0, 0));
            m13 = _mm_unpacklo_epi64(m14, m14);
            ((__m128i  *) X)[0] = _mm_subs_epu8(((__m128i  *) X)[0], m13);
            ((__m128i  *) X)[1] = _mm_subs_epu8(((__m128i  *) X)[1], m13);
            ((__m128i  *) X)[2] = _mm_subs_epu8(((__m128i  *) X)[2], m13);
            ((__m128i  *) X)[3] = _mm_subs_epu8(((__m128i  *) X)[3], m13);
            ((__m128i  *) X)[4] = _mm_subs_epu8(((__m128i  *) X)[4], m13);
            ((__m128i  *) X)[5] = _mm_subs_epu8(((__m128i  *) X)[5], m13);
            ((__m128i  *) X)[6] = _mm_subs_epu8(((__m128i  *) X)[6], m13);
            ((__m128i  *) X)[7] = _mm_subs_epu8(((__m128i  *) X)[7], m13);
            ((__m128i  *) X)[8] = _mm_subs_epu8(((__m128i  *) X)[8], m13);
            ((__m128i  *) X)[9] = _mm_subs_epu8(((__m128i  *) X)[9], m13);
            ((__m128i  *) X)[10] = _mm_subs_epu8(((__m128i  *) X)[10], m13);
            ((__m128i  *) X)[11] = _mm_subs_epu8(((__m128i  *) X)[11], m13);
            ((__m128i  *) X)[12] = _mm_subs_epu8(((__m128i  *) X)[12], m13);
            ((__m128i  *) X)[13] = _mm_subs_epu8(((__m128i  *) X)[13], m13);
            ((__m128i  *) X)[14] = _mm_subs_epu8(((__m128i  *) X)[14], m13);
            ((__m128i  *) X)[15] = _mm_subs_epu8(((__m128i  *) X)[15], m13);
        }
    }
    /* skip */
}

void update_spiral29(spiral29 *vp, COMPUTETYPE *syms, int nbits) {
  FULL_SPIRAL(vp->new_metrics->t, vp->old_metrics->t, syms, vp->decisions->c, Branchtab, nbits/2);
}
