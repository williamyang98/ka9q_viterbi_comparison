#include <pmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <mmintrin.h>
#include <math.h>
#include "./spiral49.h"
#include "../src/parity.h"

#define K 9
#define RATE 4
#define NUMSTATES 256
#define DECISIONTYPE unsigned char
#define DECISIONTYPE_BITSIZE 8
#define COMPUTETYPE unsigned char
#define METRICSHIFT 2
#define PRECISIONSHIFT 2
#define RENORMALIZE_THRESHOLD 137

//decision_t is a BIT vector
typedef union {
  DECISIONTYPE t[NUMSTATES/DECISIONTYPE_BITSIZE];
  unsigned int w[NUMSTATES/32];
  unsigned short s[NUMSTATES/16];
  unsigned char c[NUMSTATES/8];
} decision_t;

typedef union {
  COMPUTETYPE t[NUMSTATES];
} metric_t;

inline void renormalize(COMPUTETYPE* X, COMPUTETYPE threshold){
    if (X[0]>threshold){
        COMPUTETYPE min=X[0];
        for(int i=0;i<NUMSTATES;i++) {
            if (min>X[i])min=X[i];
        }
        for(int i=0;i<NUMSTATES;i++) X[i]-=min;
    }
}

static COMPUTETYPE Branchtab[NUMSTATES/2*RATE];

/* State info for instance of Viterbi decoder
 */
struct spiral49 {
  metric_t metrics1; /* path metric buffer 1 */
  metric_t metrics2; /* path metric buffer 2 */
  metric_t *old_metrics,*new_metrics; /* Pointers to path metrics, swapped on every bit */
  decision_t *decisions;   /* decisions */
};

/* Initialize Viterbi decoder for start of new frame */
int init_spiral49(spiral49 *vp,int starting_state){
  int i;
  for(i=0;i<NUMSTATES;i++) vp->metrics1.t[i] = 63;
  vp->old_metrics = &vp->metrics1;
  vp->new_metrics = &vp->metrics2;
  vp->old_metrics->t[starting_state & (NUMSTATES-1)] = 0; /* Bias known start state */
  return 0;
}

/* Create a new instance of a Viterbi decoder */
spiral49 *create_spiral49(const int *poly, int len){
  struct spiral49 *vp;
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

  vp = (spiral49*)malloc(sizeof(struct spiral49));
  vp->decisions = (decision_t*)malloc((len+(K-1))*sizeof(decision_t));
  init_spiral49(vp,0);
  return vp;
}

/* Viterbi chainback */
int chainback_spiral49(
      spiral49 *vp,
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
void delete_spiral49(spiral49 *vp){
  if(vp != NULL){
    free(vp->decisions);
    free(vp);
  }
}

static void FULL_SPIRAL(unsigned char  *Y, unsigned char  *X, unsigned char  *syms, unsigned char  *dec, unsigned char  *Branchtab, int N) {
    for(int i9 = 0; i9 < N; i9++) {
        unsigned char a542, a552, a562, a572;
        int a540, a587;
        short int s104, s105, s110, s111, s68, s69, s74
    , s75, s80, s81, s86, s87, s92, s93, s98
    , s99;
        unsigned char  *a541, *a551, *a561, *a571, *b56;
        short int  *a586, *a588, *a589, *a627, *a628, *a666, *a667
                , *a705, *a706, *a744, *a745, *a783, *a784, *a822, *a823
                , *a861, *a862;
        __m128i  *a538, *a539, *a544, *a554, *a564, *a574, *a590
                , *a591, *a592, *a593, *a594, *a601, *a608, *a615, *a629
                , *a630, *a631, *a632, *a633, *a640, *a647, *a654, *a668
                , *a669, *a670, *a671, *a672, *a679, *a686, *a693, *a707
                , *a708, *a709, *a710, *a711, *a718, *a725, *a732, *a746
                , *a747, *a748, *a749, *a750, *a757, *a764, *a771, *a785
                , *a786, *a787, *a788, *a789, *a796, *a803, *a810, *a824
                , *a825, *a826, *a827, *a828, *a835, *a842, *a849, *a863
                , *a864;
        __m128i a547, a548, a557, a558, a567, a568, a577
                , a578, a581, a582, a597, a598, a604, a605, a611
                , a612, a618, a619, a622, a623, a636, a637, a643
                , a644, a650, a651, a657, a658, a661, a662, a675
                , a676, a682, a683, a689, a690, a696, a697, a700
                , a701, a714, a715, a721, a722, a728, a729, a735
                , a736, a739, a740, a753, a754, a760, a761, a767
                , a768, a774, a775, a778, a779, a792, a793, a799
                , a800, a806, a807, a813, a814, a817, a818, a831
                , a832, a838, a839, a845, a846, a852, a853, a856
                , a857;
        __m128i a543, a545, a546, a549, a550, a553, a555
                , a556, a559, a560, a563, a565, a566, a569, a570
                , a573, a575, a576, a579, a580, a583, a584, a585
                , a595, a596, a599, a600, a602, a603, a606, a607
                , a609, a610, a613, a614, a616, a617, a620, a621
                , a624, a625, a626, a634, a635, a638, a639, a641
                , a642, a645, a646, a648, a649, a652, a653, a655
                , a656, a659, a660, a663, a664, a665, a673, a674
                , a677, a678, a680, a681, a684, a685, a687, a688
                , a691, a692, a694, a695, a698, a699, a702, a703
                , a704, a712, a713, a716, a717, a719, a720, a723
                , a724, a726, a727, a730, a731, a733, a734, a737
                , a738, a741, a742, a743, a751, a752, a755, a756
                , a758, a759, a762, a763, a765, a766, a769, a770
                , a772, a773, a776, a777, a780, a781, a782, a790
                , a791, a794, a795, a797, a798, a801, a802, a804
                , a805, a808, a809, a811, a812, a815, a816, a819
                , a820, a821, a829, a830, a833, a834, a836, a837
                , a840, a841, a843, a844, a847, a848, a850, a851
                , a854, a855, a858, a859, a860, b57, b58, b59
                , b60, b61, b62, b63, b64, b65, b66, b67
                , b68, b69, b70, b71, b72, d21, d22, d23
                , d24, d25, d26, d27, d28, d29, d30, d31
                , d32, d33, d34, d35, d36, m47, m48, m49
                , m50, m51, m52, m53, m54, m55, m56, m57
                , m58, m59, m60, m61, m62, m63, m64, m65
                , m66, m67, m68, m69, m70, m71, m72, m73
                , m74, m75, m76, m77, m78, s100, s101, s102
                , s103, s106, s107, s108, s109, s112, s113, s66
                , s67, s70, s71, s72, s73, s76, s77, s78
                , s79, s82, s83, s84, s85, s88, s89, s90
                , s91, s94, s95, s96, s97, t31, t32, t33
                , t34, t35, t36, t37, t38, t39, t40, t41
                , t42, t43, t44, t45, t46, t47, t48, t49
                , t50, t51, t52, t53, t54;
        a538 = ((__m128i  *) X);
        s66 = *(a538);
        a539 = (a538 + 8);
        s67 = *(a539);
        a540 = (8 * i9);
        a541 = (syms + a540);
        a542 = *(a541);
        a543 = _mm_set1_epi8(a542);
        a544 = ((__m128i  *) Branchtab);
        a545 = *(a544);
        a546 = _mm_xor_si128(a543, a545);
        a547 = ((__m128i ) a546);
        a548 = _mm_srli_epi16(a547, 2);
        a549 = ((__m128i ) a548);
        a550 = _mm_and_si128(a549, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b56 = (a540 + syms);
        a551 = (b56 + 1);
        a552 = *(a551);
        a553 = _mm_set1_epi8(a552);
        a554 = (a544 + 8);
        a555 = *(a554);
        a556 = _mm_xor_si128(a553, a555);
        a557 = ((__m128i ) a556);
        a558 = _mm_srli_epi16(a557, 2);
        a559 = ((__m128i ) a558);
        a560 = _mm_and_si128(a559, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a561 = (b56 + 2);
        a562 = *(a561);
        a563 = _mm_set1_epi8(a562);
        a564 = (a544 + 16);
        a565 = *(a564);
        a566 = _mm_xor_si128(a563, a565);
        a567 = ((__m128i ) a566);
        a568 = _mm_srli_epi16(a567, 2);
        a569 = ((__m128i ) a568);
        a570 = _mm_and_si128(a569, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a571 = (b56 + 3);
        a572 = *(a571);
        a573 = _mm_set1_epi8(a572);
        a574 = (a544 + 24);
        a575 = *(a574);
        a576 = _mm_xor_si128(a573, a575);
        a577 = ((__m128i ) a576);
        a578 = _mm_srli_epi16(a577, 2);
        a579 = ((__m128i ) a578);
        a580 = _mm_and_si128(a579, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b57 = _mm_adds_epu8(a550, a560);
        b58 = _mm_adds_epu8(b57, a570);
        t31 = _mm_adds_epu8(b58, a580);
        a581 = ((__m128i ) t31);
        a582 = _mm_srli_epi16(a581, 2);
        a583 = ((__m128i ) a582);
        t32 = _mm_and_si128(a583, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t33 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t32);
        m47 = _mm_adds_epu8(s66, t32);
        m48 = _mm_adds_epu8(s67, t33);
        m49 = _mm_adds_epu8(s66, t33);
        m50 = _mm_adds_epu8(s67, t32);
        a584 = _mm_min_epu8(m48, m47);
        d21 = _mm_cmpeq_epi8(a584, m48);
        a585 = _mm_min_epu8(m50, m49);
        d22 = _mm_cmpeq_epi8(a585, m50);
        s68 = _mm_movemask_epi8(_mm_unpacklo_epi8(d21,d22));
        a586 = ((short int  *) dec);
        a587 = (32 * i9);
        a588 = (a586 + a587);
        *(a588) = s68;
        s69 = _mm_movemask_epi8(_mm_unpackhi_epi8(d21,d22));
        a589 = (a588 + 1);
        *(a589) = s69;
        s70 = _mm_unpacklo_epi8(a584, a585);
        s71 = _mm_unpackhi_epi8(a584, a585);
        a590 = ((__m128i  *) Y);
        *(a590) = s70;
        a591 = (a590 + 1);
        *(a591) = s71;
        a592 = (a538 + 1);
        s72 = *(a592);
        a593 = (a538 + 9);
        s73 = *(a593);
        a594 = (a544 + 1);
        a595 = *(a594);
        a596 = _mm_xor_si128(a543, a595);
        a597 = ((__m128i ) a596);
        a598 = _mm_srli_epi16(a597, 2);
        a599 = ((__m128i ) a598);
        a600 = _mm_and_si128(a599, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a601 = (a544 + 9);
        a602 = *(a601);
        a603 = _mm_xor_si128(a553, a602);
        a604 = ((__m128i ) a603);
        a605 = _mm_srli_epi16(a604, 2);
        a606 = ((__m128i ) a605);
        a607 = _mm_and_si128(a606, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a608 = (a544 + 17);
        a609 = *(a608);
        a610 = _mm_xor_si128(a563, a609);
        a611 = ((__m128i ) a610);
        a612 = _mm_srli_epi16(a611, 2);
        a613 = ((__m128i ) a612);
        a614 = _mm_and_si128(a613, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a615 = (a544 + 25);
        a616 = *(a615);
        a617 = _mm_xor_si128(a573, a616);
        a618 = ((__m128i ) a617);
        a619 = _mm_srli_epi16(a618, 2);
        a620 = ((__m128i ) a619);
        a621 = _mm_and_si128(a620, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b59 = _mm_adds_epu8(a600, a607);
        b60 = _mm_adds_epu8(b59, a614);
        t34 = _mm_adds_epu8(b60, a621);
        a622 = ((__m128i ) t34);
        a623 = _mm_srli_epi16(a622, 2);
        a624 = ((__m128i ) a623);
        t35 = _mm_and_si128(a624, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t36 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t35);
        m51 = _mm_adds_epu8(s72, t35);
        m52 = _mm_adds_epu8(s73, t36);
        m53 = _mm_adds_epu8(s72, t36);
        m54 = _mm_adds_epu8(s73, t35);
        a625 = _mm_min_epu8(m52, m51);
        d23 = _mm_cmpeq_epi8(a625, m52);
        a626 = _mm_min_epu8(m54, m53);
        d24 = _mm_cmpeq_epi8(a626, m54);
        s74 = _mm_movemask_epi8(_mm_unpacklo_epi8(d23,d24));
        a627 = (a588 + 2);
        *(a627) = s74;
        s75 = _mm_movemask_epi8(_mm_unpackhi_epi8(d23,d24));
        a628 = (a588 + 3);
        *(a628) = s75;
        s76 = _mm_unpacklo_epi8(a625, a626);
        s77 = _mm_unpackhi_epi8(a625, a626);
        a629 = (a590 + 2);
        *(a629) = s76;
        a630 = (a590 + 3);
        *(a630) = s77;
        a631 = (a538 + 2);
        s78 = *(a631);
        a632 = (a538 + 10);
        s79 = *(a632);
        a633 = (a544 + 2);
        a634 = *(a633);
        a635 = _mm_xor_si128(a543, a634);
        a636 = ((__m128i ) a635);
        a637 = _mm_srli_epi16(a636, 2);
        a638 = ((__m128i ) a637);
        a639 = _mm_and_si128(a638, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a640 = (a544 + 10);
        a641 = *(a640);
        a642 = _mm_xor_si128(a553, a641);
        a643 = ((__m128i ) a642);
        a644 = _mm_srli_epi16(a643, 2);
        a645 = ((__m128i ) a644);
        a646 = _mm_and_si128(a645, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a647 = (a544 + 18);
        a648 = *(a647);
        a649 = _mm_xor_si128(a563, a648);
        a650 = ((__m128i ) a649);
        a651 = _mm_srli_epi16(a650, 2);
        a652 = ((__m128i ) a651);
        a653 = _mm_and_si128(a652, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a654 = (a544 + 26);
        a655 = *(a654);
        a656 = _mm_xor_si128(a573, a655);
        a657 = ((__m128i ) a656);
        a658 = _mm_srli_epi16(a657, 2);
        a659 = ((__m128i ) a658);
        a660 = _mm_and_si128(a659, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b61 = _mm_adds_epu8(a639, a646);
        b62 = _mm_adds_epu8(b61, a653);
        t37 = _mm_adds_epu8(b62, a660);
        a661 = ((__m128i ) t37);
        a662 = _mm_srli_epi16(a661, 2);
        a663 = ((__m128i ) a662);
        t38 = _mm_and_si128(a663, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t39 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t38);
        m55 = _mm_adds_epu8(s78, t38);
        m56 = _mm_adds_epu8(s79, t39);
        m57 = _mm_adds_epu8(s78, t39);
        m58 = _mm_adds_epu8(s79, t38);
        a664 = _mm_min_epu8(m56, m55);
        d25 = _mm_cmpeq_epi8(a664, m56);
        a665 = _mm_min_epu8(m58, m57);
        d26 = _mm_cmpeq_epi8(a665, m58);
        s80 = _mm_movemask_epi8(_mm_unpacklo_epi8(d25,d26));
        a666 = (a588 + 4);
        *(a666) = s80;
        s81 = _mm_movemask_epi8(_mm_unpackhi_epi8(d25,d26));
        a667 = (a588 + 5);
        *(a667) = s81;
        s82 = _mm_unpacklo_epi8(a664, a665);
        s83 = _mm_unpackhi_epi8(a664, a665);
        a668 = (a590 + 4);
        *(a668) = s82;
        a669 = (a590 + 5);
        *(a669) = s83;
        a670 = (a538 + 3);
        s84 = *(a670);
        a671 = (a538 + 11);
        s85 = *(a671);
        a672 = (a544 + 3);
        a673 = *(a672);
        a674 = _mm_xor_si128(a543, a673);
        a675 = ((__m128i ) a674);
        a676 = _mm_srli_epi16(a675, 2);
        a677 = ((__m128i ) a676);
        a678 = _mm_and_si128(a677, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a679 = (a544 + 11);
        a680 = *(a679);
        a681 = _mm_xor_si128(a553, a680);
        a682 = ((__m128i ) a681);
        a683 = _mm_srli_epi16(a682, 2);
        a684 = ((__m128i ) a683);
        a685 = _mm_and_si128(a684, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a686 = (a544 + 19);
        a687 = *(a686);
        a688 = _mm_xor_si128(a563, a687);
        a689 = ((__m128i ) a688);
        a690 = _mm_srli_epi16(a689, 2);
        a691 = ((__m128i ) a690);
        a692 = _mm_and_si128(a691, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a693 = (a544 + 27);
        a694 = *(a693);
        a695 = _mm_xor_si128(a573, a694);
        a696 = ((__m128i ) a695);
        a697 = _mm_srli_epi16(a696, 2);
        a698 = ((__m128i ) a697);
        a699 = _mm_and_si128(a698, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b63 = _mm_adds_epu8(a678, a685);
        b64 = _mm_adds_epu8(b63, a692);
        t40 = _mm_adds_epu8(b64, a699);
        a700 = ((__m128i ) t40);
        a701 = _mm_srli_epi16(a700, 2);
        a702 = ((__m128i ) a701);
        t41 = _mm_and_si128(a702, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t42 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t41);
        m59 = _mm_adds_epu8(s84, t41);
        m60 = _mm_adds_epu8(s85, t42);
        m61 = _mm_adds_epu8(s84, t42);
        m62 = _mm_adds_epu8(s85, t41);
        a703 = _mm_min_epu8(m60, m59);
        d27 = _mm_cmpeq_epi8(a703, m60);
        a704 = _mm_min_epu8(m62, m61);
        d28 = _mm_cmpeq_epi8(a704, m62);
        s86 = _mm_movemask_epi8(_mm_unpacklo_epi8(d27,d28));
        a705 = (a588 + 6);
        *(a705) = s86;
        s87 = _mm_movemask_epi8(_mm_unpackhi_epi8(d27,d28));
        a706 = (a588 + 7);
        *(a706) = s87;
        s88 = _mm_unpacklo_epi8(a703, a704);
        s89 = _mm_unpackhi_epi8(a703, a704);
        a707 = (a590 + 6);
        *(a707) = s88;
        a708 = (a590 + 7);
        *(a708) = s89;
        a709 = (a538 + 4);
        s90 = *(a709);
        a710 = (a538 + 12);
        s91 = *(a710);
        a711 = (a544 + 4);
        a712 = *(a711);
        a713 = _mm_xor_si128(a543, a712);
        a714 = ((__m128i ) a713);
        a715 = _mm_srli_epi16(a714, 2);
        a716 = ((__m128i ) a715);
        a717 = _mm_and_si128(a716, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a718 = (a544 + 12);
        a719 = *(a718);
        a720 = _mm_xor_si128(a553, a719);
        a721 = ((__m128i ) a720);
        a722 = _mm_srli_epi16(a721, 2);
        a723 = ((__m128i ) a722);
        a724 = _mm_and_si128(a723, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a725 = (a544 + 20);
        a726 = *(a725);
        a727 = _mm_xor_si128(a563, a726);
        a728 = ((__m128i ) a727);
        a729 = _mm_srli_epi16(a728, 2);
        a730 = ((__m128i ) a729);
        a731 = _mm_and_si128(a730, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a732 = (a544 + 28);
        a733 = *(a732);
        a734 = _mm_xor_si128(a573, a733);
        a735 = ((__m128i ) a734);
        a736 = _mm_srli_epi16(a735, 2);
        a737 = ((__m128i ) a736);
        a738 = _mm_and_si128(a737, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b65 = _mm_adds_epu8(a717, a724);
        b66 = _mm_adds_epu8(b65, a731);
        t43 = _mm_adds_epu8(b66, a738);
        a739 = ((__m128i ) t43);
        a740 = _mm_srli_epi16(a739, 2);
        a741 = ((__m128i ) a740);
        t44 = _mm_and_si128(a741, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t45 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t44);
        m63 = _mm_adds_epu8(s90, t44);
        m64 = _mm_adds_epu8(s91, t45);
        m65 = _mm_adds_epu8(s90, t45);
        m66 = _mm_adds_epu8(s91, t44);
        a742 = _mm_min_epu8(m64, m63);
        d29 = _mm_cmpeq_epi8(a742, m64);
        a743 = _mm_min_epu8(m66, m65);
        d30 = _mm_cmpeq_epi8(a743, m66);
        s92 = _mm_movemask_epi8(_mm_unpacklo_epi8(d29,d30));
        a744 = (a588 + 8);
        *(a744) = s92;
        s93 = _mm_movemask_epi8(_mm_unpackhi_epi8(d29,d30));
        a745 = (a588 + 9);
        *(a745) = s93;
        s94 = _mm_unpacklo_epi8(a742, a743);
        s95 = _mm_unpackhi_epi8(a742, a743);
        a746 = (a590 + 8);
        *(a746) = s94;
        a747 = (a590 + 9);
        *(a747) = s95;
        a748 = (a538 + 5);
        s96 = *(a748);
        a749 = (a538 + 13);
        s97 = *(a749);
        a750 = (a544 + 5);
        a751 = *(a750);
        a752 = _mm_xor_si128(a543, a751);
        a753 = ((__m128i ) a752);
        a754 = _mm_srli_epi16(a753, 2);
        a755 = ((__m128i ) a754);
        a756 = _mm_and_si128(a755, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a757 = (a544 + 13);
        a758 = *(a757);
        a759 = _mm_xor_si128(a553, a758);
        a760 = ((__m128i ) a759);
        a761 = _mm_srli_epi16(a760, 2);
        a762 = ((__m128i ) a761);
        a763 = _mm_and_si128(a762, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a764 = (a544 + 21);
        a765 = *(a764);
        a766 = _mm_xor_si128(a563, a765);
        a767 = ((__m128i ) a766);
        a768 = _mm_srli_epi16(a767, 2);
        a769 = ((__m128i ) a768);
        a770 = _mm_and_si128(a769, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a771 = (a544 + 29);
        a772 = *(a771);
        a773 = _mm_xor_si128(a573, a772);
        a774 = ((__m128i ) a773);
        a775 = _mm_srli_epi16(a774, 2);
        a776 = ((__m128i ) a775);
        a777 = _mm_and_si128(a776, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b67 = _mm_adds_epu8(a756, a763);
        b68 = _mm_adds_epu8(b67, a770);
        t46 = _mm_adds_epu8(b68, a777);
        a778 = ((__m128i ) t46);
        a779 = _mm_srli_epi16(a778, 2);
        a780 = ((__m128i ) a779);
        t47 = _mm_and_si128(a780, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t48 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t47);
        m67 = _mm_adds_epu8(s96, t47);
        m68 = _mm_adds_epu8(s97, t48);
        m69 = _mm_adds_epu8(s96, t48);
        m70 = _mm_adds_epu8(s97, t47);
        a781 = _mm_min_epu8(m68, m67);
        d31 = _mm_cmpeq_epi8(a781, m68);
        a782 = _mm_min_epu8(m70, m69);
        d32 = _mm_cmpeq_epi8(a782, m70);
        s98 = _mm_movemask_epi8(_mm_unpacklo_epi8(d31,d32));
        a783 = (a588 + 10);
        *(a783) = s98;
        s99 = _mm_movemask_epi8(_mm_unpackhi_epi8(d31,d32));
        a784 = (a588 + 11);
        *(a784) = s99;
        s100 = _mm_unpacklo_epi8(a781, a782);
        s101 = _mm_unpackhi_epi8(a781, a782);
        a785 = (a590 + 10);
        *(a785) = s100;
        a786 = (a590 + 11);
        *(a786) = s101;
        a787 = (a538 + 6);
        s102 = *(a787);
        a788 = (a538 + 14);
        s103 = *(a788);
        a789 = (a544 + 6);
        a790 = *(a789);
        a791 = _mm_xor_si128(a543, a790);
        a792 = ((__m128i ) a791);
        a793 = _mm_srli_epi16(a792, 2);
        a794 = ((__m128i ) a793);
        a795 = _mm_and_si128(a794, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a796 = (a544 + 14);
        a797 = *(a796);
        a798 = _mm_xor_si128(a553, a797);
        a799 = ((__m128i ) a798);
        a800 = _mm_srli_epi16(a799, 2);
        a801 = ((__m128i ) a800);
        a802 = _mm_and_si128(a801, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a803 = (a544 + 22);
        a804 = *(a803);
        a805 = _mm_xor_si128(a563, a804);
        a806 = ((__m128i ) a805);
        a807 = _mm_srli_epi16(a806, 2);
        a808 = ((__m128i ) a807);
        a809 = _mm_and_si128(a808, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a810 = (a544 + 30);
        a811 = *(a810);
        a812 = _mm_xor_si128(a573, a811);
        a813 = ((__m128i ) a812);
        a814 = _mm_srli_epi16(a813, 2);
        a815 = ((__m128i ) a814);
        a816 = _mm_and_si128(a815, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b69 = _mm_adds_epu8(a795, a802);
        b70 = _mm_adds_epu8(b69, a809);
        t49 = _mm_adds_epu8(b70, a816);
        a817 = ((__m128i ) t49);
        a818 = _mm_srli_epi16(a817, 2);
        a819 = ((__m128i ) a818);
        t50 = _mm_and_si128(a819, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t51 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t50);
        m71 = _mm_adds_epu8(s102, t50);
        m72 = _mm_adds_epu8(s103, t51);
        m73 = _mm_adds_epu8(s102, t51);
        m74 = _mm_adds_epu8(s103, t50);
        a820 = _mm_min_epu8(m72, m71);
        d33 = _mm_cmpeq_epi8(a820, m72);
        a821 = _mm_min_epu8(m74, m73);
        d34 = _mm_cmpeq_epi8(a821, m74);
        s104 = _mm_movemask_epi8(_mm_unpacklo_epi8(d33,d34));
        a822 = (a588 + 12);
        *(a822) = s104;
        s105 = _mm_movemask_epi8(_mm_unpackhi_epi8(d33,d34));
        a823 = (a588 + 13);
        *(a823) = s105;
        s106 = _mm_unpacklo_epi8(a820, a821);
        s107 = _mm_unpackhi_epi8(a820, a821);
        a824 = (a590 + 12);
        *(a824) = s106;
        a825 = (a590 + 13);
        *(a825) = s107;
        a826 = (a538 + 7);
        s108 = *(a826);
        a827 = (a538 + 15);
        s109 = *(a827);
        a828 = (a544 + 7);
        a829 = *(a828);
        a830 = _mm_xor_si128(a543, a829);
        a831 = ((__m128i ) a830);
        a832 = _mm_srli_epi16(a831, 2);
        a833 = ((__m128i ) a832);
        a834 = _mm_and_si128(a833, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a835 = (a544 + 15);
        a836 = *(a835);
        a837 = _mm_xor_si128(a553, a836);
        a838 = ((__m128i ) a837);
        a839 = _mm_srli_epi16(a838, 2);
        a840 = ((__m128i ) a839);
        a841 = _mm_and_si128(a840, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a842 = (a544 + 23);
        a843 = *(a842);
        a844 = _mm_xor_si128(a563, a843);
        a845 = ((__m128i ) a844);
        a846 = _mm_srli_epi16(a845, 2);
        a847 = ((__m128i ) a846);
        a848 = _mm_and_si128(a847, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a849 = (a544 + 31);
        a850 = *(a849);
        a851 = _mm_xor_si128(a573, a850);
        a852 = ((__m128i ) a851);
        a853 = _mm_srli_epi16(a852, 2);
        a854 = ((__m128i ) a853);
        a855 = _mm_and_si128(a854, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b71 = _mm_adds_epu8(a834, a841);
        b72 = _mm_adds_epu8(b71, a848);
        t52 = _mm_adds_epu8(b72, a855);
        a856 = ((__m128i ) t52);
        a857 = _mm_srli_epi16(a856, 2);
        a858 = ((__m128i ) a857);
        t53 = _mm_and_si128(a858, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t54 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t53);
        m75 = _mm_adds_epu8(s108, t53);
        m76 = _mm_adds_epu8(s109, t54);
        m77 = _mm_adds_epu8(s108, t54);
        m78 = _mm_adds_epu8(s109, t53);
        a859 = _mm_min_epu8(m76, m75);
        d35 = _mm_cmpeq_epi8(a859, m76);
        a860 = _mm_min_epu8(m78, m77);
        d36 = _mm_cmpeq_epi8(a860, m78);
        s110 = _mm_movemask_epi8(_mm_unpacklo_epi8(d35,d36));
        a861 = (a588 + 14);
        *(a861) = s110;
        s111 = _mm_movemask_epi8(_mm_unpackhi_epi8(d35,d36));
        a862 = (a588 + 15);
        *(a862) = s111;
        s112 = _mm_unpacklo_epi8(a859, a860);
        s113 = _mm_unpackhi_epi8(a859, a860);
        a863 = (a590 + 14);
        *(a863) = s112;
        a864 = (a590 + 15);
        *(a864) = s113;
        if ((((unsigned char  *) Y)[0]>103)) {
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
        unsigned char a1405, a1415, a1425, a1435;
        int a1403, a1450;
        short int s180, s181, s186, s187, s192, s193, s198
    , s199, s204, s205, s210, s211, s216, s217, s222
    , s223;
        unsigned char  *a1404, *a1414, *a1424, *a1434, *b137;
        short int  *a1449, *a1451, *a1452, *a1490, *a1491, *a1529, *a1530
                , *a1568, *a1569, *a1607, *a1608, *a1646, *a1647, *a1685, *a1686
                , *a1724, *a1725, *b140;
        __m128i  *a1401, *a1402, *a1407, *a1417, *a1427, *a1437, *a1453
                , *a1454, *a1455, *a1456, *a1457, *a1464, *a1471, *a1478, *a1492
                , *a1493, *a1494, *a1495, *a1496, *a1503, *a1510, *a1517, *a1531
                , *a1532, *a1533, *a1534, *a1535, *a1542, *a1549, *a1556, *a1570
                , *a1571, *a1572, *a1573, *a1574, *a1581, *a1588, *a1595, *a1609
                , *a1610, *a1611, *a1612, *a1613, *a1620, *a1627, *a1634, *a1648
                , *a1649, *a1650, *a1651, *a1652, *a1659, *a1666, *a1673, *a1687
                , *a1688, *a1689, *a1690, *a1691, *a1698, *a1705, *a1712, *a1726
                , *a1727;
        __m128i a1410, a1411, a1420, a1421, a1430, a1431, a1440
                , a1441, a1444, a1445, a1460, a1461, a1467, a1468, a1474
                , a1475, a1481, a1482, a1485, a1486, a1499, a1500, a1506
                , a1507, a1513, a1514, a1520, a1521, a1524, a1525, a1538
                , a1539, a1545, a1546, a1552, a1553, a1559, a1560, a1563
                , a1564, a1577, a1578, a1584, a1585, a1591, a1592, a1598
                , a1599, a1602, a1603, a1616, a1617, a1623, a1624, a1630
                , a1631, a1637, a1638, a1641, a1642, a1655, a1656, a1662
                , a1663, a1669, a1670, a1676, a1677, a1680, a1681, a1694
                , a1695, a1701, a1702, a1708, a1709, a1715, a1716, a1719
                , a1720;
        __m128i a1406, a1408, a1409, a1412, a1413, a1416, a1418
                , a1419, a1422, a1423, a1426, a1428, a1429, a1432, a1433
                , a1436, a1438, a1439, a1442, a1443, a1446, a1447, a1448
                , a1458, a1459, a1462, a1463, a1465, a1466, a1469, a1470
                , a1472, a1473, a1476, a1477, a1479, a1480, a1483, a1484
                , a1487, a1488, a1489, a1497, a1498, a1501, a1502, a1504
                , a1505, a1508, a1509, a1511, a1512, a1515, a1516, a1518
                , a1519, a1522, a1523, a1526, a1527, a1528, a1536, a1537
                , a1540, a1541, a1543, a1544, a1547, a1548, a1550, a1551
                , a1554, a1555, a1557, a1558, a1561, a1562, a1565, a1566
                , a1567, a1575, a1576, a1579, a1580, a1582, a1583, a1586
                , a1587, a1589, a1590, a1593, a1594, a1596, a1597, a1600
                , a1601, a1604, a1605, a1606, a1614, a1615, a1618, a1619
                , a1621, a1622, a1625, a1626, a1628, a1629, a1632, a1633
                , a1635, a1636, a1639, a1640, a1643, a1644, a1645, a1653
                , a1654, a1657, a1658, a1660, a1661, a1664, a1665, a1667
                , a1668, a1671, a1672, a1674, a1675, a1678, a1679, a1682
                , a1683, a1684, a1692, a1693, a1696, a1697, a1699, a1700
                , a1703, a1704, a1706, a1707, a1710, a1711, a1713, a1714
                , a1717, a1718, a1721, a1722, a1723, b138, b139, b141
                , b142, b143, b144, b145, b146, b147, b148, b149
                , b150, b151, b152, b153, b154, d53, d54, d55
                , d56, d57, d58, d59, d60, d61, d62, d63
                , d64, d65, d66, d67, d68, m111, m112, m113
                , m114, m115, m116, m117, m118, m119, m120, m121
                , m122, m123, m124, m125, m126, m127, m128, m129
                , m130, m131, m132, m133, m134, m135, m136, m137
                , m138, m139, m140, m141, m142, s178, s179, s182
                , s183, s184, s185, s188, s189, s190, s191, s194
                , s195, s196, s197, s200, s201, s202, s203, s206
                , s207, s208, s209, s212, s213, s214, s215, s218
                , s219, s220, s221, s224, s225, t100, t101, t102
                , t79, t80, t81, t82, t83, t84, t85, t86
                , t87, t88, t89, t90, t91, t92, t93, t94
                , t95, t96, t97, t98, t99;
        a1401 = ((__m128i  *) Y);
        s178 = *(a1401);
        a1402 = (a1401 + 8);
        s179 = *(a1402);
        a1403 = (8 * i9);
        b137 = (a1403 + syms);
        a1404 = (b137 + 4);
        a1405 = *(a1404);
        a1406 = _mm_set1_epi8(a1405);
        a1407 = ((__m128i  *) Branchtab);
        a1408 = *(a1407);
        a1409 = _mm_xor_si128(a1406, a1408);
        a1410 = ((__m128i ) a1409);
        a1411 = _mm_srli_epi16(a1410, 2);
        a1412 = ((__m128i ) a1411);
        a1413 = _mm_and_si128(a1412, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1414 = (b137 + 5);
        a1415 = *(a1414);
        a1416 = _mm_set1_epi8(a1415);
        a1417 = (a1407 + 8);
        a1418 = *(a1417);
        a1419 = _mm_xor_si128(a1416, a1418);
        a1420 = ((__m128i ) a1419);
        a1421 = _mm_srli_epi16(a1420, 2);
        a1422 = ((__m128i ) a1421);
        a1423 = _mm_and_si128(a1422, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1424 = (b137 + 6);
        a1425 = *(a1424);
        a1426 = _mm_set1_epi8(a1425);
        a1427 = (a1407 + 16);
        a1428 = *(a1427);
        a1429 = _mm_xor_si128(a1426, a1428);
        a1430 = ((__m128i ) a1429);
        a1431 = _mm_srli_epi16(a1430, 2);
        a1432 = ((__m128i ) a1431);
        a1433 = _mm_and_si128(a1432, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1434 = (b137 + 7);
        a1435 = *(a1434);
        a1436 = _mm_set1_epi8(a1435);
        a1437 = (a1407 + 24);
        a1438 = *(a1437);
        a1439 = _mm_xor_si128(a1436, a1438);
        a1440 = ((__m128i ) a1439);
        a1441 = _mm_srli_epi16(a1440, 2);
        a1442 = ((__m128i ) a1441);
        a1443 = _mm_and_si128(a1442, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b138 = _mm_adds_epu8(a1413, a1423);
        b139 = _mm_adds_epu8(b138, a1433);
        t79 = _mm_adds_epu8(b139, a1443);
        a1444 = ((__m128i ) t79);
        a1445 = _mm_srli_epi16(a1444, 2);
        a1446 = ((__m128i ) a1445);
        t80 = _mm_and_si128(a1446, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t81 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t80);
        m111 = _mm_adds_epu8(s178, t80);
        m112 = _mm_adds_epu8(s179, t81);
        m113 = _mm_adds_epu8(s178, t81);
        m114 = _mm_adds_epu8(s179, t80);
        a1447 = _mm_min_epu8(m112, m111);
        d53 = _mm_cmpeq_epi8(a1447, m112);
        a1448 = _mm_min_epu8(m114, m113);
        d54 = _mm_cmpeq_epi8(a1448, m114);
        s180 = _mm_movemask_epi8(_mm_unpacklo_epi8(d53,d54));
        a1449 = ((short int  *) dec);
        a1450 = (32 * i9);
        b140 = (a1449 + a1450);
        a1451 = (b140 + 16);
        *(a1451) = s180;
        s181 = _mm_movemask_epi8(_mm_unpackhi_epi8(d53,d54));
        a1452 = (b140 + 17);
        *(a1452) = s181;
        s182 = _mm_unpacklo_epi8(a1447, a1448);
        s183 = _mm_unpackhi_epi8(a1447, a1448);
        a1453 = ((__m128i  *) X);
        *(a1453) = s182;
        a1454 = (a1453 + 1);
        *(a1454) = s183;
        a1455 = (a1401 + 1);
        s184 = *(a1455);
        a1456 = (a1401 + 9);
        s185 = *(a1456);
        a1457 = (a1407 + 1);
        a1458 = *(a1457);
        a1459 = _mm_xor_si128(a1406, a1458);
        a1460 = ((__m128i ) a1459);
        a1461 = _mm_srli_epi16(a1460, 2);
        a1462 = ((__m128i ) a1461);
        a1463 = _mm_and_si128(a1462, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1464 = (a1407 + 9);
        a1465 = *(a1464);
        a1466 = _mm_xor_si128(a1416, a1465);
        a1467 = ((__m128i ) a1466);
        a1468 = _mm_srli_epi16(a1467, 2);
        a1469 = ((__m128i ) a1468);
        a1470 = _mm_and_si128(a1469, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1471 = (a1407 + 17);
        a1472 = *(a1471);
        a1473 = _mm_xor_si128(a1426, a1472);
        a1474 = ((__m128i ) a1473);
        a1475 = _mm_srli_epi16(a1474, 2);
        a1476 = ((__m128i ) a1475);
        a1477 = _mm_and_si128(a1476, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1478 = (a1407 + 25);
        a1479 = *(a1478);
        a1480 = _mm_xor_si128(a1436, a1479);
        a1481 = ((__m128i ) a1480);
        a1482 = _mm_srli_epi16(a1481, 2);
        a1483 = ((__m128i ) a1482);
        a1484 = _mm_and_si128(a1483, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b141 = _mm_adds_epu8(a1463, a1470);
        b142 = _mm_adds_epu8(b141, a1477);
        t82 = _mm_adds_epu8(b142, a1484);
        a1485 = ((__m128i ) t82);
        a1486 = _mm_srli_epi16(a1485, 2);
        a1487 = ((__m128i ) a1486);
        t83 = _mm_and_si128(a1487, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t84 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t83);
        m115 = _mm_adds_epu8(s184, t83);
        m116 = _mm_adds_epu8(s185, t84);
        m117 = _mm_adds_epu8(s184, t84);
        m118 = _mm_adds_epu8(s185, t83);
        a1488 = _mm_min_epu8(m116, m115);
        d55 = _mm_cmpeq_epi8(a1488, m116);
        a1489 = _mm_min_epu8(m118, m117);
        d56 = _mm_cmpeq_epi8(a1489, m118);
        s186 = _mm_movemask_epi8(_mm_unpacklo_epi8(d55,d56));
        a1490 = (b140 + 18);
        *(a1490) = s186;
        s187 = _mm_movemask_epi8(_mm_unpackhi_epi8(d55,d56));
        a1491 = (b140 + 19);
        *(a1491) = s187;
        s188 = _mm_unpacklo_epi8(a1488, a1489);
        s189 = _mm_unpackhi_epi8(a1488, a1489);
        a1492 = (a1453 + 2);
        *(a1492) = s188;
        a1493 = (a1453 + 3);
        *(a1493) = s189;
        a1494 = (a1401 + 2);
        s190 = *(a1494);
        a1495 = (a1401 + 10);
        s191 = *(a1495);
        a1496 = (a1407 + 2);
        a1497 = *(a1496);
        a1498 = _mm_xor_si128(a1406, a1497);
        a1499 = ((__m128i ) a1498);
        a1500 = _mm_srli_epi16(a1499, 2);
        a1501 = ((__m128i ) a1500);
        a1502 = _mm_and_si128(a1501, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1503 = (a1407 + 10);
        a1504 = *(a1503);
        a1505 = _mm_xor_si128(a1416, a1504);
        a1506 = ((__m128i ) a1505);
        a1507 = _mm_srli_epi16(a1506, 2);
        a1508 = ((__m128i ) a1507);
        a1509 = _mm_and_si128(a1508, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1510 = (a1407 + 18);
        a1511 = *(a1510);
        a1512 = _mm_xor_si128(a1426, a1511);
        a1513 = ((__m128i ) a1512);
        a1514 = _mm_srli_epi16(a1513, 2);
        a1515 = ((__m128i ) a1514);
        a1516 = _mm_and_si128(a1515, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1517 = (a1407 + 26);
        a1518 = *(a1517);
        a1519 = _mm_xor_si128(a1436, a1518);
        a1520 = ((__m128i ) a1519);
        a1521 = _mm_srli_epi16(a1520, 2);
        a1522 = ((__m128i ) a1521);
        a1523 = _mm_and_si128(a1522, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b143 = _mm_adds_epu8(a1502, a1509);
        b144 = _mm_adds_epu8(b143, a1516);
        t85 = _mm_adds_epu8(b144, a1523);
        a1524 = ((__m128i ) t85);
        a1525 = _mm_srli_epi16(a1524, 2);
        a1526 = ((__m128i ) a1525);
        t86 = _mm_and_si128(a1526, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t87 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t86);
        m119 = _mm_adds_epu8(s190, t86);
        m120 = _mm_adds_epu8(s191, t87);
        m121 = _mm_adds_epu8(s190, t87);
        m122 = _mm_adds_epu8(s191, t86);
        a1527 = _mm_min_epu8(m120, m119);
        d57 = _mm_cmpeq_epi8(a1527, m120);
        a1528 = _mm_min_epu8(m122, m121);
        d58 = _mm_cmpeq_epi8(a1528, m122);
        s192 = _mm_movemask_epi8(_mm_unpacklo_epi8(d57,d58));
        a1529 = (b140 + 20);
        *(a1529) = s192;
        s193 = _mm_movemask_epi8(_mm_unpackhi_epi8(d57,d58));
        a1530 = (b140 + 21);
        *(a1530) = s193;
        s194 = _mm_unpacklo_epi8(a1527, a1528);
        s195 = _mm_unpackhi_epi8(a1527, a1528);
        a1531 = (a1453 + 4);
        *(a1531) = s194;
        a1532 = (a1453 + 5);
        *(a1532) = s195;
        a1533 = (a1401 + 3);
        s196 = *(a1533);
        a1534 = (a1401 + 11);
        s197 = *(a1534);
        a1535 = (a1407 + 3);
        a1536 = *(a1535);
        a1537 = _mm_xor_si128(a1406, a1536);
        a1538 = ((__m128i ) a1537);
        a1539 = _mm_srli_epi16(a1538, 2);
        a1540 = ((__m128i ) a1539);
        a1541 = _mm_and_si128(a1540, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1542 = (a1407 + 11);
        a1543 = *(a1542);
        a1544 = _mm_xor_si128(a1416, a1543);
        a1545 = ((__m128i ) a1544);
        a1546 = _mm_srli_epi16(a1545, 2);
        a1547 = ((__m128i ) a1546);
        a1548 = _mm_and_si128(a1547, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1549 = (a1407 + 19);
        a1550 = *(a1549);
        a1551 = _mm_xor_si128(a1426, a1550);
        a1552 = ((__m128i ) a1551);
        a1553 = _mm_srli_epi16(a1552, 2);
        a1554 = ((__m128i ) a1553);
        a1555 = _mm_and_si128(a1554, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1556 = (a1407 + 27);
        a1557 = *(a1556);
        a1558 = _mm_xor_si128(a1436, a1557);
        a1559 = ((__m128i ) a1558);
        a1560 = _mm_srli_epi16(a1559, 2);
        a1561 = ((__m128i ) a1560);
        a1562 = _mm_and_si128(a1561, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b145 = _mm_adds_epu8(a1541, a1548);
        b146 = _mm_adds_epu8(b145, a1555);
        t88 = _mm_adds_epu8(b146, a1562);
        a1563 = ((__m128i ) t88);
        a1564 = _mm_srli_epi16(a1563, 2);
        a1565 = ((__m128i ) a1564);
        t89 = _mm_and_si128(a1565, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t90 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t89);
        m123 = _mm_adds_epu8(s196, t89);
        m124 = _mm_adds_epu8(s197, t90);
        m125 = _mm_adds_epu8(s196, t90);
        m126 = _mm_adds_epu8(s197, t89);
        a1566 = _mm_min_epu8(m124, m123);
        d59 = _mm_cmpeq_epi8(a1566, m124);
        a1567 = _mm_min_epu8(m126, m125);
        d60 = _mm_cmpeq_epi8(a1567, m126);
        s198 = _mm_movemask_epi8(_mm_unpacklo_epi8(d59,d60));
        a1568 = (b140 + 22);
        *(a1568) = s198;
        s199 = _mm_movemask_epi8(_mm_unpackhi_epi8(d59,d60));
        a1569 = (b140 + 23);
        *(a1569) = s199;
        s200 = _mm_unpacklo_epi8(a1566, a1567);
        s201 = _mm_unpackhi_epi8(a1566, a1567);
        a1570 = (a1453 + 6);
        *(a1570) = s200;
        a1571 = (a1453 + 7);
        *(a1571) = s201;
        a1572 = (a1401 + 4);
        s202 = *(a1572);
        a1573 = (a1401 + 12);
        s203 = *(a1573);
        a1574 = (a1407 + 4);
        a1575 = *(a1574);
        a1576 = _mm_xor_si128(a1406, a1575);
        a1577 = ((__m128i ) a1576);
        a1578 = _mm_srli_epi16(a1577, 2);
        a1579 = ((__m128i ) a1578);
        a1580 = _mm_and_si128(a1579, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1581 = (a1407 + 12);
        a1582 = *(a1581);
        a1583 = _mm_xor_si128(a1416, a1582);
        a1584 = ((__m128i ) a1583);
        a1585 = _mm_srli_epi16(a1584, 2);
        a1586 = ((__m128i ) a1585);
        a1587 = _mm_and_si128(a1586, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1588 = (a1407 + 20);
        a1589 = *(a1588);
        a1590 = _mm_xor_si128(a1426, a1589);
        a1591 = ((__m128i ) a1590);
        a1592 = _mm_srli_epi16(a1591, 2);
        a1593 = ((__m128i ) a1592);
        a1594 = _mm_and_si128(a1593, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1595 = (a1407 + 28);
        a1596 = *(a1595);
        a1597 = _mm_xor_si128(a1436, a1596);
        a1598 = ((__m128i ) a1597);
        a1599 = _mm_srli_epi16(a1598, 2);
        a1600 = ((__m128i ) a1599);
        a1601 = _mm_and_si128(a1600, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b147 = _mm_adds_epu8(a1580, a1587);
        b148 = _mm_adds_epu8(b147, a1594);
        t91 = _mm_adds_epu8(b148, a1601);
        a1602 = ((__m128i ) t91);
        a1603 = _mm_srli_epi16(a1602, 2);
        a1604 = ((__m128i ) a1603);
        t92 = _mm_and_si128(a1604, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t93 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t92);
        m127 = _mm_adds_epu8(s202, t92);
        m128 = _mm_adds_epu8(s203, t93);
        m129 = _mm_adds_epu8(s202, t93);
        m130 = _mm_adds_epu8(s203, t92);
        a1605 = _mm_min_epu8(m128, m127);
        d61 = _mm_cmpeq_epi8(a1605, m128);
        a1606 = _mm_min_epu8(m130, m129);
        d62 = _mm_cmpeq_epi8(a1606, m130);
        s204 = _mm_movemask_epi8(_mm_unpacklo_epi8(d61,d62));
        a1607 = (b140 + 24);
        *(a1607) = s204;
        s205 = _mm_movemask_epi8(_mm_unpackhi_epi8(d61,d62));
        a1608 = (b140 + 25);
        *(a1608) = s205;
        s206 = _mm_unpacklo_epi8(a1605, a1606);
        s207 = _mm_unpackhi_epi8(a1605, a1606);
        a1609 = (a1453 + 8);
        *(a1609) = s206;
        a1610 = (a1453 + 9);
        *(a1610) = s207;
        a1611 = (a1401 + 5);
        s208 = *(a1611);
        a1612 = (a1401 + 13);
        s209 = *(a1612);
        a1613 = (a1407 + 5);
        a1614 = *(a1613);
        a1615 = _mm_xor_si128(a1406, a1614);
        a1616 = ((__m128i ) a1615);
        a1617 = _mm_srli_epi16(a1616, 2);
        a1618 = ((__m128i ) a1617);
        a1619 = _mm_and_si128(a1618, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1620 = (a1407 + 13);
        a1621 = *(a1620);
        a1622 = _mm_xor_si128(a1416, a1621);
        a1623 = ((__m128i ) a1622);
        a1624 = _mm_srli_epi16(a1623, 2);
        a1625 = ((__m128i ) a1624);
        a1626 = _mm_and_si128(a1625, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1627 = (a1407 + 21);
        a1628 = *(a1627);
        a1629 = _mm_xor_si128(a1426, a1628);
        a1630 = ((__m128i ) a1629);
        a1631 = _mm_srli_epi16(a1630, 2);
        a1632 = ((__m128i ) a1631);
        a1633 = _mm_and_si128(a1632, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1634 = (a1407 + 29);
        a1635 = *(a1634);
        a1636 = _mm_xor_si128(a1436, a1635);
        a1637 = ((__m128i ) a1636);
        a1638 = _mm_srli_epi16(a1637, 2);
        a1639 = ((__m128i ) a1638);
        a1640 = _mm_and_si128(a1639, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b149 = _mm_adds_epu8(a1619, a1626);
        b150 = _mm_adds_epu8(b149, a1633);
        t94 = _mm_adds_epu8(b150, a1640);
        a1641 = ((__m128i ) t94);
        a1642 = _mm_srli_epi16(a1641, 2);
        a1643 = ((__m128i ) a1642);
        t95 = _mm_and_si128(a1643, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t96 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t95);
        m131 = _mm_adds_epu8(s208, t95);
        m132 = _mm_adds_epu8(s209, t96);
        m133 = _mm_adds_epu8(s208, t96);
        m134 = _mm_adds_epu8(s209, t95);
        a1644 = _mm_min_epu8(m132, m131);
        d63 = _mm_cmpeq_epi8(a1644, m132);
        a1645 = _mm_min_epu8(m134, m133);
        d64 = _mm_cmpeq_epi8(a1645, m134);
        s210 = _mm_movemask_epi8(_mm_unpacklo_epi8(d63,d64));
        a1646 = (b140 + 26);
        *(a1646) = s210;
        s211 = _mm_movemask_epi8(_mm_unpackhi_epi8(d63,d64));
        a1647 = (b140 + 27);
        *(a1647) = s211;
        s212 = _mm_unpacklo_epi8(a1644, a1645);
        s213 = _mm_unpackhi_epi8(a1644, a1645);
        a1648 = (a1453 + 10);
        *(a1648) = s212;
        a1649 = (a1453 + 11);
        *(a1649) = s213;
        a1650 = (a1401 + 6);
        s214 = *(a1650);
        a1651 = (a1401 + 14);
        s215 = *(a1651);
        a1652 = (a1407 + 6);
        a1653 = *(a1652);
        a1654 = _mm_xor_si128(a1406, a1653);
        a1655 = ((__m128i ) a1654);
        a1656 = _mm_srli_epi16(a1655, 2);
        a1657 = ((__m128i ) a1656);
        a1658 = _mm_and_si128(a1657, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1659 = (a1407 + 14);
        a1660 = *(a1659);
        a1661 = _mm_xor_si128(a1416, a1660);
        a1662 = ((__m128i ) a1661);
        a1663 = _mm_srli_epi16(a1662, 2);
        a1664 = ((__m128i ) a1663);
        a1665 = _mm_and_si128(a1664, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1666 = (a1407 + 22);
        a1667 = *(a1666);
        a1668 = _mm_xor_si128(a1426, a1667);
        a1669 = ((__m128i ) a1668);
        a1670 = _mm_srli_epi16(a1669, 2);
        a1671 = ((__m128i ) a1670);
        a1672 = _mm_and_si128(a1671, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1673 = (a1407 + 30);
        a1674 = *(a1673);
        a1675 = _mm_xor_si128(a1436, a1674);
        a1676 = ((__m128i ) a1675);
        a1677 = _mm_srli_epi16(a1676, 2);
        a1678 = ((__m128i ) a1677);
        a1679 = _mm_and_si128(a1678, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b151 = _mm_adds_epu8(a1658, a1665);
        b152 = _mm_adds_epu8(b151, a1672);
        t97 = _mm_adds_epu8(b152, a1679);
        a1680 = ((__m128i ) t97);
        a1681 = _mm_srli_epi16(a1680, 2);
        a1682 = ((__m128i ) a1681);
        t98 = _mm_and_si128(a1682, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t99 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t98);
        m135 = _mm_adds_epu8(s214, t98);
        m136 = _mm_adds_epu8(s215, t99);
        m137 = _mm_adds_epu8(s214, t99);
        m138 = _mm_adds_epu8(s215, t98);
        a1683 = _mm_min_epu8(m136, m135);
        d65 = _mm_cmpeq_epi8(a1683, m136);
        a1684 = _mm_min_epu8(m138, m137);
        d66 = _mm_cmpeq_epi8(a1684, m138);
        s216 = _mm_movemask_epi8(_mm_unpacklo_epi8(d65,d66));
        a1685 = (b140 + 28);
        *(a1685) = s216;
        s217 = _mm_movemask_epi8(_mm_unpackhi_epi8(d65,d66));
        a1686 = (b140 + 29);
        *(a1686) = s217;
        s218 = _mm_unpacklo_epi8(a1683, a1684);
        s219 = _mm_unpackhi_epi8(a1683, a1684);
        a1687 = (a1453 + 12);
        *(a1687) = s218;
        a1688 = (a1453 + 13);
        *(a1688) = s219;
        a1689 = (a1401 + 7);
        s220 = *(a1689);
        a1690 = (a1401 + 15);
        s221 = *(a1690);
        a1691 = (a1407 + 7);
        a1692 = *(a1691);
        a1693 = _mm_xor_si128(a1406, a1692);
        a1694 = ((__m128i ) a1693);
        a1695 = _mm_srli_epi16(a1694, 2);
        a1696 = ((__m128i ) a1695);
        a1697 = _mm_and_si128(a1696, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1698 = (a1407 + 15);
        a1699 = *(a1698);
        a1700 = _mm_xor_si128(a1416, a1699);
        a1701 = ((__m128i ) a1700);
        a1702 = _mm_srli_epi16(a1701, 2);
        a1703 = ((__m128i ) a1702);
        a1704 = _mm_and_si128(a1703, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1705 = (a1407 + 23);
        a1706 = *(a1705);
        a1707 = _mm_xor_si128(a1426, a1706);
        a1708 = ((__m128i ) a1707);
        a1709 = _mm_srli_epi16(a1708, 2);
        a1710 = ((__m128i ) a1709);
        a1711 = _mm_and_si128(a1710, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        a1712 = (a1407 + 31);
        a1713 = *(a1712);
        a1714 = _mm_xor_si128(a1436, a1713);
        a1715 = ((__m128i ) a1714);
        a1716 = _mm_srli_epi16(a1715, 2);
        a1717 = ((__m128i ) a1716);
        a1718 = _mm_and_si128(a1717, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        b153 = _mm_adds_epu8(a1697, a1704);
        b154 = _mm_adds_epu8(b153, a1711);
        t100 = _mm_adds_epu8(b154, a1718);
        a1719 = ((__m128i ) t100);
        a1720 = _mm_srli_epi16(a1719, 2);
        a1721 = ((__m128i ) a1720);
        t101 = _mm_and_si128(a1721, _mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63));
        t102 = _mm_subs_epu8(_mm_set_epi8(63, 63, 63, 63, 63, 63, 63
    , 63, 63, 63, 63, 63, 63, 63, 63
    , 63), t101);
        m139 = _mm_adds_epu8(s220, t101);
        m140 = _mm_adds_epu8(s221, t102);
        m141 = _mm_adds_epu8(s220, t102);
        m142 = _mm_adds_epu8(s221, t101);
        a1722 = _mm_min_epu8(m140, m139);
        d67 = _mm_cmpeq_epi8(a1722, m140);
        a1723 = _mm_min_epu8(m142, m141);
        d68 = _mm_cmpeq_epi8(a1723, m142);
        s222 = _mm_movemask_epi8(_mm_unpacklo_epi8(d67,d68));
        a1724 = (b140 + 30);
        *(a1724) = s222;
        s223 = _mm_movemask_epi8(_mm_unpackhi_epi8(d67,d68));
        a1725 = (b140 + 31);
        *(a1725) = s223;
        s224 = _mm_unpacklo_epi8(a1722, a1723);
        s225 = _mm_unpackhi_epi8(a1722, a1723);
        a1726 = (a1453 + 14);
        *(a1726) = s224;
        a1727 = (a1453 + 15);
        *(a1727) = s225;
        if ((((unsigned char  *) X)[0]>103)) {
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


void update_spiral49(spiral49 *vp, COMPUTETYPE *syms, int nbits) {
  FULL_SPIRAL(vp->new_metrics->t, vp->old_metrics->t, syms, vp->decisions->c, Branchtab, nbits/2);
}
