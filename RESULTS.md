# Caveats
Our viterbi decoder implementation differs to ka9q's in several ways.
- Due to error metrics renormalisation the magnitude of our soft decision symbols matters alot. A higher symbol magnitude means the error metric accumulates more quickly, leading to more frequent renormalisation and lower performance. This is why the "hard8" config performs better than "soft8" config for our decoders.
- Overall structure of the butterfly algorithm are the same, however we use saturating arithmetic over modular arithmetic. This means we avoid error metric overflows however we suffer from lower performance due to higher CPI.
- Chainback algorithm is the same but ours is more suspectible to the whims of the compiler and its optimisations since the decision bits type varies. It is sometimes much faster than ka9q due to vectorisation, and sometimes alot slower. It is unclear what specifically causes the performance differences.
- Chainback performance isn't written explicitly with intrinsics so whether or not it undergoes auto-vectorisation will depend on the compiler.
- Assembly versions for ka9q decoders where they exist, were not able to be compiled since they were written for 32bit. Instead the C version with intrinsics was used instead.

# Results
- CPU: AMD 7735HS
- Compiler: clang 16.0.5 x86_64-pc-windows-msvc
## Update symbol rate
| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |
| 7 | 2 | 465±9.3M | 457±8.3M | 553±13M | 485±7.9M | 321±7.3M | 406±6.9M |
| 7 | 4 | --- | 589±14M | 922±26M | 952±48M | 476±12M | 774±11M |
| 9 | 2 | 152±5M | 137±3.3M | 167±3.4M | 248±8.8M | 79.1±6M | 136±5.8M |
| 9 | 4 | --- | 198±5.9M | 242±4.6M | 361±13M | 120±2.7M | 210±3M |
| 15 | 6 | 3.19±0.039M | 3.5±0.056M | 3.31±0.048M | 5.7±0.083M | 2.13±0.02M | 3.81±0.033M |
| 24 | 2 | 1.09±0.044k | --- | 2.57±0.11k | 2.82±0.06k | 1.09±0.016k | 1.13±0.011k |

## Chainback bit rate
| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |
| 7 | 2 | 509±9.7M | 475±7M | 867±22M | 876±16M | 875±22M | 872±21M |
| 7 | 4 | --- | 474±7.9M | 871±28M | 861±41M | 877±29M | 876±15M |
| 9 | 2 | 448±11M | 469±11M | 388±6.9M | 396±13M | 389±13M | 389±17M |
| 9 | 4 | --- | 471±11M | 391±8.9M | 395±12M | 393±9.3M | 396±6.5M |
| 15 | 6 | 91.5±6.8M | 93±7.5M | 88.2±9.1M | 90.9±3.7M | 86.4±8M | 87.5±4.5M |
| 24 | 2 | 3.41±0.16M | --- | 3.68±0.09M | 3.51±0.077M | 3.58±0.39M | 3.62±0.096M |

# Results
- CPU: AMD Ryzen 5 3600
- Compiler: clang 15.0.1 x86_64-pc-windows-msvc
## Update symbol rate
| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |
| 7 | 2 | 363±9.8M | 373±6.6M | 472±9.4M | 408±9.8M | 274±4.8M | 332±5.6M |
| 7 | 4 | --- | 458±7.4M | 760±13M | 800±21M | 424±6.9M | 606±12M |
| 9 | 2 | 105±2.1M | 89±1.3M | 130±2M | 228±5M | 70.4±0.99M | 125±2.2M |
| 9 | 4 | --- | 119±1.6M | 204±2.7M | 355±6.3M | 110±1.6M | 199±5.4M |
| 15 | 6 | 2.55±0.018M | 2.97±0.016M | 2.43±0.035M | 4.63±0.015M | 2.04±0.0049M | 3.88±0.01M |
| 24 | 2 | 779±2 | --- | 1.89±0.0057k | 1.99±0.0061k | 974±4.5 | 1±0.0025k |

## Chainback bit rate
| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |
| 7 | 2 | 429±10M | 404±12M | 748±18M | 747±19M | 747±17M | 746±17M |
| 7 | 4 | --- | 401±9M | 746±20M | 747±15M | 746±18M | 747±17M |
| 9 | 2 | 433±11M | 404±9.7M | 371±9.9M | 371±8.9M | 371±9.8M | 371±8.2M |
| 9 | 4 | --- | 404±10M | 370±9.2M | 371±7.7M | 371±9.6M | 371±11M |
| 15 | 6 | 94.3±5M | 96.3±2.6M | 89.5±11M | 93.4±1.6M | 92.8±1.4M | 92.7±1.6M |
| 24 | 2 | 5.4±0.14M | --- | 5.29±0.14M | 5.47±0.18M | 5.31±0.11M | 5.19±0.22M |

# Results
- CPU: Intel i5-7200u
- Compiler: clang 14.0.0 x86_64-pc-linux-gnu
## Update symbol rate
| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |
| 7 | 2 | 251±6.9M | 271±7.1M | 290±9.2M | 352±11M | 150±3.3M | 234±6.9M |
| 7 | 4 | --- | 326±8.5M | 423±19M | 644±36M | 263±8.6M | 384±16M |
| 9 | 2 | 67.8±2.5M | 67±2.5M | 76.1±2.8M | 122±5.7M | 40.5±1.2M | 74.1±2.9M |
| 9 | 4 | --- | 89.9±2.6M | 121±3.9M | 197±7.6M | 67±0.78M | 117±1.5M |
| 15 | 6 | 1.32±0.035M | 1.64±0.039M | 1.83±0.036M | 3.16±0.062M | 1.22±0.01M | 2±0.025M |
| 24 | 2 | 394±7.4 | --- | 798±7.3 | 780±9.5 | 394±2.2 | 395±1.7 |

## Chainback bit rate
| K    | R    | ka9q | spiral | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| ---- | ---- | ---- | ------ | ------ | ------ | ------- | ------- |
| 7 | 2 | 294±10M | 306±9M | 403±13M | 403±13M | 357±10M | 358±8.5M |
| 7 | 4 | --- | 305±12M | 401±21M | 401±23M | 357±16M | 360±16M |
| 9 | 2 | 167±9.4M | 257±17M | 216±14M | 214±16M | 202±11M | 212±13M |
| 9 | 4 | --- | 265±19M | 222±16M | 223±15M | 221±14M | 221±13M |
| 15 | 6 | 22.5±3M | 55.7±14M | 56.5±12M | 58±13M | 47.9±11M | 49.7±10M |
| 24 | 2 | 6.97±0.28M | --- | 6.6±0.19M | 6.57±0.2M | 6.51±0.22M | 6.63±0.2M |

