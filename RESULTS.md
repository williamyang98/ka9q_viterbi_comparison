# Caveats
Our viterbi decoder implementation differs to ka9q's in several ways.
- Due to error metrics renormalisation the magnitude of our soft decision symbols matters alot. A higher symbol magnitude means the error metric accumulates more quickly, leading to more frequent renormalisation and lower performance. This is why the "hard8" config performs better than "soft8" config for our decoders.
- Overall structure of the butterfly algorithm are the same, however we use saturating arithmetic over modular arithmetic. This means we avoid error metric overflows however we suffer from lower performance due to higher CPI.
- Chainback algorithm is the same but ours is more suspectible to the whims of the compiler and its optimisations since the decision bits type varies. It is sometimes much faster than ka9q due to vectorisation, and sometimes alot slower. It is unclear what specifically causes the performance differences.
- Chainback performance isn't written explicitly with intrinsics so whether or not it undergoes auto-vectorisation will depend on the compiler.

# Results
- CPU: AMD 7735HS
- Compiler: clang 16.0.5 x86_64-pc-windows-msvc
## Update symbol rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 462M | 551M | 480M | 321M | 405M |
| 9 | 2 | 155M | 171M | 242M | 80.4M | 139M |
| 15 | 6 | 3.07M | 3.23M | 5.61M | 2.13M | 3.87M |
| 24 | 2 | 1.09k | 2.58k | 2.81k | 1.10k | 1.14k |

## Chainback bit rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 506M | 806M | 812M | 813M | 813M |
| 9 | 2 | 447M | 387M | 379M | 384M | 387M |
| 15 | 6 | 88.8M | 87.8M | 90.3M | 87.1M | 88.2M |
| 24 | 2 | 3.79M | 3.49M | 3.15M | 3.76M | 3.85M |

# Results
- CPU: AMD Ryzen 5 3600
- Compiler: clang 15.0.1 x86_64-pc-windows-msvc
## Update symbol rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 360M | 468M | 406M | 273M | 332M |
| 9 | 2 | 105M | 130M | 229M | 57.4M | 125M |
| 15 | 6 | 2.54M | 3.02M | 5.20M | 1.98M | 3.69M |
| 24 | 2 | 787 | 1.79k | 1.93k | 922 | 958 |

## Chainback bit rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 428M | 744M | 744M | 747M | 745M |
| 9 | 2 | 434M | 396M | 396M | 396M | 396M |
| 15 | 6 | 94.9M | 93.3M | 93.9M | 93.8M | 93.9M |
| 24 | 2 | 5.36M | 5.16M | 5.27M | 5.74M | 5.71M |

# Results
- CPU: Intel i5-7200u
- Compiler: clang 14.0.0 x86_64-pc-linux-gnu
## Update symbol rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 251M | 283M | 350M | 168M | 229M |
| 7 | 9 | 67.6M | 76.1M | 122M | 43.4M | 73.1M |
| 15 | 6 | 1.33M | 1.76M | 2.87M | 1.23M | 2.12M |
| 24 | 2 | 383 | 772 | 769 | 390 | 390 |

## Chainback bit rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 288M | 404M | 400M | 356M | 356M |
| 7 | 9 | 173M | 207M | 209M | 201M | 201M |
| 15 | 6 | 25.2M | 39.3M | 41.8M | 34.7M | 40.6M |
| 24 | 2 | 6.62M | 6.50M | 6.52M | 6.86M | 6.68M |

