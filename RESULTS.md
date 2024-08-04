# Results
- CPU: AMD 7735HS
- Compiler: clang 16.0.5 x86_64-pc-windows-msvc
## Update symbol rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 448M | 544M | 474M | 313M | 388M |
| 15 | 6 | 3.05M | 3.24M | 5.55M | 2.15M | 3.81M |
| 24 | 2 | 1.00k | 2.35k | 2.65k | 1.04k | 1.07k |

## Chainback bit rate
| K   | R   | ka9q | sse-u8 | avx-u8 | sse-u16 | avx-u16 |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2 | 490M | 799M | 802M | 801M | 784M |
| 15 | 6 | 77.1M | 76.3M | 81.6M | 72.5M | 80.9M |
| 24 | 2 | 3.44M | 3.33M | 2.71M | 3.55M | 3.46M |

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
