# Introduction
Benchmark comparisons between Viterbi decoders
- [ka9q](https://github.com/ka9q/libfec)
- [williamyang98](https://github.com/williamyang98/ViterbiDecoderCpp)

# Build instructions
1. Configure cmake: ```cmake . -B build --preset windows-msvc -DCMAKE_BUILD_TYPE=Release```.
2. Compile program: ```cmake --build build```.
3. Run program: ```./build/main.exe```.

# Modify program to benchmark other codes
Change ```#define CONFIG 0``` to a different value.
