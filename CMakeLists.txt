cmake_minimum_required(VERSION 3.10)
project(viterbi_compare)

set(viterbi_DIR ${CMAKE_SOURCE_DIR}/williamyang_viterbi_lib)
find_package(viterbi CONFIG REQUIRED)

set(KA9Q_DIR ${CMAKE_SOURCE_DIR}/ka9q_libfec_port)
add_library(ka9q_port STATIC 
    ${KA9Q_DIR}/viterbi27_sse2.cpp 
    ${KA9Q_DIR}/viterbi29_sse2.cpp 
    ${KA9Q_DIR}/viterbi615_sse2.cpp
    ${KA9Q_DIR}/viterbi224_sse2.cpp
)
target_compile_features(ka9q_port PRIVATE cxx_std_17)
target_include_directories(ka9q_port PRIVATE ${KA9Q_DIR})

set(SPIRAL_DIR ${CMAKE_SOURCE_DIR}/spiral)
add_library(spiral STATIC 
    ${SPIRAL_DIR}/spiral27.cpp 
    ${SPIRAL_DIR}/spiral29.cpp 
    ${SPIRAL_DIR}/spiral47.cpp 
    ${SPIRAL_DIR}/spiral49.cpp 
    ${SPIRAL_DIR}/spiral615.cpp 
)
target_compile_features(spiral PRIVATE cxx_std_17)
target_include_directories(spiral PRIVATE ${SPIRAL_DIR})

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
add_executable(main ${SRC_DIR}/main.cpp)
target_include_directories(main PRIVATE ${SRC_DIR} ${KA9Q_DIR} ${SPIRAL_DIR})
target_compile_features(main PRIVATE cxx_std_17)
target_link_libraries(main PRIVATE viterbi spiral ka9q_port)
