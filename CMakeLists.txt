cmake_minimum_required(VERSION 3.20)
project(cuda_checkers LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda-checkers
    src/main.cu
)

target_compile_options(cuda-checkers PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-DFORMATTING> # remove to disable console formatting
    $<$<COMPILE_LANGUAGE:CUDA>:-DINTRIN> # remove to disable intrinsics usage (if it doesn't compile on old machines)
)

set_target_properties(cuda-checkers PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

