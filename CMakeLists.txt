cmake_minimum_required(VERSION 3.16)
project(Demo)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2")

find_package(OpenMP REQUIRED)
add_subdirectory(benchmark)
add_subdirectory(aggregation)


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/include
        ${CMAKE_CURRENT_SOURCE_DIR}/aggregation/include
)


find_package(OpenMP REQUIRED)

add_executable(DemoGeMM main_GEMM.cpp
        GEMM.h)

target_link_libraries(DemoGeMM
        swbench::swbench )

add_executable(DemoSpMM main_SPMM.cpp
        SPMM.h)

target_link_libraries(DemoSpMM
        aggregation
        swbench::swbench )
