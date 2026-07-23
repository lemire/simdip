cmake_minimum_required(VERSION 3.10)

project(BenchmarkProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include(cmake/CPM.cmake)

include_directories(benchmarks src include)

add_executable(benchmark benchmarks/benchmark.cpp)
target_include_directories(benchmark PRIVATE include)
target_compile_options(benchmark PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>
)

CPMAddPackage("gh:lemire/counters#v3.1.0")
target_link_libraries(benchmark PRIVATE counters)

# --- Competitor parsers fetched via CPM (no local installs required) ---

# jrepp/ipv6-parse: full-RFC pure-C IPv6 parser (correctness/feature reference).
# LIBRARY_ONLY (its default) builds just the `ipv6-parse` static library.
CPMAddPackage(
    NAME ipv6-parse
    GITHUB_REPOSITORY jrepp/ipv6-parse
    GIT_TAG dc4808b0ab75a94165a915587e885c14860d4337
    OPTIONS "IPV6_PARSE_LIBRARY_ONLY ON"
)
target_link_libraries(benchmark PRIVATE ipv6-parse)

# VladimirShaleev/ipaddress: header-only modern C++ IPv4/IPv6 library.
CPMAddPackage("gh:VladimirShaleev/ipaddress@1.2.1")
target_link_libraries(benchmark PRIVATE ipaddress::ipaddress)

# Boost.Asio: ip::make_address_v4/v6 (widely used high-level C++ API).
# Only the asio library (and its dependencies) is built.
set(BOOST_INCLUDE_LIBRARIES asio)
set(BOOST_ENABLE_CMAKE ON)
CPMAddPackage(
    NAME Boost
    VERSION 1.89.0
    URL https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.xz
)
target_link_libraries(benchmark PRIVATE Boost::asio)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
