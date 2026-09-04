cmake_minimum_required(VERSION 3.22)
project(idr_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── GoogleTest ───────────────────────────────────────────────────────────────
# Automatically fetched if not found locally.
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)   # Windows only, harmless elsewhere
FetchContent_MakeAvailable(googletest)

# ─── Include paths (mirror the Android CMakeLists.txt) ───────────────────────
set(ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ─── Test executable ──────────────────────────────────────────────────────────
add_executable(idr_tests
    test_filters.cpp
)

target_include_directories(idr_tests PRIVATE
    ${ENGINE_ROOT}          # allow: #include "core/types.h"
    ${ENGINE_ROOT}/core
    ${ENGINE_ROOT}/filters
)

target_link_libraries(idr_tests
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(idr_tests PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# ─── Register with CTest ──────────────────────────────────────────────────────
include(GoogleTest)
gtest_discover_tests(idr_tests)
