cmake_minimum_required(VERSION 3.22.1)
project("dr_nav")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Engine source files ──────────────────────────────────────────────────────
# All filter implementations are header-only; only the JNI bridge needs compiling.
add_library(${CMAKE_PROJECT_NAME} SHARED
    native-lib.cpp
)

# ─── Include directories ──────────────────────────────────────────────────────
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}          # so #include "core/types.h" works
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/filters
)

# ─── Compiler flags ───────────────────────────────────────────────────────────
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    -O2
    -Wall
    -Wno-unused-parameter
)

# ─── Link Android system libraries ────────────────────────────────────────────
target_link_libraries(${CMAKE_PROJECT_NAME}
    android
    log
)