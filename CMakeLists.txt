cmake_minimum_required(VERSION 3.22)
project(alphazero_chess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/O2 /EHsc)
else()
  add_compile_options(-O3 -Wall -Wextra)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(cpp)
