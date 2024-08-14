cmake_minimum_required(VERSION 3.14)
project(Annoy
        DESCRIPTION "Approximate Nearest Neighbors Oh Yeah"
        VERSION 1.1.1
        LANGUAGES CXX
)

add_library(Annoy INTERFACE)

set(ANNOY_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/annoy)
file(MAKE_DIRECTORY ${ANNOY_INCLUDE_DIR})

target_include_directories(Annoy INTERFACE src/)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Add Python set-up code here.
endif()

add_subdirectory(pybind11)
# or
# find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(module src/annoymodule.cpp)
#install(TARGET module DESTINATION .)
target_compile_definitions(module
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})