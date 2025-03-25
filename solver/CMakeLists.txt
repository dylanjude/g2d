#
# G2D CMakeLists.txt file
#
cmake_minimum_required(VERSION 3.17)
#
# Reset some options BEFORE declaring project
#
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Project Build Type")

project(G2D)

enable_language(CUDA C CXX)

option(G2D_DEBUG "Debugging Mode and Flags" OFF)
option(G2D_PYTHON "Look for Python" ON)
# 
if(${CMAKE_VERSION} VERSION_GREATER "3.11")
  cmake_policy(SET CMP0074 NEW)
endif()

# -----------------------------------------------------------------
# CMake Setup
#

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DEBUG_OPTS}" )
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DEBUG_OPTS}" )
SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${DEBUG_OPTS}" )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
# ----------------------------------------------------------
# Python
if(G2D_PYTHON)
  find_package(PythonInterp 3)
  find_package(PythonLibs 3)
  if(NOT ${PYTHONLIBS_FOUND})
    find_package(PythonInterp 2.7 REQUIRED)
    find_package(PythonLibs 2.7 REQUIRED)
  endif()
endif(G2D_PYTHON)


# -----------------------------------------------------
# CUDA
set(CMAKE_CUDA_FLAGS "-lineinfo -rdynamic --Wno-deprecated-gpu-targets -m64")
if(${CMAKE_VERSION} VERSION_GREATER "3.18")
  set(CMAKE_CUDA_ARCHITECTURES 80 70)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")
endif()
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_EXTENSIONS OFF)
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# ------------------------------------------------------
# Include Directories
set(INCLUDE_DIRS 
  ${G2D_SOURCE_DIR}/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ----------------------------------------------------------------
# Main Library (GPU + CPU)
#
# Note: need to link against CPU lib to get fortran functions. Fortran
# functions are only really used for setup and for debugging.
#
file(GLOB sources "${G2D_SOURCE_DIR}/src/*.cpp" "${G2D_SOURCE_DIR}/src/*.cu")

add_library(g2d_common STATIC ${sources})
target_include_directories(g2d_common PRIVATE ${INCLUDE_DIRS})
target_compile_definitions(g2d_common PRIVATE ${G2D_DEFINITIONS})
target_link_libraries(g2d_common PUBLIC ${CUDART_LIBRARY})
# set_property(TARGET lib1 PROPERTY 
set_target_properties(g2d_common PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY ${G2D_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${G2D_BINARY_DIR}/lib
  PREFIX "" # no lib prefix
  )


# ------------------------------------------------------------------
# Executable binary
#
file(GLOB exe_srcs "${G2D_SOURCE_DIR}/driver/*.cpp")

add_executable(g2d.exe ${exe_srcs})
target_include_directories(g2d.exe PUBLIC ${INCLUDE_DIRS})
target_link_libraries(g2d.exe g2d_common)
set_target_properties(g2d.exe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  PREFIX "" # no lib prefix
  )

# ------------------------------------------------------------------
# Python-wrapped Library
#
if(G2D_PYTHON)
  file(GLOB py_sources "${G2D_SOURCE_DIR}/interface/*.cpp")
  pybind11_add_module(garfoil ${py_sources})
  target_include_directories(garfoil PUBLIC ${INCLUDE_DIRS})
  target_link_libraries(garfoil PRIVATE g2d_common)
  set_target_properties(garfoil PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    PREFIX "" # no lib prefix
    )
endif()

# file(GLOB py_sources "${G2D_SOURCE_DIR}/python/*.py")
# foreach(pypath ${py_sources})
#   get_filename_component(pyname ${pypath} NAME)
#   configure_file(${pypath} ${G2D_BINARY_DIR}/lib/${pyname})
# endforeach(pypath)






