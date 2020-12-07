#
# G2D CMakeLists.txt file
#
cmake_minimum_required(VERSION 2.8)
#
# Reset some options BEFORE declaring project
#
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Project Build Type")

project(G2D)

option(G2D_DEBUG "Debugging Mode and Flags" OFF)
option(G2D_PYTHON "Look for Python" OFF)
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
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print( numpy.get_include() )"
    OUTPUT_VARIABLE NUMPY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "numpy: ${NUMPY_INCLUDE}")
  if(${PYTHON_VERSION_STRING} VERSION_LESS "3")
    set(PYTHON_2_COMPATIBILITY "\nfrom __future__ import absolute_import, division, print_function")
  else()
    set(PYTHON_2_COMPATIBILITY "# Using Python 3")
  endif()
endif(G2D_PYTHON)
# ------------------------------------------------------
# Include Directories
set(INCLUDE_DIRS ${G2D_SOURCE_DIR}/src)

# -----------------------------------------------------
# CUDA
find_package(CUDA REQUIRED)
if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
  list(APPEND CUDA_OPTS "-gencode=arch=compute_60,code=sm_60")
  list(APPEND CUDA_OPTS "-gencode=arch=compute_70,code=sm_70")
elseif((CUDA_VERSION_MAJOR EQUAL 8) OR (CUDA_VERSION_MAJOR GREATER 8))
  list(APPEND CUDA_OPTS "-gencode=arch=compute_60,code=sm_60")
elseif((CUDA_VERSION_MAJOR EQUAL 7) OR (CUDA_VERSION_MAJOR GREATER 7))
  set(CUDA_OPTS "-arch=sm_30")
  list(APPEND CUDA_OPTS "-gencode=arch=compute_30,code=sm_30")
else()
  set(CUDA_OPTS "-arch=sm_20")
  list(APPEND CUDA_OPTS "-gencode=arch=compute_20,code=sm_20")
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF CACHE INTERNAL "Propagate host flags")
list(APPEND CUDA_OPTS "-std=c++11")
# force use of shared cuda runtime lib (required for Vulcanite cluster)
find_library(CUDA_SHARED_RT libcudart.so PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" NO_DEFAULT_PATH)
list(REMOVE_ITEM CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a")
list(APPEND CUDA_LIBRARIES "${CUDA_SHARED_RT}")

message(STATUS "CUDA target : ${CUDA_OPTS}")
message(STATUS "CUDA libs   : ${CUDA_LIBRARIES}")
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
# 
if(G2D_DEBUG)
  set(OLEVEL "-O0")
  list(APPEND G2D_DEFINITIONS "-DDEBUG_MODE")
  set(DEBUG_OPTS "-g")
  list(APPEND CUDA_OPTS "${OLEVEL}")
  list(APPEND CUDA_OPTS "-DDEBUG_MODE")
  list(APPEND CUDA_OPTS "-g")
  list(APPEND CARTG_DEFINITIONS "-DDEBUG_MODE")
  message(STATUS "Debug Flags are ON")
else(G2D_DEBUG)
  list(APPEND CUDA_OPTS ${OLEVEL})
  set(OLEVEL "-O3")
  message(STATUS "Debug Flags are OFF")
endif(G2D_DEBUG)

# ----------------------------------------------------------------
# Main Library (GPU + CPU)
#
# Note: need to link against CPU lib to get fortran functions. Fortran
# functions are only really used for setup and for debugging.
#
file(GLOB sources "${G2D_SOURCE_DIR}/src/*.cpp" "${G2D_SOURCE_DIR}/src/*.cu")

list(APPEND CUDA_OPTS --compiler-options -fPIC)

cuda_include_directories(${INCLUDE_DIRS})
cuda_add_library(g2d_common SHARED ${sources} OPTIONS ${CUDA_OPTS} ${G2D_DEFINITIONS})
# target_link_libraries(g2d_common ${CUDA_CUBLAS_LIBRARIES})
target_include_directories(g2d_common PUBLIC ${INCLUDE_DIRS})
target_compile_definitions(g2d_common PRIVATE ${G2D_DEFINITIONS})
set_target_properties(g2d_common PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${G2D_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${G2D_BINARY_DIR}/lib
  PREFIX "" # no lib prefix
  )


# ------------------------------------------------------------------
# Executable binary
#
file(GLOB exe_srcs "${G2D_SOURCE_DIR}/driver/*.cpp")

add_executable(g2d ${exe_srcs})
target_include_directories(g2d PUBLIC ${INCLUDE_DIRS})
target_link_libraries(g2d g2d_common)
set_target_properties(g2d PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${G2D_BINARY_DIR}/bin
  PREFIX "" # no lib prefix
  )

# ------------------------------------------------------------------
# Python-wrapped Library
#
if(G2D_PYTHON)
  file(GLOB py_sources "${G2D_SOURCE_DIR}/interface/pyb/*.cpp")
  add_subdirectory("${G2D_SOURCE_DIR}/interface/pyb/pybind11")
  pybind11_add_module(g2d ${py_sources})
  target_include_directories(g2d PUBLIC ${INCLUDE_DIRS})
  target_link_libraries(g2d PRIVATE g2d_common ${MPI_LIBRARIES})
  set_target_properties(g2d PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${G2D_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${G2D_BINARY_DIR}/lib
    PREFIX "" # no lib prefix
    )
endif()

file(GLOB py_sources "${G2D_SOURCE_DIR}/python/*.py")
foreach(pypath ${py_sources})
  get_filename_component(pyname ${pypath} NAME)
  configure_file(${pypath} ${G2D_BINARY_DIR}/lib/${pyname})
endforeach(pypath)






