# GEN2D CMakeLists.txt file
cmake_minimum_required(VERSION 2.8)
#
# Reset some options BEFORE declaring project
#
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Project Build Type")
project(GEN2D)
mark_as_advanced(FORCE CMAKE_INSTALL_PREFIX)
option(GEN2D_DEBUG "Debugging Mode and Flags" OFF)

#
# Include Directories
#
set(INCLUDE_DIRS ${GEN2D_SOURCE_DIR}/include)

#
# Library: gen2d
#
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/interface/pybind11")
file(GLOB sources "${GEN2D_SOURCE_DIR}/src/*.cpp" "${GEN2D_SOURCE_DIR}/src/*.c" "${GEN2D_SOURCE_DIR}/interface/*.cpp")
pybind11_add_module(libgen2d ${sources})
set_target_properties(libgen2d PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${GEN2D_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${GEN2D_BINARY_DIR}/lib
  )
target_include_directories(libgen2d PUBLIC ${INCLUDE_DIRS})
