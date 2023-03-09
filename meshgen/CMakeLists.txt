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
file(GLOB sources "${GEN2D_SOURCE_DIR}/src/*.cpp" "${GEN2D_SOURCE_DIR}/src/*.c" "${GEN2D_SOURCE_DIR}/interface/*.cpp")
pybind11_add_module(ogen ${sources})
set_target_properties(ogen PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
target_include_directories(ogen PUBLIC ${INCLUDE_DIRS})
