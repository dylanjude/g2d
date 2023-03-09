
cmake_minimum_required(VERSION 3.17)

project(2dtools)

find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)

if(DEFINED ENV{PYBIND11_SRC})          
  message(STATUS "Adding pybind11 source from ptools")                 
  add_subdirectory($ENV{PYBIND11_SRC} interface/pyb/pybind11)          
else()                 
  message(STATUS "Adding pybind11 source from Python installation")    
  execute_process(COMMAND ${Python_EXECUTABLE} -c "import pybind11; print( pybind11.get_cmake_dir() )"                 
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  cmake_policy(SET CMP0057 NEW)        
  find_package(pybind11 REQUIRED)
endif()

add_subdirectory(solver)
add_subdirectory(meshgen)