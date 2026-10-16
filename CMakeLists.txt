cmake_minimum_required(VERSION 3.16)
project(PathoVoice LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PATHOVOICE_BUILD_TESTS "Build PathoVoice tests" ON)
option(PATHOVOICE_BUILD_TOOLS "Build PathoVoice command line tools" ON)
