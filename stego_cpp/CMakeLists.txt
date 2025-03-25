cmake_minimum_required(VERSION 3.14)
project(stego_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add executable
add_executable(stego_app 
    src/main.cpp 
    src/stego.cpp
    src/gpt2_model.cpp
)

# Set optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Copy the executable to the bin directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Installation
install(TARGETS stego_app DESTINATION bin)
install(FILES README.md DESTINATION .)