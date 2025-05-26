# CMake minimum version
cmake_minimum_required(VERSION 3.10)

project(hypre_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Project name and languages used
project(HelloHYPRE C)

# Find MPI package
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_library(M_LIB m)

# Set the path to your HYPRE build (adjust this to match your setup)
set(HYPRE_DIR "~/phd/codes/hypre/src")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add include directories for HYPRE
include_directories(${HYPRE_DIR}/hypre/include)

# Add the HYPRE library directory
link_directories(${HYPRE_DIR}/build)


# Create an executable target
add_executable(hello_hypre src/hello.c)
add_executable(poisson src/poisson.c)
add_executable(diffusion src/diffusion.c)

set_source_files_properties(src/poisson.c PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/diffusion.c PROPERTIES LANGUAGE CXX)


target_link_libraries(hello_hypre HYPRE ${MPI_LIBRARIES} ${M_LIB})
target_link_libraries(poisson HYPRE ${MPI_LIBRARIES} stdc++ ${M_LIB})
target_link_libraries(diffusion HYPRE ${MPI_LIBRARIES} stdc++ ${M_LIB})
