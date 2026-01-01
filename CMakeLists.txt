cmake_minimum_required(VERSION 3.20)
project(rarecell LANGUAGES CXX CUDA)

# ----------------------------------------
# Global language standards & defaults
# ----------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release on single-config generators (Ninja/Make)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------
# MSVC: define macros and scope options to C++ only (NOT CUDA)
# ----------------------------------------
if(MSVC)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
  # IMPORTANT: scope these flags to CXX only, so they do not leak into NVCC
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/MP>
  )
endif()

# ----------------------------------------
# Dependencies
# ----------------------------------------
# MPI
find_package(MPI REQUIRED)

# HDF5 (C + HL). vcpkg provides imported targets or variables depending on version.
find_package(HDF5 REQUIRED COMPONENTS C HL)

# CUDA (runtime)
find_package(CUDAToolkit REQUIRED)

# JSON (header-only, via vcpkg: nlohmann-json)
find_package(nlohmann_json CONFIG REQUIRED)

find_package(igraph CONFIG REQUIRED)


# ----------------------------------------
# Include dirs (project)
# ----------------------------------------
# All targets will be able to include headers from src/
include_directories(${CMAKE_SOURCE_DIR}/src)

# ----------------------------------------
# Library: H5AD reader
# ----------------------------------------
add_library(rarecell_io
  src/io/h5ad_reader.cpp
)

target_include_directories(rarecell_io PUBLIC
  ${HDF5_INCLUDE_DIRS}
)

# Link to MPI + HDF5 (handle both imported targets and legacy vars)
if(TARGET HDF5::HDF5_C)
  target_link_libraries(rarecell_io PUBLIC MPI::MPI_CXX HDF5::HDF5_C HDF5::HDF5_HL)
elseif(DEFINED HDF5_C_LIBRARIES)
  target_link_libraries(rarecell_io PUBLIC MPI::MPI_CXX ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES})
else()
  target_link_libraries(rarecell_io PUBLIC MPI::MPI_CXX ${HDF5_LIBRARIES})
endif()

# If the found HDF5 is parallel, define this so code uses MPI-IO paths
if(DEFINED HDF5_IS_PARALLEL AND HDF5_IS_PARALLEL)
  target_compile_definitions(rarecell_io PUBLIC H5_HAVE_PARALLEL=1)
endif()

# ----------------------------------------
# Library: config loader (nlohmann_json)
# ----------------------------------------
add_library(rarecell_cfg
  src/config/filter_config.cpp
)
target_link_libraries(rarecell_cfg PUBLIC nlohmann_json::nlohmann_json)

# ----------------------------------------
# Library: CUDA kernels + filter engine
# ----------------------------------------
add_library(rarecell_filter
  src/filter/filter.cpp
  src/cuda/filter_kernels.cu
)

# Make sure we link CUDA runtime and our other libs
target_link_libraries(rarecell_filter PUBLIC
  CUDA::cudart
  MPI::MPI_CXX
  rarecell_io
  rarecell_cfg
)

# Enable separable compilation (device linking) for CUDA
set_target_properties(rarecell_filter PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Choose CUDA architectures (adjust to your GPUs; you can also omit to use NVCC defaults)
# Examples: "75" (Turing), "80;86" (Ampere), "90" (Hopper).
# set_target_properties(rarecell_filter PROPERTIES CUDA_ARCHITECTURES "75")

# NOTE: Do NOT add raw MSVC /flags at target-level for CUDA.
# If you must pass host flags for CUDA files, always wrap with -Xcompiler, e.g.:
# target_compile_options(rarecell_filter PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
# )

# ----------------------------------------
# Executable
# ----------------------------------------
add_executable(rarecell_reader
  src/main.cpp
)

target_link_libraries(rarecell_reader PUBLIC rarecell_filter)

# ----------------------------------------
# Library: metrics (Fano, Gini, Palma)
# ----------------------------------------
add_library(rarecell_metrics
  src/metrics/fano.cpp
  src/metrics/gini.cpp
  src/metrics/palma.cpp
  src/metrics/lowess.cpp
  src/graph/selection.cpp
  src/cuda/metrics_cuda.cu
  src/graph/binarize.cpp
  src/graph/knn.cpp
  src/graph/mix.cpp
  src/cluster/leiden.cpp
  src/refine/refine_pca_cosine.cpp
)
target_include_directories(rarecell_metrics PUBLIC src)
target_link_libraries(rarecell_metrics PUBLIC MPI::MPI_CXX rarecell_filter)

# Link metrics into the app
target_link_libraries(rarecell_reader PUBLIC rarecell_metrics)

target_link_libraries(rarecell_metrics PRIVATE igraph::igraph)


# ----------------------------------------
# (Optional) Install rules
# ----------------------------------------
# install(TARGETS rarecell_reader RUNTIME DESTINATION bin)

# ----------------------------------------
# Summary
# ----------------------------------------
message(STATUS "==== rarecell build summary ====")
message(STATUS "C++ compiler:       ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA toolkit:       ${CUDAToolkit_VERSION}")
message(STATUS "MPI CXX compiler:   ${MPI_CXX_COMPILER}")
message(STATUS "HDF5 include dirs:  ${HDF5_INCLUDE_DIRS}")
if(DEFINED HDF5_IS_PARALLEL)
  message(STATUS "HDF5 parallel I/O:  ${HDF5_IS_PARALLEL}")
endif()

