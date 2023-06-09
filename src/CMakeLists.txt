﻿cmake_minimum_required (VERSION 3.20 FATAL_ERROR)
project(CVRemap LANGUAGES CUDA CXX VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# 7.5 is 20 series, 8.6 is 30 series
set(CMAKE_CUDA_ARCHITECTURES "75;86")

enable_language(CUDA)
find_package(CUDAToolkit)

# Fix error: Target "CVRemap" requires the language dialect "CUDA20" (with compiler
#  extensions).  But the current compiler "NVIDIA" does not support this, or
#  CMake does not know the flags to enable it.
set(CMAKE_CUDA_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()

set(CMAKE_CUDA_FLAGS "-std=c++20")

# other dependencies
find_package(fmt CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(SOURCE_FILES

	avxRemapPoly.h
	avxRemapPoly.cpp
	brownConradyModel.h
	brownConradyModel.cpp
	cudaAddKernel.h
	cudaAddKernel.cu
	cudaRemapPoly.h
	cudaRemapPoly.cu
	cudaUtil.h
	cudaUtil.cu
	poly.h
	poly.cpp
	remap.h
	remap.cpp

	main.cpp
)

add_executable (cvremap ${SOURCE_FILES})
target_include_directories(cvremap PUBLIC ".")
target_link_libraries(cvremap PUBLIC ${OpenCV_LIBS} fmt::fmt CUDA::cudart_static CppOpenCVUtilLib Eigen3::Eigen)
