# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pyfaiss
  DESCRIPTION "Python bindings for faiss."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

macro(configure_swigfaiss source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
  if(FAISS_ENABLE_GPU)
    set_source_files_properties(${source} PROPERTIES
      COMPILE_DEFINITIONS GPU_WRAPPER
    )
    if (FAISS_ENABLE_ROCM)
      set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS FAISS_ENABLE_ROCM
      )
    endif()
    if (FAISS_ENABLE_CUVS)
      set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS FAISS_ENABLE_CUVS
      )
    endif()
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
if(FAISS_BUILD_AVX_VERSIONS)
  configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx2.swig COPYONLY)
  configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx512.swig COPYONLY)
  configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx512_spr.swig COPYONLY)
endif()
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_sve.swig COPYONLY)

configure_swigfaiss(swigfaiss.swig)
if(FAISS_BUILD_AVX_VERSIONS)
  configure_swigfaiss(swigfaiss_avx2.swig)
  configure_swigfaiss(swigfaiss_avx512.swig)
  configure_swigfaiss(swigfaiss_avx512_spr.swig)
endif()
configure_swigfaiss(swigfaiss_sve.swig)
configure_swigfaiss(faiss_example_external_module.swig)

if(TARGET faiss)
  # Manually add headers as extra dependencies of swigfaiss.
  set(SWIG_MODULE_swigfaiss_EXTRA_DEPS)
  foreach(h ${FAISS_HEADERS})
    list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    if(FAISS_BUILD_AVX_VERSIONS)
      list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/${h}")
      list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/${h}")
      list(APPEND SWIG_MODULE_swigfaiss_avx512_spr_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/${h}")
    endif()
    list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_faiss_example_external_module_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
  endforeach()
  if(FAISS_ENABLE_ROCM)
    foreach(h ${FAISS_GPU_HEADERS})
      list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
      if(FAISS_BUILD_AVX_VERSIONS)
        list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
          "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
        list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
          "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
        list(APPEND SWIG_MODULE_swigfaiss_avx512_spr_EXTRA_DEPS
          "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
      endif()
      list(APPEND SWIG_MODULE_faiss_example_external_module_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
    endforeach()
  else()
    foreach(h ${FAISS_GPU_HEADERS})
      list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
      if(FAISS_BUILD_AVX_VERSIONS)
        list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
          "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
        list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
          "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
        list(APPEND SWIG_MODULE_swigfaiss_avx512_spr_EXTRA_DEPS
          "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
      endif()
      list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
      list(APPEND SWIG_MODULE_faiss_example_external_module_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    endforeach()
  endif()
else()
  find_package(faiss REQUIRED)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "AIX")
swig_add_library(swigfaiss
  TYPE MODULE
  LANGUAGE python
  SOURCES swigfaiss.swig
)
else ()
swig_add_library(swigfaiss
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss.swig
)
endif()

set_property(TARGET swigfaiss PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

if(FAISS_BUILD_AVX_VERSIONS)
  set_property(SOURCE swigfaiss_avx2.swig
    PROPERTY SWIG_MODULE_NAME swigfaiss_avx2)
  swig_add_library(swigfaiss_avx2
    TYPE SHARED
    LANGUAGE python
    SOURCES swigfaiss_avx2.swig
  )
  set_property(TARGET swigfaiss_avx2 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
  if(NOT FAISS_OPT_LEVEL STREQUAL "avx2")
    set_target_properties(swigfaiss_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()

  set_property(SOURCE swigfaiss_avx512.swig
    PROPERTY SWIG_MODULE_NAME swigfaiss_avx512)
  swig_add_library(swigfaiss_avx512
    TYPE SHARED
    LANGUAGE python
    SOURCES swigfaiss_avx512.swig
  )
  set_property(TARGET swigfaiss_avx512 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
  if(NOT FAISS_OPT_LEVEL STREQUAL "avx512")
    set_target_properties(swigfaiss_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()

  set_property(SOURCE swigfaiss_avx512_spr.swig
    PROPERTY SWIG_MODULE_NAME swigfaiss_avx512_spr)
  swig_add_library(swigfaiss_avx512_spr
    TYPE SHARED
    LANGUAGE python
    SOURCES swigfaiss_avx512_spr.swig
  )
  set_property(TARGET swigfaiss_avx512_spr PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
  if(NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
    set_target_properties(swigfaiss_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endif() # FAISS_BUILD_AVX_VERSIONS

set_property(SOURCE swigfaiss_sve.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_sve)
swig_add_library(swigfaiss_sve
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_sve.swig
)
set_property(TARGET swigfaiss_sve PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(swigfaiss_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE faiss_example_external_module.swig
  PROPERTY SWIG_MODULE_NAME faiss_example_external_module)
swig_add_library(faiss_example_external_module
  TYPE SHARED
  LANGUAGE python
  SOURCES faiss_example_external_module.swig
)
set_property(TARGET faiss_example_external_module PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

if(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swigfaiss PROPERTIES SUFFIX .so)
  if(FAISS_BUILD_AVX_VERSIONS)
    set_target_properties(swigfaiss_avx2 PROPERTIES SUFFIX .so)
    set_target_properties(swigfaiss_avx512 PROPERTIES SUFFIX .so)
    set_target_properties(swigfaiss_avx512_spr PROPERTIES SUFFIX .so)
  endif()
  set_target_properties(swigfaiss_sve PROPERTIES SUFFIX .so)
  set_target_properties(faiss_example_external_module PROPERTIES SUFFIX .so)
else()
  # we need bigobj for the swig wrapper
  target_compile_options(swigfaiss PRIVATE /bigobj)
  if(FAISS_BUILD_AVX_VERSIONS)
    target_compile_options(swigfaiss_avx2 PRIVATE /bigobj)
    target_compile_options(swigfaiss_avx512 PRIVATE /bigobj)
    target_compile_options(swigfaiss_avx512_spr PRIVATE /bigobj)
  endif()
  target_compile_options(swigfaiss_sve PRIVATE /bigobj)
  target_compile_options(faiss_example_external_module PRIVATE /bigobj)
endif()

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    target_link_libraries(swigfaiss PRIVATE hip::host)
    if(FAISS_BUILD_AVX_VERSIONS)
      target_link_libraries(swigfaiss_avx2 PRIVATE hip::host)
      target_link_libraries(swigfaiss_avx512 PRIVATE hip::host)
      target_link_libraries(swigfaiss_avx512_spr PRIVATE hip::host)
    endif()
    target_link_libraries(faiss_example_external_module PRIVATE hip::host)
  else()
    find_package(CUDAToolkit REQUIRED)
    if(FAISS_ENABLE_CUVS)
      find_package(cuvs)
    endif()
    target_link_libraries(swigfaiss PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    if(FAISS_BUILD_AVX_VERSIONS)
      target_link_libraries(swigfaiss_avx2 PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
      target_link_libraries(swigfaiss_avx512 PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
      target_link_libraries(swigfaiss_avx512_spr PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    endif()
    target_link_libraries(swigfaiss_sve PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  endif()
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swigfaiss PRIVATE
  faiss
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

if(FAISS_BUILD_AVX_VERSIONS)
  target_link_libraries(swigfaiss_avx2 PRIVATE
    faiss_avx2
    Python::Module
    Python::NumPy
    OpenMP::OpenMP_CXX
  )

  target_link_libraries(swigfaiss_avx512 PRIVATE
    faiss_avx512
    Python::Module
    Python::NumPy
    OpenMP::OpenMP_CXX
  )

  target_link_libraries(swigfaiss_avx512_spr PRIVATE
    faiss_avx512_spr
    Python::Module
    Python::NumPy
    OpenMP::OpenMP_CXX
  )
endif()

target_link_libraries(swigfaiss_sve PRIVATE
  faiss_sve
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(faiss_example_external_module PRIVATE
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
  swigfaiss
  faiss
)

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(swigfaiss PRIVATE ${PROJECT_SOURCE_DIR}/../..)
if(FAISS_BUILD_AVX_VERSIONS)
  target_include_directories(swigfaiss_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
  target_include_directories(swigfaiss_avx512 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
  target_include_directories(swigfaiss_avx512_spr PRIVATE ${PROJECT_SOURCE_DIR}/../..)
endif()
target_include_directories(swigfaiss_sve PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(faiss_example_external_module PRIVATE ${PROJECT_SOURCE_DIR}/../..)

# Try to find Python in different ways for better compatibility
# First check if Python was already found by parent CMakeLists.txt
message(STATUS "Faiss Python CMake: Python_FOUND=${Python_FOUND}")
message(STATUS "Faiss Python CMake: Python_EXECUTABLE=${Python_EXECUTABLE}")
message(STATUS "Faiss Python CMake: Python_INCLUDE_DIRS=${Python_INCLUDE_DIRS}")
message(STATUS "Faiss Python CMake: Python_NumPy_INCLUDE_DIRS=${Python_NumPy_INCLUDE_DIRS}")

if(NOT Python_FOUND)
  find_package(Python COMPONENTS Interpreter Development.Module NumPy)
endif()
if(NOT Python_FOUND)
  find_package(Python3 COMPONENTS Interpreter Development.Module NumPy)
  if(Python3_FOUND)
    set(Python_FOUND ${Python3_FOUND})
    set(Python_VERSION ${Python3_VERSION})
    set(Python_EXECUTABLE ${Python3_EXECUTABLE})
    set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(Python_LIBRARIES ${Python3_LIBRARIES})
    set(Python_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS})
  endif()
endif()

if(NOT Python_FOUND)
  message(FATAL_ERROR "Could not find Python with NumPy")
endif()

add_library(faiss_python_callbacks EXCLUDE_FROM_ALL
  python_callbacks.cpp
)
set_property(TARGET faiss_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

if (${CMAKE_SYSTEM_NAME} MATCHES "AIX")
target_link_libraries(faiss_python_callbacks PRIVATE faiss)
endif()

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(faiss_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(faiss_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(swigfaiss PRIVATE faiss_python_callbacks)
if(FAISS_BUILD_AVX_VERSIONS)
  target_link_libraries(swigfaiss_avx2 PRIVATE faiss_python_callbacks)
  target_link_libraries(swigfaiss_avx512 PRIVATE faiss_python_callbacks)
  target_link_libraries(swigfaiss_avx512_spr PRIVATE faiss_python_callbacks)
endif()
target_link_libraries(swigfaiss_sve PRIVATE faiss_python_callbacks)
target_link_libraries(faiss_example_external_module PRIVATE faiss_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)
configure_file(class_wrappers.py class_wrappers.py COPYONLY)
configure_file(gpu_wrappers.py gpu_wrappers.py COPYONLY)
configure_file(extra_wrappers.py extra_wrappers.py COPYONLY)
configure_file(array_conversions.py array_conversions.py COPYONLY)

# file(GLOB files "${PROJECT_SOURCE_DIR}/../../contrib/*.py")
file(COPY ${PROJECT_SOURCE_DIR}/../../contrib  DESTINATION .)

# ====================================================================
# =========               BEGIN LEANN PROJECT PATCH                =========
# ====================================================================
# Following diskann approach, install all Python components as sibling modules in the main package directory

# 1. Install C++ extension, target is main package root directory leann_backend_hnsw
install(TARGETS swigfaiss
    DESTINATION leann_backend_hnsw
)

# 2. Install SWIG wrapper, rename to faiss.py, target is main package root directory
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/swigfaiss.py
    DESTINATION leann_backend_hnsw
    RENAME faiss.py
)

# 3. Install all static helper .py files, target is main package root directory
#    Note: We no longer need faiss's own __init__.py file
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/loader.py
    ${CMAKE_CURRENT_SOURCE_DIR}/class_wrappers.py
    ${CMAKE_CURRENT_SOURCE_DIR}/array_conversions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/extra_wrappers.py
    DESTINATION leann_backend_hnsw
)

# ====================================================================
# =========                END LEANN PROJECT PATCH                 =========
# ====================================================================