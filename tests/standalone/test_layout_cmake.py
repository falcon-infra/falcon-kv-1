# SPDX-License-Identifier: Apache-2.0
"""Parse real native CMake entries with compiler/SDK commands replaced by stubs."""

# Standard
from pathlib import Path
import subprocess
import tomllib

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["name"]

HARNESS = r"""
cmake_minimum_required(VERSION 3.26.1)
set(CMAKE_CURRENT_SOURCE_DIR "${TEST_ROOT}")
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
set(CMAKE_INSTALL_PREFIX "${TEST_INSTALL}")
set(P1_HOST_INSTALL_DIR "${TEST_INSTALL}")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(GLIBCXX_USE_CXX11_ABI 1)
set(BUILD_MOONCAKE OFF)
function(project)
  get_property(seen GLOBAL PROPERTY LAYOUT_PROJECT)
  if(seen)
    message(FATAL_ERROR "A second project entry was retained")
  endif()
  set_property(GLOBAL PROPERTY LAYOUT_PROJECT "${ARGV0}")
endfunction()
function(find_package)
endfunction()
function(execute_process)
  message(FATAL_ERROR "No subprocess/compiler/device probe allowed")
endfunction()
function(enable_language)
  message(FATAL_ERROR "No compiler probe allowed")
endfunction()
function(append_cmake_prefix_path)
endfunction()
function(run_python out)
  set(${out} "2.9.0" PARENT_SCOPE)
endfunction()
macro(include path)
  if("${path}" MATCHES "(^|/)utils.cmake$")
    if(NOT EXISTS "${path}" AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
      message(FATAL_ERROR "Missing moved CMake helper ${path}")
    endif()
  elseif("${path}" MATCHES "/ascendc.cmake$")
    # The CANN device compiler is deliberately outside this host test.
  else()
    _include("${path}")
  endif()
endmacro()
function(add_subdirectory path)
  if("${path}" STREQUAL "csrc/graph")
    set_property(GLOBAL APPEND PROPERTY LAYOUT_TARGETS cache_kernels)
    return()
  endif()
  set(CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
  include("${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
endfunction()
function(check_sources target)
  if(NOT ARGN)
    message(FATAL_ERROR "Empty source list for ${target}")
  endif()
  foreach(source IN LISTS ARGN)
    if(NOT IS_ABSOLUTE "${source}")
      set(source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    endif()
    if(NOT EXISTS "${source}")
      message(FATAL_ERROR "Source path broken: ${source}")
    endif()
  endforeach()
  set_property(GLOBAL APPEND PROPERTY LAYOUT_TARGETS "${target}")
endfunction()
function(pybind11_add_module target)
  check_sources("${target}" ${ARGN})
endfunction()
function(ascendc_library target kind)
  check_sources("${target}" ${ARGN})
endfunction()
@STUBS@
include("${TEST_ROOT}/CMakeLists.txt")
get_property(targets GLOBAL PROPERTY LAYOUT_TARGETS)
message(STATUS "LAYOUT_TARGETS=${targets}")
"""


@pytest.mark.parametrize("hixl,hcomm", [("ON", "ON"), ("OFF", "OFF")])
def test_native_entry_resolves_moved_sources_without_compiling(
    tmp_path: Path, hixl: str, hcomm: str
) -> None:
    cann = tmp_path / "fake-cann/tools/tikcpp/ascendc_kernel_cmake"
    cann.mkdir(parents=True)
    stubs = "\n".join(
        f"function({name})\nendfunction()"
        for name in (
            "add_compile_definitions",
            "include_directories",
            "target_link_options",
            "target_link_directories",
            "target_link_libraries",
            "target_include_directories",
            "target_compile_definitions",
            "target_compile_features",
            "install",
            "set_target_properties",
        )
    )
    harness = tmp_path / "entry.cmake"
    harness.write_text(HARNESS.replace("@STUBS@", stubs))
    result = subprocess.run(
        [
            "cmake",
            f"-DTEST_ROOT={ROOT}",
            f"-DTEST_INSTALL={tmp_path / 'install'}",
            f"-DASCEND_HOME_PATH={tmp_path / 'fake-cann'}",
            f"-DUSE_HIXL={hixl}",
            f"-DUSE_HCOMM_ONESIDED={hcomm}",
            "-DSOC_VERSION=" + ("ascend910b3" if PRIMARY == "vllm" else "Ascend910B3"),
            "-P",
            str(harness),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = (
        {"_ascend_C", "vllm_ascend_kernels"}
        if PRIMARY == "vllm"
        else {
            "c_ops",
            "cache_kernels",
            "native_storage_ops",
            "lmcache_fs",
            "lmcache_redis",
            "hixl_npu_comms" if hixl == "ON" else "hccl_npu_comms",
        }
    )
    if PRIMARY == "lmcache" and hcomm == "ON":
        expected.add("hcomm_onesided")
    line = next(
        line for line in result.stdout.splitlines() if "LAYOUT_TARGETS=" in line
    )
    assert set(line.split("LAYOUT_TARGETS=", 1)[1].split(";")) == expected
    assert not (tmp_path / "CMakeCache.txt").exists()
    assert not (tmp_path / "CMakeFiles").exists()
