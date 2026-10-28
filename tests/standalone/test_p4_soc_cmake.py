# SPDX-License-Identifier: Apache-2.0
"""Exercise the real CMake SoC gate without a compiler, CANN or torch."""

# Standard
from pathlib import Path
import shutil
import subprocess

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")


def run_soc_gate(tmp_path: Path, soc: str | None) -> subprocess.CompletedProcess[str]:
    """Run the production gate and check its canonical downstream SoC value."""
    if CMAKE is None:
        pytest.fail(
            "cmake is required for the P4 SoC gate regression (script mode only)"
        )
    script = tmp_path / "soc_probe.cmake"
    gate = (ROOT / "cmake/ascend_soc.cmake").as_posix()
    script.write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        f'include("{gate}")\n'
        'if(NOT SOC_VERSION STREQUAL "Ascend910B3")\n'
        '  message(FATAL_ERROR "CANN requires canonical Ascend910B3 spelling")\n'
        "endif()\n"
    )
    options = [] if soc is None else [f"-DSOC_VERSION={soc}"]
    return subprocess.run(
        [CMAKE, *options, "-P", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    "soc", ["Ascend910B3", "ascend910b3", "ASCEND910B3", "aScEnD910b3"]
)
def test_910b3_spelling_is_canonicalized(tmp_path: Path, soc: str) -> None:
    """Accept the Python builder's value and case aliases, not other devices."""
    result = run_soc_gate(tmp_path, soc)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "soc",
    [
        None,
        "",
        "Ascend910B",
        "Ascend910B2",
        "Ascend910B4",
        "Ascend910_9391",
        "Ascend310P3",
        "Ascend950",
        "Ascend910B3;Ascend910B2",
        "Ascend910B3-extra",
    ],
)
def test_unsupported_soc_is_rejected(tmp_path: Path, soc: str | None) -> None:
    """Missing, ambiguous and unsupported values remain a hard build error."""
    result = run_soc_gate(tmp_path, soc)
    assert result.returncode != 0
    assert "P4 supports only SOC_VERSION=ascend910b3" in result.stdout


def test_gate_precedes_native_subdirectories() -> None:
    """The tested gate is wired into the actual native CMake entry point."""
    source = (ROOT / "cmake/npu_extensions.cmake").read_text()
    assert source.index(
        'include("${CMAKE_CURRENT_LIST_DIR}/ascend_soc.cmake")'
    ) < source.index("add_subdirectory(csrc/graph)")
