import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "train_single_node.sh"


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    capture_path = tmp_path / "calls.txt"
    python_path = tmp_path / "python"
    python_path.write_text(
        """#!/usr/bin/env bash
{
  printf '%s\\n' __CALL__
  printf '%s\\n' "$@"
} >> "$DFACTORY_LAUNCH_CAPTURE"
if [[ "${1:-}" == "-c" ]]; then
  printf '%s\\n' "${DFACTORY_FAKE_CUDA_DEVICE_COUNT:-4}"
fi
""",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    return python_path, capture_path


def test_single_node_launcher_validates_then_starts_torchrun(tmp_path: Path) -> None:
    python_path, capture_path = _fake_python(tmp_path)
    config_path = ROOT / "configs" / "recipes" / "sft" / "llada2_mini.yaml"
    environment = os.environ | {
        "DFACTORY_PYTHON": str(python_path),
        "DFACTORY_LAUNCH_CAPTURE": str(capture_path),
        "NPROC_PER_NODE": "4",
    }

    subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            str(config_path),
            "--set",
            "run.max_steps=2",
            "--set",
            "distributed.dp=99",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    calls = [block.splitlines() for block in capture_path.read_text(encoding="utf-8").split("__CALL__\n")[1:]]
    dfactory_args = [
        "--config",
        str(config_path),
        "--set",
        "run.max_steps=2",
        "--set",
        "distributed.dp=99",
        "--set",
        "distributed.dp=4",
    ]
    assert calls == [
        ["-m", "dfactory", "config", "validate", *dfactory_args],
        [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            "dfactory",
            "train",
            *dfactory_args,
        ],
    ]


def test_single_node_launcher_uses_visible_gpu_count(tmp_path: Path) -> None:
    python_path, capture_path = _fake_python(tmp_path)
    config_path = ROOT / "configs" / "recipes" / "sft" / "llada2_mini.yaml"
    environment = os.environ | {
        "DFACTORY_PYTHON": str(python_path),
        "DFACTORY_LAUNCH_CAPTURE": str(capture_path),
        "DFACTORY_FAKE_CUDA_DEVICE_COUNT": "4",
    }
    environment.pop("NPROC_PER_NODE", None)

    subprocess.run(
        ["bash", str(LAUNCHER), str(config_path)],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    calls = [block.splitlines() for block in capture_path.read_text(encoding="utf-8").split("__CALL__\n")[1:]]
    dfactory_args = ["--config", str(config_path), "--set", "distributed.dp=4"]
    assert calls == [
        ["-c", "import torch; print(torch.cuda.device_count())"],
        ["-m", "dfactory", "config", "validate", *dfactory_args],
        [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            "dfactory",
            "train",
            *dfactory_args,
        ],
    ]


def test_single_node_launcher_rejects_a_single_process(tmp_path: Path) -> None:
    python_path, capture_path = _fake_python(tmp_path)
    environment = os.environ | {
        "DFACTORY_PYTHON": str(python_path),
        "DFACTORY_LAUNCH_CAPTURE": str(capture_path),
        "NPROC_PER_NODE": "1",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER), "configs/recipes/sft/llada2_mini.yaml"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "NPROC_PER_NODE must be an integer of at least 2" in result.stderr
    assert not capture_path.exists()
