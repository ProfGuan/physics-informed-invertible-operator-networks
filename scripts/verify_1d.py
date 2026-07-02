"""Lightweight verification for the 1D GI-ION surface-wave example.

This script reuses the retained notebook behavior without invoking training:
load the public 1D data, normalize the model parameters, load the published
GI-ION checkpoint, draw posterior samples for one held-out test case, and
compare stable summary metrics with a saved reference result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from FrEIA.framework import InputNode, Node, OutputNode, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "GI_ION_master" / "GI-INO" / "1D_inverse" / "1D_IND"
MODEL_DATA = EXPERIMENT_DIR / "data" / "model1D.npy"
PHASE_DATA = EXPERIMENT_DIR / "data" / "phase.npy"
CHECKPOINT = EXPERIMENT_DIR / "models" / "GI-INO_model.pt"
REFERENCE_DIR = REPO_ROOT / "reference" / "1d"
REFERENCE_METRICS = REFERENCE_DIR / "reference_metrics.json"
REFERENCE_SUMMARY = REFERENCE_DIR / "reference_summary.npz"

UB = np.array([0.63, 0.79, 0.88, 0.95, 1.05, 1.2, 1.28, 1.36, 1.36, 1.4], dtype=np.float32)
LB = np.array([0.33, 0.43, 0.48, 0.51, 0.55, 0.6, 0.58, 0.56, 0.56, 0.6], dtype=np.float32)

TRAIN_SIZE = 100000
NDIM_X = 10
NDIM_PAD_X = 12
NDIM_Y = 14
NDIM_Z = 22
NDIM_PAD_ZY = 0
N_BLOCKS = 4
HIDDEN_LAYER_SIZE = 256
EXPONENT_CLAMPING = 4.0
ADD_Y_NOISE = 1e-2
INIT_SCALE = 0.001
DEFAULT_SEED = 20260702
DEFAULT_POSTERIOR_SAMPLES = 512
DEFAULT_TOLERANCE = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the 1D GI-ION checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "verify_1d"))
    parser.add_argument("--posterior-samples", type=int, default=DEFAULT_POSTERIOR_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--test-index",
        type=int,
        default=0,
        help="Index within the held-out test split. Default 0 selects global row 100000.",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Write the generated result as reference/1d/reference_*.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"Verification: FAIL\n原因: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"缺少必需文件: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import importlib.metadata as metadata

        for name in ("FrEIA", "matplotlib", "disba", "tqdm"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "not installed"
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        versions["metadata_error"] = str(exc)
    return versions


def normalize_models(models: np.ndarray) -> np.ndarray:
    return 4.0 * (models.astype(np.float32) - (UB + LB) / 2.0) / (UB - LB)


def denormalize_models(models: np.ndarray) -> np.ndarray:
    return models * (UB - LB) / 4.0 + (UB + LB) / 2.0


def subnet_fc(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(c_in, HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, c_out),
    )


def build_model(device: torch.device) -> ReversibleGraphNet:
    nodes = [InputNode(NDIM_X + NDIM_Y + NDIM_PAD_X, name="input")]
    for i in range(N_BLOCKS):
        nodes.append(
            Node(
                nodes[-1].out0,
                GLOWCouplingBlock,
                {"subnet_constructor": subnet_fc, "clamp": EXPONENT_CLAMPING},
                name=f"coupling_{i}",
            )
        )
        nodes.append(
            Node(nodes[-1].out0, PermuteRandom, {"seed": i}, name=f"permute_{i}")
        )
    nodes.append(OutputNode([nodes[-1].out0], name="output"))
    model = ReversibleGraphNet(nodes, verbose=False).to(device)
    for param in model.parameters():
        if param.requires_grad:
            param.data = INIT_SCALE * torch.randn(param.data.shape, device=device)
    return model


def load_checkpoint(model: ReversibleGraphNet, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "net" not in state:
        fail(f"checkpoint 未包含预期的 'net' 权重: {checkpoint_path}")
    model.load_state_dict(state["net"])


def make_figure(output_path: Path, true_model: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    layer = np.arange(1, NDIM_X + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(layer, true_model, marker="o", label="true")
    plt.plot(layer, mean, marker="s", label="posterior mean")
    plt.fill_between(layer, mean - std, mean + std, alpha=0.25, label="+/- 1 std")
    plt.xlabel("Layer index")
    plt.ylabel("Vs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def compare_with_reference(
    metrics: dict[str, Any],
    summary: dict[str, np.ndarray],
    tolerance: float,
) -> tuple[str, dict[str, float]]:
    if not REFERENCE_METRICS.is_file() or not REFERENCE_SUMMARY.is_file():
        return "REFERENCE_MISSING", {}

    ref_metrics = json.loads(REFERENCE_METRICS.read_text(encoding="utf-8"))
    ref_summary_npz = np.load(REFERENCE_SUMMARY)
    diffs = {
        "mse_abs_diff": abs(float(metrics["parameter_mse"]) - float(ref_metrics["parameter_mse"])),
        "posterior_mean_max_abs_diff": float(
            np.max(np.abs(summary["posterior_mean"] - ref_summary_npz["posterior_mean"]))
        ),
        "posterior_std_max_abs_diff": float(
            np.max(np.abs(summary["posterior_std"] - ref_summary_npz["posterior_std"]))
        ),
    }
    passed = all(value <= tolerance for value in diffs.values())
    return ("PASS" if passed else "FAIL"), diffs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=to_jsonable) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()

    for path in (MODEL_DATA, PHASE_DATA, CHECKPOINT):
        require_file(path)

    if args.posterior_samples <= 0:
        fail("--posterior-samples 必须为正整数")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        fail(f"请求了 {args.device}，但当前环境未检测到 CUDA")
    device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    models_raw = np.load(MODEL_DATA)
    phases = np.load(PHASE_DATA).astype(np.float32)
    if models_raw.shape[0] != phases.shape[0]:
        fail("model1D.npy 与 phase.npy 样本数不一致")
    if models_raw.shape[0] <= TRAIN_SIZE:
        fail("数据量不足以构造 notebook 使用的 held-out test split")

    test_size = models_raw.shape[0] - TRAIN_SIZE
    if args.test_index < 0 or args.test_index >= test_size:
        fail(f"--test-index 超出 held-out test split 范围: 0..{test_size - 1}")

    models_norm = normalize_models(models_raw)
    global_index = TRAIN_SIZE + args.test_index
    x0_norm = models_norm[global_index].astype(np.float32)
    y0 = phases[global_index].astype(np.float32)
    true_model = denormalize_models(x0_norm).astype(np.float32)

    model = build_model(device)
    load_checkpoint(model, CHECKPOINT, device)
    model.eval()

    with torch.no_grad():
        y_test = np.tile(y0, (args.posterior_samples, 1)).astype(np.float32)
        y_tensor = torch.from_numpy(y_test).to(device)
        noise = ADD_Y_NOISE * torch.randn(args.posterior_samples, NDIM_Y, device=device)
        y_tensor = y_tensor + noise
        if NDIM_PAD_ZY:
            y_tensor = torch.cat(
                (ADD_Y_NOISE * torch.randn(args.posterior_samples, NDIM_PAD_ZY, device=device), y_tensor),
                dim=1,
            )
        latent = torch.randn(args.posterior_samples, NDIM_Z, device=device)
        inverse_input = torch.cat((latent, y_tensor), dim=1)
        x_noise_pred = model(inverse_input, rev=True)[0]
        posterior_norm = x_noise_pred[:, :NDIM_X].detach().cpu().numpy()

    posterior_samples = denormalize_models(posterior_norm).astype(np.float32)
    posterior_mean = posterior_samples.mean(axis=0)
    posterior_std = posterior_samples.std(axis=0)
    parameter_mse = float(np.mean((posterior_mean - true_model) ** 2))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "result.png"
    make_figure(figure_path, true_model, posterior_mean, posterior_std)

    summary = {
        "posterior_samples": posterior_samples,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "true_model": true_model,
        "selected_global_index": np.array(global_index, dtype=np.int64),
        "selected_test_index": np.array(args.test_index, dtype=np.int64),
        "random_seed": np.array(args.seed, dtype=np.int64),
        "posterior_sample_count": np.array(args.posterior_samples, dtype=np.int64),
    }

    comparison_status, comparison_diffs = compare_with_reference(
        {"parameter_mse": parameter_mse}, summary, args.tolerance
    )
    if args.write_reference:
        comparison_status = "REFERENCE_WRITTEN"

    runtime_seconds = time.perf_counter() - start_time
    metrics: dict[str, Any] = {
        "verification_status": comparison_status,
        "selected_global_index": global_index,
        "selected_test_index": args.test_index,
        "random_seed": args.seed,
        "posterior_sample_count": args.posterior_samples,
        "device": str(device),
        "runtime_seconds": runtime_seconds,
        "checkpoint_loaded": True,
        "checkpoint_filename": str(CHECKPOINT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "model1D_sha256": sha256_file(MODEL_DATA),
        "phase_sha256": sha256_file(PHASE_DATA),
        "parameter_mse": parameter_mse,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "true_model": true_model,
        "comparison_tolerance": args.tolerance,
        "comparison_diffs": comparison_diffs,
        "package_versions": package_versions(),
        "output_files": [
            str((output_dir / "metrics.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            str((output_dir / "posterior_summary.npz").relative_to(REPO_ROOT)).replace("\\", "/"),
            str(figure_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        ],
    }

    np.savez_compressed(output_dir / "posterior_summary.npz", **summary)
    write_json(output_dir / "metrics.json", metrics)

    if args.write_reference:
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(REFERENCE_SUMMARY, **summary)
        write_json(REFERENCE_METRICS, metrics)
        make_figure(REFERENCE_DIR / "reference_result.png", true_model, posterior_mean, posterior_std)

    print(f"Checkpoint loaded: {CHECKPOINT}")
    print(f"Selected held-out test index: {args.test_index} (global row {global_index})")
    print(f"Posterior samples: {args.posterior_samples}")
    print(f"Parameter-space MSE: {parameter_mse:.10g}")
    print(f"Output directory: {output_dir}")

    if comparison_status == "PASS":
        print("Verification: PASS")
        return 0
    if comparison_status == "REFERENCE_WRITTEN":
        print("Verification: REFERENCE_WRITTEN")
        return 0

    if comparison_status == "REFERENCE_MISSING":
        print("Verification: FAIL")
        print(f"原因: 未找到参考结果。先运行: python scripts/verify_1d.py --write-reference")
        return 1

    print("Verification: FAIL")
    print(f"差异: {comparison_diffs}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
