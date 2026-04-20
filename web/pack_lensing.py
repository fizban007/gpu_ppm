#!/usr/bin/env python3
"""Pack tables/lensing/<setting>/*.txt into a single float32 binary + JSON header
for the WebGPU visualizer.

Output:
    web/public/lensing.bin   — concatenated float32 arrays
    web/public/lensing.json  — dims, axis ranges, byte offsets

Run from the repository root:
    python3 web/pack_lensing.py              # default: setting=std
    python3 web/pack_lensing.py --setting high
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np


def read_axis(path: Path) -> tuple[float, float, int]:
    tokens = path.read_text().split()
    if len(tokens) != 3:
        sys.exit(f"{path}: expected 3 tokens (min max N), got {len(tokens)}")
    return float(tokens[0]), float(tokens[1]), int(tokens[2])


def read_matrix(path: Path, n_rows: int, n_cols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.shape != (n_rows, n_cols):
        sys.exit(f"{path}: expected shape ({n_rows}, {n_cols}), got {arr.shape}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="std", choices=["std", "high"])
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parent.parent, type=Path)
    args = parser.parse_args()

    src = args.repo_root / "tables" / "lensing" / args.setting
    if not src.is_dir():
        sys.exit(f"missing {src} — run `python3 c_lensing.py` first")

    u_min, u_max, n_u = read_axis(src / "u.txt")
    cp_min, cp_max, n_cp = read_axis(src / "cos_psi.txt")
    ca_min, ca_max, n_ca = read_axis(src / "cos_alpha.txt")

    cos_alpha_of_u_cp = read_matrix(src / "cos_alpha_of_u_cos_psi.txt", n_u, n_cp)
    lf_of_u_cp = read_matrix(src / "lf_of_u_cos_psi.txt", n_u, n_cp)
    cdt_of_u_ca = read_matrix(src / "cdt_over_R_of_u_cos_alpha.txt", n_u, n_ca)

    out_dir = args.repo_root / "web" / "public"
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = {}
    offset = 0
    with (out_dir / "lensing.bin").open("wb") as f:
        for name, arr in [
            ("cos_alpha_of_u_cos_psi", cos_alpha_of_u_cp),
            ("lf_of_u_cos_psi", lf_of_u_cp),
            ("cdt_over_R_of_u_cos_alpha", cdt_of_u_ca),
        ]:
            data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            f.write(data)
            offsets[name] = {"byte_offset": offset, "n_bytes": len(data)}
            offset += len(data)

    header = {
        "setting": args.setting,
        "u": {"min": u_min, "max": u_max, "n": n_u},
        "cos_psi": {"min": cp_min, "max": cp_max, "n": n_cp},
        "cos_alpha": {"min": ca_min, "max": ca_max, "n": n_ca},
        "arrays": offsets,
        "total_bytes": offset,
        "dtype": "float32",
    }
    (out_dir / "lensing.json").write_text(json.dumps(header, indent=2))

    print(f"wrote {out_dir / 'lensing.bin'} ({offset / 1024:.1f} KiB)")
    print(f"wrote {out_dir / 'lensing.json'}")

    ref_src = args.repo_root / "output"
    ref_dst = out_dir / "reference"
    ref_dst.mkdir(exist_ok=True)
    copied = 0
    for letter in "abcdefghij":
        src_file = ref_src / f"os1{letter}_gpu.txt"
        if src_file.exists():
            shutil.copy(src_file, ref_dst / src_file.name)
            copied += 1
    if copied:
        print(f"copied {copied} OS1 reference outputs to {ref_dst}")
    else:
        print(f"no reference outputs found in {ref_src} (run ./build/os1_gpu to generate)")


if __name__ == "__main__":
    main()
