from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face Whisper checkpoint to CTranslate2 format."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--output-dir", required=True, help="Directory for the converted model")
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=["float32", "float16", "int16", "int8", "int8_float16", "int8_float32"],
        help="Quantization to apply to the converted model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    parser.add_argument(
        "--copy-file",
        action="append",
        dest="copy_files",
        default=None,
        help=(
            "Additional file to copy into the converted model directory. "
            "By default this script copies tokenizer.json and preprocessor_config.json."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    converter_binary = shutil.which("ct2-transformers-converter")
    if converter_binary is None:
        print(
            "ct2-transformers-converter is not available in the active environment. "
            "Install the project dependencies first.",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    copy_files = args.copy_files or ["tokenizer.json", "preprocessor_config.json"]

    command = [
        converter_binary,
        "--model",
        args.model,
        "--output_dir",
        str(output_dir),
        "--copy_files",
        *copy_files,
        "--quantization",
        args.quantization,
    ]
    if args.force:
        command.append("--force")

    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"Conversion failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    expected_files = [output_dir / "config.json", output_dir / "model.bin"]
    missing_files = [str(path) for path in expected_files if not path.exists()]
    if missing_files:
        print(
            "Conversion command exited successfully, but the converted model is incomplete. "
            f"Missing files: {', '.join(missing_files)}",
            file=sys.stderr,
        )
        return 1

    print(f"Converted model written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
