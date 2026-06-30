"""Generate the PLNS protobuf Python module.

This example converts the PLNS protocol definition file, usually
``chassis_sdk/proto/main.proto``, into the generated Python module used by the
SDK, usually ``chassis_sdk/proto/PlnsProto_pb2.py``.

The SDK uses this generated module to build and parse PLNS request, response,
command, and notification messages. Regenerate it when the PLNS ``main.proto``
file changes, or when you need a generated file compatible with the protobuf
compiler/runtime installed in your Python environment.

Install compiler dependencies before running:
    pip install ".[grpcio]"

Usage:
    python examples/generate_plns_proto.py <input_proto> <output_path>

Examples:
    python examples/generate_plns_proto.py chassis_sdk/proto/main.proto chassis_sdk/proto/PlnsProto_pb2.py
    python examples/generate_plns_proto.py chassis_sdk/proto/main.proto runs/generated_proto/

Parameter behavior:
- ``input_proto`` is the source ``.proto`` file.
- ``output_path`` can be a Python file path or a directory.
- If ``output_path`` is a directory, the script writes ``PlnsProto_pb2.py``
  inside that directory.

Implementation note:
The protobuf compiler normally names generated files after the input proto
file, for example ``main.proto`` becomes ``main_pb2.py``. This script compiles
to a temporary directory first, then copies the generated file to the requested
stable SDK module name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DEFAULT_OUTPUT_NAME = "PlnsProto_pb2.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PlnsProto_pb2.py from the PLNS main.proto file."
    )
    parser.add_argument(
        "input_proto",
        type=Path,
        help="Input PLNS .proto file path, for example chassis_sdk/proto/main.proto.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help=(
            "Output Python file path, for example chassis_sdk/proto/PlnsProto_pb2.py. "
            "If a directory is passed, PlnsProto_pb2.py is written inside it."
        ),
    )
    return parser.parse_args()


def normalize_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path
    return output_path / DEFAULT_OUTPUT_NAME


def validate_paths(input_proto: Path, output_file: Path) -> None:
    if not input_proto.exists():
        raise FileNotFoundError(f"Input proto file does not exist: {input_proto}")
    if not input_proto.is_file():
        raise ValueError(f"Input proto path is not a file: {input_proto}")
    if input_proto.suffix != ".proto":
        raise ValueError(f"Input file must be a .proto file: {input_proto}")
    if output_file.suffix != ".py":
        raise ValueError(f"Output path must be a Python .py file or a directory: {output_file}")


def run_proto_compiler(input_proto: Path, temp_dir: Path) -> None:
    proto_dir = input_proto.parent

    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "--proto_path",
        str(proto_dir),
        "--python_out",
        str(temp_dir),
        input_proto.name,
    ]

    subprocess.run(command, check=True)


def generate_proto(input_proto: Path, output_path: Path) -> Path:
    input_proto = input_proto.resolve()
    output_file = normalize_output_path(output_path).resolve()
    validate_paths(input_proto, output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="plns_proto_") as temp_name:
        temp_dir = Path(temp_name)
        run_proto_compiler(input_proto, temp_dir)

        generated_file = temp_dir / f"{input_proto.stem}_pb2.py"
        if not generated_file.exists():
            raise FileNotFoundError(
                f"Compiler did not create the expected file: {generated_file}"
            )

        shutil.copy2(generated_file, output_file)

    return output_file


def main() -> None:
    args = parse_args()
    output_file = generate_proto(args.input_proto, args.output_path)
    print(f"Generated: {output_file}")


if __name__ == "__main__":
    main()
