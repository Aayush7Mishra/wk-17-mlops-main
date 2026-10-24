"""Upload the bundled sample document to a running backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--session-id", default="sample")
    parser.add_argument("--api-key", default=os.getenv("APP_API_KEY", ""))
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).parents[1] / "sample_data" / "employee_handbook.md",
    )
    args = parser.parse_args()
    with args.file.open("rb") as stream:
        response = httpx.post(
            f"{args.backend.rstrip('/')}/api/v1/documents",
            files={"file": (args.file.name, stream, "text/markdown")},
            data={"session_id": args.session_id},
            headers={"X-API-Key": args.api_key} if args.api_key else {},
            timeout=60.0,
        )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
