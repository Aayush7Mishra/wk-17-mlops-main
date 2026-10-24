"""Run: python -m agentic verify 'query' or python -m agentic evaluate."""

from __future__ import annotations

import argparse
import asyncio
import json

from agentic import AgentConfig, OfflinePolicy, OpenAIPlanner, SourceStore, VerificationAgent
from agentic.evaluation import run_suite


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "evaluate"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", default="evaluation/results")
    parser.add_argument(
        "--input-price", type=float, default=None, help="USD per million input tokens"
    )
    parser.add_argument(
        "--output-price", type=float, default=None, help="USD per million output tokens"
    )
    args = parser.parse_args()
    config = AgentConfig(
        max_steps=args.max_steps,
        input_price_per_million=args.input_price,
        output_price_per_million=args.output_price,
    )
    if args.command == "evaluate":
        report = await run_suite(config=config, mode=args.mode, output_dir=args.output)
        print(json.dumps(report["metrics"], indent=2))
        return
    if not args.query:
        parser.error("verify requires a query")
    planner = OfflinePolicy() if args.mode == "offline" else OpenAIPlanner()
    try:
        result = await VerificationAgent(
            planner=planner, tools=SourceStore.default(), config=config
        ).run(args.query)
        print(json.dumps(result.to_dict(), indent=2))
    finally:
        if hasattr(planner, "close"):
            await planner.close()


if __name__ == "__main__":
    asyncio.run(main())
