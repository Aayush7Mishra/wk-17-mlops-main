"""Whitelisted tools exposed to the language model."""

from __future__ import annotations

import ast
import inspect
import math
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, ValidationError

from app.schemas import Source, ToolExecution


class CalculatorArguments(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class TimeArguments(BaseModel):
    timezone: str = Field(default="UTC", min_length=1, max_length=80)


class SearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=4, ge=1, le=8)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def _evaluate_node(node: ast.AST, *, depth: int = 0) -> float:
    if depth > 20:
        raise ValueError("expression is too deeply nested")
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, depth=depth + 1)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        value = float(node.value)
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left, depth=depth + 1)
        right = _evaluate_node(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent magnitude must be at most 12")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, depth=depth + 1))
    else:
        raise ValueError("only numeric literals and arithmetic operators are allowed")
    if not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("result is outside the supported numeric range")
    return value


def calculate(expression: str) -> dict[str, int | float | str]:
    """Safely evaluate arithmetic without eval, names, calls, or attribute access."""

    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate_node(tree)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"invalid arithmetic expression: {exc}") from exc
    rendered: int | float = int(value) if value.is_integer() else value
    return {"expression": expression, "value": rendered}


def current_time(timezone: str = "UTC") -> dict[str, str]:
    """Return an ISO-8601 timestamp for an IANA time zone."""

    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown IANA timezone: {timezone}") from exc
    now = datetime.now(tz=zone)
    return {"timezone": timezone, "iso8601": now.isoformat(), "display": now.strftime("%c %Z")}


Retriever = Callable[[str, int, str], Awaitable[list[Source]]]


class ToolRegistry:
    """Execute only explicitly registered functions with validated arguments."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever

    def definitions(self, *, include_retrieval: bool = True) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a numeric arithmetic expression safely.",
                    "parameters": CalculatorArguments.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": (
                        "Get the current time in an IANA timezone such as Asia/Kathmandu."
                    ),
                    "parameters": TimeArguments.model_json_schema(),
                },
            },
        ]
        if include_retrieval and self._retriever is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "description": (
                            "Search the ingested knowledge base for supporting passages."
                        ),
                        "parameters": SearchArguments.model_json_schema(),
                    },
                }
            )
        return tools

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        namespace: str = "default",
        allow_retrieval: bool = True,
    ) -> ToolExecution:
        try:
            if name == "calculate":
                args = CalculatorArguments.model_validate(arguments)
                result = calculate(args.expression)
            elif name == "get_current_time":
                args = TimeArguments.model_validate(arguments)
                result = current_time(args.timezone)
            elif (
                name == "search_knowledge_base" and allow_retrieval and self._retriever is not None
            ):
                args = SearchArguments.model_validate(arguments)
                result = [
                    source.model_dump(mode="json")
                    for source in await self._retriever(args.query, args.limit, namespace)
                ]
            else:
                raise ValueError(f"tool is not registered: {name}")
            if inspect.isawaitable(result):
                result = await result
            return ToolExecution(name=name, arguments=arguments, ok=True, result=result)
        except (ValidationError, ValueError, ArithmeticError) as exc:
            return ToolExecution(
                name=name,
                arguments=arguments,
                ok=False,
                error=str(exc),
            )
        except Exception as exc:
            return ToolExecution(
                name=name,
                arguments=arguments,
                ok=False,
                error=f"tool execution failed ({type(exc).__name__})",
            )
