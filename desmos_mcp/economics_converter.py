from __future__ import annotations

import re


def convert_economics_variables(latex: str) -> str:
    """Convert common economics notation to Desmos-compatible form.

    - q -> x (quantity)
    - p -> y (price)
    - C(q) -> y (cost function)
    - U(...), f(...) -> y (utility/production), keep arguments as-is
    - sqrt() -> \sqrt{}
    """
    expr = latex.strip()

    # Normal function to LaTeX sqrt
    expr = re.sub(r"\bsqrt\((.*?)\)", r"\\sqrt{\1}", expr)

    # Cost function -> y
    expr = re.sub(r"\bC\s*\(\s*q\s*\)", "y", expr)

    # Utility/production -> y
    expr = re.sub(r"\bU\s*\([^)]*\)", "y", expr)
    expr = re.sub(r"\bf\s*\([^)]*\)", "y", expr)

    # Variables: price/quantity mapping
    expr = re.sub(r"\bq\b", "x", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bp\b", "y", expr, flags=re.IGNORECASE)

    return expr
