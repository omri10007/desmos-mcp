from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel

from .models import ExpressionState, MathBounds


class TemplateParam(BaseModel):
    name: str
    description: str


def _as_label(value: object, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _to_float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _with_domain(latex_y_equals: str,
                 x_min: float | None,
                 x_max: float | None) -> str:
    if x_min is not None and x_max is not None:
        return f"{latex_y_equals} \\{{ {x_min} < x < {x_max} \\}}"
    if x_min is not None:
        return f"{latex_y_equals} \\{{ x \\ge {x_min} \\}}"
    if x_max is not None:
        return f"{latex_y_equals} \\{{ x \\le {x_max} \\}}"
    return latex_y_equals


# Registry for templates and preset bounds
_AVAILABLE_TEMPLATES = {"parabola", "y=x^2", "sine", "y=sin(x)", "sin", "supply_demand", "econ_sd", "econ-sd"}


def render_template(
    name: str,
    params: Dict[str, object] | None = None,
) -> List[ExpressionState]:
    params = params or {}
    n = name.lower().strip()

    if n not in _AVAILABLE_TEMPLATES:
        raise ValueError(
            f"Template '{name}' not found. Available: {sorted(_AVAILABLE_TEMPLATES)}"
        )

    if n in {"parabola", "y=x^2"}:
        color = "#2d70b3"
        label = _as_label(params.get("label"), "Parabola")
        return [
            ExpressionState(
                latex="y=x^2",
                color=color,
                label=label,
                showLabel=False,
            )
        ]

    if n in {"sine", "y=sin(x)", "sin"}:
        color = "#e15759"
        label = _as_label(params.get("label"), "Sine")
        return [
            ExpressionState(
                latex="y=\\sin(x)",
                color=color,
                label=label,
                showLabel=False,
            )
        ]

    if n in {"supply_demand", "econ_sd", "econ-sd"}:
        # Accept rhs strings; default to simple linear curves
        d_rhs = str(params.get("demand_rhs", "20 - 0.5x"))
        s_rhs = str(params.get("supply_rhs", "2 + 0.2x"))
        raw_x_min = params.get("x_min", 0.0)
        raw_x_max = params.get("x_max", 50.0)
        x_min = _to_float_or_none(raw_x_min)
        x_max = _to_float_or_none(raw_x_max)
        demand_label = _as_label(params.get("demand_label"), "Demand")
        supply_label = _as_label(params.get("supply_label"), "Supply")

        d_expr = _with_domain(f"y={d_rhs}", x_min, x_max)
        s_expr = _with_domain(f"y={s_rhs}", x_min, x_max)
        return [
            ExpressionState(
                latex=d_expr,
                color="#c74440",
                label=demand_label,
                showLabel=True,
            ),
            ExpressionState(
                latex=s_expr,
                color="#2d70b3",
                label=supply_label,
                showLabel=True,
            ),
        ]

    # Unreachable because we raised on unknown above
    return [
        ExpressionState(
            latex="y=x",
            color="#2d70b3",
            label=name,
            showLabel=False,
        )
    ]


_PRESET_BOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "square10": (-10.0, 10.0, 10.0, -10.0),
    "square5": (-5.0, 5.0, 5.0, -5.0),
    "wide": (-20.0, 20.0, 10.0, -10.0),
    "econ": (0.0, 50.0, 50.0, 0.0),
}


def preset_bounds(name: str) -> MathBounds:
    key = name.lower().strip()
    if key not in _PRESET_BOUNDS:
        raise ValueError(
            f"Preset bounds '{name}' not found. Available: {sorted(_PRESET_BOUNDS.keys())}"
        )
    left, right, top, bottom = _PRESET_BOUNDS[key]
    return MathBounds(left=left, right=right, top=top, bottom=bottom)


def list_available_templates() -> List[str]:
    return sorted(_AVAILABLE_TEMPLATES)


def list_available_bounds() -> List[str]:
    return sorted(_PRESET_BOUNDS.keys())
