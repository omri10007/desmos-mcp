from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SliderBounds(BaseModel):
    min: str
    max: str
    step: Optional[str] = None


class DomainBounds(BaseModel):
    min: str
    max: str


class ExpressionState(BaseModel):
    """Represents a Desmos expression supported by calculator.setExpression."""

    type: str = Field(
        default="expression",
        description="Only 'expression' supported by this server",
    )
    latex: str = Field(description="LaTeX expression, e.g. y=x^2")
    id: Optional[str] = Field(
        default=None,
        description="Stable identifier for the expression",
    )
    color: Optional[str] = Field(
        default=None,
        description="Hex color like #c74440",
    )
    style: Optional[str] = Field(
        default=None,
        description="Drawing style; see Desmos styles",
    )
    hidden: Optional[bool] = Field(
        default=None,
        description="Whether to hide the graph",
    )
    secret: Optional[bool] = Field(
        default=None,
        description=(
            "Hide from expressions list but still graph"
        ),
    )
    sliderBounds: Optional[SliderBounds] = None
    domain: Optional[DomainBounds] = None
    dragMode: Optional[str] = None
    label: Optional[str] = None
    showLabel: Optional[bool] = None

    @field_validator("color")
    @classmethod
    def _validate_color(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        v = value.strip()
        if (
            len(v) == 7
            and v.startswith("#")
            and all(c in "0123456789abcdefABCDEF" for c in v[1:])
        ):
            return v
        raise ValueError("color must be a hex like #rrggbb")


class MathBounds(BaseModel):
    left: Optional[float] = None
    right: Optional[float] = None
    top: Optional[float] = None
    bottom: Optional[float] = None


class DesmosState(BaseModel):
    """High-level state our HTML loader will consume.

    Note: We intentionally do not rely on Desmos.getState() schema to
    avoid compatibility risk. The embed HTML will apply these values via
    setExpression and setMathBounds.
    """

    expressions: List[ExpressionState] = Field(default_factory=list)
    mathBounds: Optional[MathBounds] = None
    degreeMode: bool = False


class EmbedOptions(BaseModel):
    width: int = Field(default=800, gt=0)
    height: int = Field(default=600, gt=0)
    # UI/behavior flags
    keypad: bool = True
    graphpaper: bool = True
    expressions: bool = True
    settingsMenu: bool = True
    zoomButtons: bool = True
    expressionsTopbar: bool = True
    border: bool = True
    lockViewport: bool = False
    projectorMode: bool = False
    language: str = "en"
