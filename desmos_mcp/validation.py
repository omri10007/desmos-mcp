from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass
from typing import List


class ValidationSeverity(Enum):
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass
class ValidationIssue:
    message: str
    severity: ValidationSeverity


_INCOMPLETE_TRAILING_OP = re.compile(r"[+\-*/^]\s*$")
_INVALID_CHARS = re.compile(
    r"[^a-zA-Z0-9+\-*/^(){}=\\,._\s\[\]<>]"
)
_SINGLE_LETTER_VARS = re.compile(r"\b([a-zA-Z])\b")


def validate_latex_syntax(latex: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Empty expression
    if not latex or not latex.strip():
        issues.append(
            ValidationIssue(
                "Empty expression",
                ValidationSeverity.ERROR,
            )
        )
        return issues

    expr = latex.strip()

    # Incomplete expression (trailing operator)
    if _INCOMPLETE_TRAILING_OP.search(expr):
        issues.append(
            ValidationIssue(
                "Incomplete expression (trailing operator)",
                ValidationSeverity.ERROR,
            )
        )

    # Obviously invalid characters (allow common LaTeX characters)
    if _INVALID_CHARS.search(expr):
        issues.append(
            ValidationIssue(
                "Invalid characters detected",
                ValidationSeverity.ERROR,
            )
        )

    # Basic contextual variable checks
    # If y=... expression, only x/e/t should appear as single-letter variables
    if expr.startswith("y="):
        rhs = expr[2:].split("\\{")[0]
        vars_found = set(
            m.group(1) for m in _SINGLE_LETTER_VARS.finditer(rhs)
        )
        allowed = {"x", "e", "t"}
        undefined = vars_found - allowed
        if undefined:
            issues.append(
                ValidationIssue(
                    "Undefined variables in y= expression: "
                    + ", ".join(sorted(undefined)),
                    ValidationSeverity.ERROR,
                )
            )

    # Domain restriction variable consistency
    if "\\{" in expr:
        try:
            domain_part = expr.split("\\{")[1].split("\\}")[0]
        except Exception:
            issues.append(
                ValidationIssue(
                    "Malformed domain restriction block",
                    ValidationSeverity.ERROR,
                )
            )
        else:
            if expr.startswith("y=") and "x" not in domain_part:
                issues.append(
                    ValidationIssue(
                        "Domain for y= expressions must use 'x'",
                        ValidationSeverity.ERROR,
                    )
                )
            if expr.startswith("x=") and "y" not in domain_part:
                issues.append(
                    ValidationIssue(
                        "Domain for x= expressions must use 'y'",
                        ValidationSeverity.ERROR,
                    )
                )

    return issues


def has_blocking_issues(issues: List[ValidationIssue]) -> bool:
    return any(
        i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        for i in issues
    )


def summarize_issues(issues: List[ValidationIssue]) -> str:
    return "; ".join(f"{i.severity.name}: {i.message}" for i in issues)
