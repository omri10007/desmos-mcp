from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import webbrowser
import logging
import os
import hashlib

from mcp.server.fastmcp import FastMCP

from .models import (
    ExpressionState,
    MathBounds,
    DesmosState,
    EmbedOptions,
)
from .templates import (
    render_template,
    preset_bounds,
    list_available_templates,
    list_available_bounds,
)
from .validation import (
    validate_latex_syntax,
    has_blocking_issues,
    summarize_issues,
)
from .economics_converter import convert_economics_variables


# ---- Temporary test helpers (kept as thin wrappers) ----
# These names are expected by some local tests. They delegate to the
# canonical implementations to avoid duplication.

def _normalize_economic_variables(expression: str) -> str:
    """Normalize economics notation (p/q/C()/U()/f()) to Desmos x/y.

    This is a thin wrapper around `convert_economics_variables`.
    """
    return convert_economics_variables(expression)


def _validate_desmos_expression(expression: str) -> List[str]:
    """Validate a Desmos expression and return a list of error messages.

    Uses the strict validator and returns human-readable messages for
    ERROR/CRITICAL issues only. Empty list means valid.
    """
    issues = validate_latex_syntax(expression)
    return [
        i.message
        for i in issues
        if i.severity.name in {"ERROR", "CRITICAL"}
    ]


# ---- Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    log_level_name = os.getenv("DESMOS_MCP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(level=log_level)


# ---- Server

mcp = FastMCP("Desmos MCP")

# In-memory cache and metrics
_cache: dict[str, str] = {}
_metrics: dict[str, int] = {
    "generate_calls": 0,
    "cache_hits": 0,
}


# ---- Conditional tool registration (Minimal vs Full toolset)
_MINIMAL_TOOL_NAMES = {
    "validate_expressions",
    "generate_state",
    "generate_embed_html",
    "quick_plot",
}


def _tool_enabled(tool_name: str) -> bool:
    toolset = os.getenv("DESMOS_MCP_TOOLSET", "minimal").lower()
    if toolset == "minimal":
        return tool_name in _MINIMAL_TOOL_NAMES
    return True


def maybe_tool(tool_name: str):
    """Decorator to conditionally register an MCP tool based on env.

    If DESMOS_MCP_TOOLSET=minimal, only a core subset is registered.
    In full mode (default), all tools are registered.
    """
    def _decorator(fn):
        if _tool_enabled(tool_name):
            return mcp.tool()(fn)
        # Leave function unregistered but callable locally (for tests)
        return fn

    return _decorator


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_script_close(js_text: str) -> str:
    r"""Prevent '</script>' from prematurely terminating the script tag.

    This replaces '</' with '<\/' which is safe in JS strings/JSON and
    avoids closing the surrounding <script> block when the HTML is parsed.
    """
    return js_text.replace("</", "<\\/")


# ---- Validation helpers

def _looks_plottable_in_desmos(latex: str) -> bool:
    """Heuristic: Desmos plots relations/functions of x/y when written
    with an equals sign and using x or y. E.g. 'y=...' or 'x=...'.
    """
    s = latex.replace(" ", "").lower()
    if "=" not in s:
        return False
    return ("y=" in s) or ("x=" in s)


def _lint_latex_for_econ(latex: str) -> List[str]:
    """Return human-readable warnings for common econ mistakes.

    - Using variables p/q without mapping to x/y
    - Missing equals sign
    - Not expressed as y=... or x=...
    """
    warnings: List[str] = []
    s = latex.replace(" ", "")
    lower = s.lower()
    if "=" not in s:
        warnings.append("Expression has no '='; Desmos may not plot it.")
    if not ("y=" in lower or "x=" in lower):
        warnings.append(
            "Prefer 'y=...' (price) or 'x=...' forms for plotting."
        )
    if (
        any(v in lower for v in ("d(", "s(", "p", "q"))
        and not ("x" in lower or "y" in lower)
    ):
        warnings.append(
            "Detected p/q or D()/S() variables without x/y. "
            "Map quantity→x and price→y."
        )
    return warnings


@maybe_tool("validate_expressions")
def validate_expressions(expressions: List[ExpressionState]) -> List[dict]:
    """Validate LaTeX expressions and summarize potential issues.

    Parameters
    ----------
    expressions : List[ExpressionState]
        List of Desmos expressions to check. Only the `latex` field is
        considered by the validator; other fields are ignored.

    Returns
    -------
    List[dict]
        For each input expression, returns a dict with:
        - 'latex': the original string
        - 'warnings': human-readable messages (non-blocking + hints)
        - 'blocked': True if strict validation found an ERROR/CRITICAL
    """
    out: List[dict] = []
    for expr in expressions:
        # Strict LaTeX validation
        issues = validate_latex_syntax(expr.latex)
        if has_blocking_issues(issues):
            out.append({
                "latex": expr.latex,
                "warnings": [summarize_issues(issues)],
                "blocked": True,
            })
            continue

        ws = _lint_latex_for_econ(expr.latex)
        if not _looks_plottable_in_desmos(expr.latex):
            if "Prefer 'y=...'" not in ws:
                ws.append("Expression may not be plottable as-is in Desmos.")
        # Include strict validator messages as warnings (non-blocking group)
        if issues:
            ws.extend(f"{i.severity.name}: {i.message}" for i in issues)
        out.append({"latex": expr.latex, "warnings": ws, "blocked": False})
    return out


@maybe_tool("generate_state")
def generate_state(
    expressions: List[ExpressionState],
    math_bounds: Optional[MathBounds] = None,
    degree_mode: bool = False,
) -> DesmosState:
    """Create a Desmos state from expressions and optional bounds.

    Parameters
    ----------
    expressions : List[ExpressionState]
        The expressions to render. See `ExpressionState` fields for
        optional color/label/visibility settings.
    math_bounds : Optional[MathBounds]
        Optional viewport bounds. If omitted, Desmos decides defaults.
    degree_mode : bool
        Whether to render in degree mode (default False = radians).

    Returns
    -------
    DesmosState
        A structured state consumed by the embed HTML.
    """
    logger.info(
        "generate_state: %d expression(s), degree_mode=%s",
        len(expressions),
        degree_mode,
    )
    return DesmosState(
        expressions=expressions,
        mathBounds=math_bounds,
        degreeMode=degree_mode,
    )


DEMO_API_KEY = "dcb31709b452b1cf9dc26972add0fda6"
DEFAULT_SCRIPT_URL = os.getenv(
    "DESMOS_SCRIPT_URL", "https://www.desmos.com/api/v1.11/calculator.js"
)


@maybe_tool("generate_embed_html")
async def generate_embed_html(
    state: DesmosState,
    api_key: Optional[str] = None,
    options: Optional[EmbedOptions] = None,
    title: str = "Desmos Graph",
    ctx: Optional[object] = None,
    minify: bool = False,
    script_url: Optional[str] = None,
    # New optional behavior for streamlined workflow
    save_to: Optional[str] = None,
    open_browser: bool = False,
) -> str:
    """Generate self-contained HTML that embeds a Desmos graph for `state`.

    Parameters
    ----------
    state : DesmosState
        Expressions, optional bounds, and degree mode.
    api_key : Optional[str]
        Desmos API key. Defaults to env `DESMOS_API_KEY` or demo key.
    options : Optional[EmbedOptions]
        UI/behavior options (size, keypad, language, etc.).
    title : str
        Document title for the HTML. Default "Desmos Graph".
    ctx : Optional[object]
        Unused. Reserved for future context injection.
    minify : bool
        If True, strips whitespace between tags in the output HTML.
    script_url : Optional[str]
        Override the Desmos script URL. Defaults to `DESMOS_SCRIPT_URL`
        or the built-in stable URL.
    save_to : Optional[str]
        If provided, also save the HTML to `preview/` with this name
        (auto-adding .html and avoiding collisions).
    open_browser : bool
        If True, opens the saved HTML in the default browser.

    Returns
    -------
    str
        The generated HTML string.
    """
    # Reject if any expression fails strict validation
    for e in state.expressions:
        issues = validate_latex_syntax(e.latex)
        if has_blocking_issues(issues):
            raise ValueError(
                (
                    "Blocking validation error for "
                    f"'{e.latex}': {summarize_issues(issues)}"
                )
            )
    # No ctx required; pure function
    opts = options or EmbedOptions()
    env_key = os.getenv("DESMOS_API_KEY")
    effective_key = api_key or env_key or DEMO_API_KEY
    effective_script = script_url or DEFAULT_SCRIPT_URL

    # Inline JSON for expressions and optional bounds
    import json

    expressions_json = json.dumps(
        [
            expr.model_dump(exclude_none=True)
            for expr in state.expressions
        ]
    )
    math_bounds_json = json.dumps(
        state.mathBounds.model_dump(exclude_none=True)
        if state.mathBounds
        else None
    )
    # Prevent '</script>' sequences from breaking out of the script tag
    expressions_json = _escape_script_close(expressions_json)
    math_bounds_json = _escape_script_close(math_bounds_json)

    degree_mode_js = "true" if state.degreeMode else "false"

    _metrics["generate_calls"] += 1

    logger.info(
        "generate_embed_html: %d expression(s), bounds=%s, "
        "degree_mode=%s, size=%dx%d",
        len(state.expressions),
        state.mathBounds is not None,
        state.degreeMode,
        opts.width,
        opts.height,
    )

    # Basic HTML template
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\"
        content=\"width=device-width, initial-scale=1\" />
  <title>{_html_escape(title)}</title>
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; }}
    #calculator {{
      width: {opts.width}px;
      height: {opts.height}px;
      margin: 0 auto;
    }}
    body {{ background: #fff; }}
  </style>
  <script src=\"{effective_script}?apiKey={effective_key}\"></script>
</head>
<body>
  <div id=\"calculator\"></div>
  <script>
    const INITIAL_EXPRESSIONS = {expressions_json};
    const INITIAL_BOUNDS = {math_bounds_json};
    const DEGREE_MODE = {degree_mode_js};

    const elt = document.getElementById('calculator');
    const calculator = Desmos.GraphingCalculator(elt, {{
      keypad: {str(opts.keypad).lower()},
      graphpaper: {str(opts.graphpaper).lower()},
      expressions: {str(opts.expressions).lower()},
      settingsMenu: {str(opts.settingsMenu).lower()},
      zoomButtons: {str(opts.zoomButtons).lower()},
      expressionsTopbar: {str(opts.expressionsTopbar).lower()},
      border: {str(opts.border).lower()},
      lockViewport: {str(opts.lockViewport).lower()},
      projectorMode: {str(opts.projectorMode).lower()},
      language: {_html_escape(
        json.dumps(opts.language)
      )}
    }});

    // Apply degree mode if needed
    if (DEGREE_MODE) {{
      calculator.updateSettings({{ degreeMode: true }});
    }}

    // Apply expressions
    for (const expr of INITIAL_EXPRESSIONS) {{
      calculator.setExpression(expr);
    }}

    if (INITIAL_BOUNDS) {{
      const b = INITIAL_BOUNDS;
      if ((b.left !== undefined && b.right !== undefined) ||
          (b.top !== undefined && b.bottom !== undefined)) {{
        const current = calculator.graphpaperBounds.mathCoordinates;
        const left = (b.left ?? current.left);
        const right = (b.right ?? current.right);
        const top = (b.top ?? current.top);
        const bottom = (b.bottom ?? current.bottom);
        calculator.setMathBounds({{ left, right, top, bottom }});
      }}
    }}
  </script>
</body>
</html>
"""

    if minify:
        # Simple minifier: strip redundant whitespace between tags
        html = "\n".join(line.strip() for line in html.splitlines())

    # Optional side-effects for streamlined workflow
    if save_to is not None or open_browser:
        path = await save_html(html=html, filename=save_to)
        if open_browser:
            uri = Path(path).resolve().as_uri()
            webbrowser.open_new_tab(uri)
            logger.info("generate_embed_html: opened browser for %s", path)
        else:
            logger.info("generate_embed_html: saved to %s", path)

    return html


def _cache_key(*parts: str) -> str:
    hasher = hashlib.sha256()
    for p in parts:
        hasher.update(p.encode("utf-8"))
    return hasher.hexdigest()


@maybe_tool("save_html")
async def save_html(html: str, filename: Optional[str] = None) -> str:
    """Save HTML content under preview/ and return the relative path."""
    preview_dir = Path("preview").resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    requested_name = filename or f"desmos-{ts}.html"

    # Sanitize the requested name to avoid path traversal and enforce
    # .html extension
    safe_name = Path(requested_name).name
    if not safe_name.lower().endswith(".html"):
        safe_name = f"{safe_name}.html"

    out_path = (preview_dir / safe_name).resolve()

    # Ensure target remains within preview_dir
    if preview_dir not in out_path.parents and out_path != preview_dir:
        logger.warning(
            "save_html: path traversal attempt, using default filename"
        )
        out_path = preview_dir / f"desmos-{ts}.html"

    # Ensure unique filename if collision
    if out_path.exists():
        base = out_path.stem
        suffix = out_path.suffix
        counter = 1
        while out_path.exists():
            out_path = preview_dir / f"{base}-{counter}{suffix}"
            counter += 1

    out_path.write_text(html, encoding="utf-8")
    logger.info("save_html: wrote %s", out_path)

    # Return posix-style relative path for cross-platform friendliness
    return out_path.relative_to(preview_dir.parent).as_posix()


@maybe_tool("open_in_browser")
async def open_in_browser(
    state: DesmosState,
    api_key: Optional[str] = None,
    options: Optional[EmbedOptions] = None,
    title: str = "Desmos Graph",
    filename: Optional[str] = None,
) -> dict:
    """Generate HTML, save under preview/, and open it in the browser.

    Returns { path, opened }.
    """
    html = await generate_embed_html(
        state=state,
        api_key=api_key,
        options=options,
        title=title,
    )
    path = await save_html(html=html, filename=filename)
    uri = Path(path).resolve().as_uri()
    opened = webbrowser.open_new_tab(uri)
    logger.info(
        "open_in_browser: opened=%s path=%s",
        bool(opened),
        path,
    )
    return {"path": path, "opened": bool(opened)}


@maybe_tool("open_saved_html")
def open_saved_html(path: str) -> bool:
    """Open an existing HTML file in the default browser.

    Returns True if a browser open was attempted.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("open_saved_html: not found %s", path)
        return False
    opened = webbrowser.open_new_tab(p.resolve().as_uri())
    logger.info(
        "open_saved_html: opened=%s path=%s",
        bool(opened),
        path,
    )
    return bool(opened)


@maybe_tool("quick_plot")
async def quick_plot(
    latex: str,
    api_key: Optional[str] = None,
    options: Optional[EmbedOptions] = None,
    filename: Optional[str] = None,
    color: Optional[str] = None,
    hidden: Optional[bool] = None,
    label: Optional[str] = None,
    show_label: Optional[bool] = None,
) -> dict:
    """Create a simple graph from a single LaTeX expression.

    Parameters
    ----------
    latex : str
        A Desmos-compatible LaTeX expression (e.g., "y=x^2").
    api_key : Optional[str]
        Desmos API key override. See `generate_embed_html`.
    options : Optional[EmbedOptions]
        UI/behavior options. See `EmbedOptions` fields.
    filename : Optional[str]
        Output filename under `preview/` (extension auto-added).
    color : Optional[str]
        Hex color like #c74440.
    hidden : Optional[bool]
        Whether to hide the expression in the graph.
    label : Optional[str]
        Label to display for the expression.
    show_label : Optional[bool]
        Whether to show the label.

    Returns
    -------
    dict
        A dict with:
        - 'path': saved HTML file relative path
        - 'html': generated HTML string
    """
    expr = ExpressionState(
        latex=latex,
        color=color,
        hidden=hidden,
        label=label,
        showLabel=show_label,
    )
    state = generate_state([expr])
    html = await generate_embed_html(
        state=state,
        api_key=api_key,
        options=options,
    )
    path = await save_html(html=html, filename=filename)
    return {"path": path, "html": html}


# ---- Economics helpers/tools

def _with_domain(
    latex_y_equals: str,
    x_min: Optional[float],
    x_max: Optional[float],
) -> str:
    """Append domain restriction to a y=... expr via Desmos `{...}`.

    Uses chained inequality when both bounds provided, or single-sided
    otherwise.
    """
    if x_min is not None and x_max is not None:
        return f"{latex_y_equals} \\{{ {x_min} < x < {x_max} \\}}"
    if x_min is not None:
        return f"{latex_y_equals} \\{{ x \\ge {x_min} \\}}"
    if x_max is not None:
        return f"{latex_y_equals} \\{{ x \\le {x_max} \\}}"
    return latex_y_equals


@maybe_tool("create_supply_demand_state")
def create_supply_demand_state(
    demand_px: str,
    supply_px: str,
    # Domain for quantity (x-axis)
    x_min: Optional[float] = 0.0,
    x_max: Optional[float] = None,
    # Optional bounds override for viewport
    left: Optional[float] = 0.0,
    right: Optional[float] = None,
    bottom: Optional[float] = 0.0,
    top: Optional[float] = None,
    # Optional equilibrium point and guide lines
    equilibrium_x: Optional[float] = None,
    equilibrium_y: Optional[float] = None,
    demand_label: Optional[str] = "Demand",
    supply_label: Optional[str] = "Supply",
) -> DesmosState:
    """Build a standard supply/demand graph state.

    - Interprets price as y and quantity as x.
    - Expects demand_px and supply_px as price-as-a-function-of-
      quantity. Can use economic variables (p,q) which will be
      converted to (x,y).
    - Examples: '100-2*q', '10+0.5*q', '100-2*x', 'sqrt(q+4)'
    - Adds optional domain restriction on x via `{ ... }`.
    - Adds optional equilibrium point and reference lines if
      equilibrium_x/y provided.
    """
    # Normalize economic variables to Desmos variables
    d_rhs = convert_economics_variables(demand_px.strip())
    s_rhs = convert_economics_variables(supply_px.strip())

    # Handle y= prefix if present
    if d_rhs.lower().startswith("y="):
        d_expr = d_rhs
    else:
        d_expr = f"y={d_rhs}"
    if s_rhs.lower().startswith("y="):
        s_expr = s_rhs
    else:
        s_expr = f"y={s_rhs}"

    # Validate pre-domain
    d_issues = validate_latex_syntax(d_expr)
    s_issues = validate_latex_syntax(s_expr)
    if has_blocking_issues(d_issues):
        raise ValueError(
            f"Invalid demand expression: {summarize_issues(d_issues)}"
        )
    if has_blocking_issues(s_issues):
        raise ValueError(
            f"Invalid supply expression: {summarize_issues(s_issues)}"
        )

    # Add domain restrictions
    d_expr = _with_domain(d_expr, x_min=x_min, x_max=x_max)
    s_expr = _with_domain(s_expr, x_min=x_min, x_max=x_max)

    # Validate after domain
    d_issues2 = validate_latex_syntax(d_expr)
    s_issues2 = validate_latex_syntax(s_expr)
    if has_blocking_issues(d_issues2):
        raise ValueError(
            f"Invalid demand domain: {summarize_issues(d_issues2)}"
        )
    if has_blocking_issues(s_issues2):
        raise ValueError(
            f"Invalid supply domain: {summarize_issues(s_issues2)}"
        )

    warnings: List[str] = []
    warnings += _lint_latex_for_econ(d_expr)
    warnings += _lint_latex_for_econ(s_expr)
    if warnings:
        logger.warning("create_supply_demand_state: %s", "; ".join(warnings))

    expressions: List[ExpressionState] = [
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

    if equilibrium_x is not None and equilibrium_y is not None:
        # Equilibrium point
        eq_point = ExpressionState(
            latex=f"({equilibrium_x},{equilibrium_y})",
            color="#000000",
            label="Equilibrium",
            showLabel=True,
        )
        # Guide lines (vertical and horizontal)
        vline = ExpressionState(latex=f"x={equilibrium_x}", color="#555555")
        hline = ExpressionState(latex=f"y={equilibrium_y}", color="#555555")
        expressions.extend([eq_point, vline, hline])

    bounds = MathBounds(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    ) if any(val is not None for val in (left, right, top, bottom)) else None

    return DesmosState(
        expressions=expressions,
        mathBounds=bounds,
        degreeMode=False,
    )


@maybe_tool("generate_and_save_embed_html")
async def generate_and_save_embed_html(
    state: DesmosState,
    api_key: Optional[str] = None,
    options: Optional[EmbedOptions] = None,
    title: str = "Desmos Graph",
    filename: Optional[str] = None,
    minify: bool = False,
) -> dict:
    """Generate HTML for a state and save it under preview/.

    Returns { path, html }.
    """
    # Cache by content
    import json

    key = _cache_key(
        json.dumps(
            [e.model_dump(exclude_none=True) for e in state.expressions],
            sort_keys=True,
        ),
        json.dumps(
            state.mathBounds model_dump(exclude_none=True)
            if state.mathBounds
            else None,
            sort_keys=True,
        ),
        json.dumps(
            (options.model_dump(exclude_none=True) if options else {}),
            sort_keys=True,
        ),
        title,
        str(minify),
    )
    if key in _cache:
        _metrics["cache_hits"] += 1
    html = await generate_embed_html(
        state=state,
        api_key=api_key,
        options=options,
        title=title,
        minify=minify,
    )
    path = await save_html(html=html, filename=filename)
    _cache[key] = path
    return {"path": path, "html": html}


@maybe_tool("list_previews")
async def list_previews(
    limit: Optional[int] = 50,
    newest_first: bool = True,
) -> List[str]:
    """List up to N preview HTML files (relative paths)."""
    preview_dir = Path("preview").resolve()
    if not preview_dir.exists():
        return []
    files = sorted(
        [p for p in preview_dir.glob("*.html") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=newest_first,
    )
    if isinstance(limit, int) and limit > 0:
        files = files[:limit]
    rel = [p.relative_to(preview_dir.parent).as_posix() for p in files]
    logger.info("list_previews: returning %d file(s)", len(rel))
    return rel


@maybe_tool("delete_preview")
async def delete_preview(filename: str) -> bool:
    """Delete a preview HTML by name (must be within preview/).

    Returns True if deleted.
    """
    preview_dir = Path("preview").resolve()
    target = (preview_dir / Path(filename).name).resolve()
    if not target exists() or not target.is_file():
        return False
    try:
        # Ensure safety: file must be within preview directory
        if preview_dir not in target.parents:
            return False
        target.unlink()
        logger.info("delete_preview: removed %s", target)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("delete_preview: failed to remove %s: %s", target, exc)
        return False


@maybe_tool("render_template_tool")
async def render_template_tool(name: str, label: Optional[str] = None) -> dict:
    """Render a named template into a state and return { state } as dict."""
    try:
        exprs = render_template(name, {"label": label} if label else None)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(str(exc))
    return {"expressions": [e.model_dump(exclude_none=True) for e in exprs]}


@maybe_tool("preset_bounds_tool")
async def preset_bounds_tool(name: str) -> dict:
    """Return preset bounds by name."""
    try:
        b = preset_bounds(name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(str(exc))
    return {"left": b.left, "right": b.right, "top": b.top,
            "bottom": b.bottom}


@maybe_tool("list_templates")
async def list_templates() -> list[str]:
    """List available template names."""
    return list_available_templates()


@maybe_tool("list_bounds")
async def list_bounds() -> list[str]:
    """List available preset bounds names."""
    return list_available_bounds()


@maybe_tool("bulk_generate")
async def bulk_generate(
    latex_list: List[str],
    filename_prefix: Optional[str] = None,
    per_file_minify: bool = False,
    max_per_call: int = 20,
) -> List[dict]:
    """Generate multiple simple graphs efficiently,
    with a soft cap (default 20)."""
    if len(latex_list) > max_per_call:
        latex_list = latex_list[:max_per_call]
    results: List[dict] = []
    for idx, latex in enumerate(latex_list, start=1):
        fname = f"{filename_prefix or 'bulk'}-{idx}.html"
        out = await quick_plot(latex=latex, filename=fname)
        if per_file_minify:
            # Regenerate with minification using state from the simple expr
            expr = ExpressionState(latex=latex)
            state = DesmosState(expressions=[expr])
            html = await generate_embed_html(state=state, minify=True)
            await save_html(html=html, filename=fname)
            out["html"] = html
        results.append(out)
    return results


@maybe_tool("get_metrics")
async def get_metrics() -> dict:
    """Return basic counters for diagnostics."""
    return dict(_metrics)


@maybe_tool("cleanup_previews")
async def cleanup_previews(
    keep: int = 200,
    older_than_days: Optional[int] = None,
) -> int:
    """Cleanup preview/ directory; keep most recent N and optionally
    delete by age.

    Returns number of files deleted.
    """
    import time

    preview_dir = Path("preview").resolve()
    if not preview_dir.exists():
        return 0
    files = [p for p in preview_dir.glob("*.html") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    deleted = 0
    now = time.time()
    to_delete = files[keep:]
    for p in to_delete:
        if older_than_days is not None:
            age_days = (now - p.stat().st_mtime) / 86400.0
            if age_days < older_than_days:
                continue
        try:
            p.unlink()
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("cleanup_previews: failed to delete %s: %s", p, exc)
    return deleted


if __name__ == "__main__":
    mcp.run()
