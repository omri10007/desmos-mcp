# Desmos MCP Server

An MCP server that generates Desmos graph states and self-contained embed HTML, and saves previews locally under `preview/`.

[![CI](https://github.com/omri10007/desmos-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/omri10007/desmos-mcp/actions/workflows/ci.yml)

## Features
- Structured Pydantic models for Desmos expressions and state
- HTML generator using the official Desmos client JS API
- Safe local saving with filename sanitization and collision handling
- Convenience tools for quick plots and one-shot generate+save
- Logging with configurable level
- Strict LaTeX validation to prevent broken graphs
- Minimal toolset by default; extended toolset with an env toggle

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python -m desmos_mcp.server
```

- Minimal mode (default): registers 4 tools: `validate_expressions`, `generate_state`, `generate_embed_html`, `quick_plot`.
- Full mode (more tools):
  - PowerShell: `$env:DESMOS_MCP_TOOLSET = "full"; python -m desmos_mcp.server`
  - CMD: `set DESMOS_MCP_TOOLSET=full && python -m desmos_mcp.server`

## Tools (Minimal)
- `generate_state(expressions, math_bounds?, degree_mode?) -> DesmosState`
- `generate_embed_html(state, api_key?, options?, title?, save_to?, open_browser?, ctx?, minify?, script_url?) -> str`
- `quick_plot(latex, api_key?, options?, filename?, color?, hidden?, label?, show_label?) -> { path, html }`
- `validate_expressions(expressions) -> [{ latex, warnings[], blocked }]`

### Full mode extras
- `generate_and_save_embed_html(state, api_key?, options?, title?, filename?, minify?) -> { path, html }`
- `save_html(html, filename?) -> str`
- `open_in_browser(state, ...) -> { path, opened }`
- `open_saved_html(path) -> bool`
- `render_template_tool(name, label?) -> { expressions }`
- `preset_bounds_tool(name) -> { left,right,top,bottom }`
- `list_templates() -> [str]`
- `list_bounds() -> [str]`
- `create_supply_demand_state(...) -> DesmosState`
- `list_previews(limit=50, newest_first=True) -> [str]`
- `delete_preview(filename) -> bool`
- `get_metrics() -> { ... }`
- `cleanup_previews(keep=200, older_than_days?) -> int`
- `bulk_generate(latex_list, filename_prefix?, per_file_minify?, max_per_call=20) -> [{ path, html }]`

## Environment
- `DESMOS_API_KEY`: Overrides the default demo API key in embeds
- `DESMOS_MCP_LOG_LEVEL`: Set to `DEBUG`, `INFO` (default), `WARNING`, etc.
- `DESMOS_SCRIPT_URL`: Override the Desmos script URL
- `DESMOS_MCP_TOOLSET`: `minimal` (default) or `full`

## Development
```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
pytest -q
```

## License
MIT