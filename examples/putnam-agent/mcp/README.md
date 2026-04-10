# DocQ FastMCP Server

Small FastMCP server exposing the DocQ workspace/state/tactic/import/intermediate-lemma tooling.

## Start

From repo root:

```bash
python examples/putnam-agent/mcp/docq_fastmcp.py --host 0.0.0.0 --port 8012
```

Default transport is `http`.

## Session Configuration

Configuration can be passed either:
- via MCP HTTP query params (recommended), or
- via environment variables.

Example URL (all optional):

```text
http://127.0.0.1:8012/mcp?
  source=examples/putnam-agent/putnam/mathcomp/putnam_1962_a6.v&
  env=coq-mathcomp&
  host=127.0.0.1&
  port=5000&
  timeout=90&
  semantic_base_url=http://127.0.0.1:8010&
  semantic_env=coq-mathcomp
```

Important params:
- `source` / `DOCQ_MCP_SOURCE`
- `env` / `DOCQ_MCP_ENV`
- `host`, `port`, `timeout` (Petanque backend)
- `semantic_base_url`, `semantic_route`, `semantic_api_key`, `semantic_env`
- `include_semantic_tool` or `disable_semantic_tool`
- `openrouter_api_key`, `openrouter_model`, `openrouter_base_url`

If OpenRouter config is not provided, you can still complete intermediate lemmas by proving them manually with `pending_lemma_*` tools and then calling `prove_intermediate_lemma`.

## Exposed Tools

- Session: `session_info`, `reset_session`
- Workspace/doc graph: `list_docs`, `checkout_doc`, `show_doc`
- Status/state/goals: `completion_status`, `current_head`, `list_states`, `get_goals`
- Tactics: `run_tac`, `run_tac_latest`
- Source access: `read_workspace_source`, `read_source_file`
- Library discovery: `explore_toc`, `semantic_doc_search`
- Imports: `add_import`, `remove_import`, `require_import` (alias)
- Intermediate lemmas:
  - `prepare_intermediate_lemma`
  - `prove_intermediate_lemma`
  - `add_intermediate_lemma`
  - `drop_pending_intermediate_lemma`
  - `list_pending_intermediate_lemmas`
  - `pending_lemma_current_head`
  - `pending_lemma_list_states`
  - `pending_lemma_get_goals`
  - `pending_lemma_run_tac`
  - `remove_intermediate_lemma`
- Materialization/validation: `materialized_source`, `validate_final_qed`

## Notes

- Each MCP session gets its own DocQ session/workspace.
- `reset_session` rebuilds a fresh workspace for the same MCP session key.
- Tools are serialized per session with an async lock.
- `list_docs` and `list_pending_intermediate_lemmas` return object payloads with
  stable keys (`docs` / `pending_lemmas`) for predictable structured clients.
- If no subagent model is configured, you can still prove prepared intermediate
  lemmas manually with `pending_lemma_*` tools, then finalize via
  `prove_intermediate_lemma`.
- In that manual mode, `add_intermediate_lemma` returns an `ok=true`
  `prepare_pending_manual` handoff payload (with `pending=true`) so clients can
  continue immediately with `pending_lemma_*`.
- `validate_final_qed` performs both checks:
  - explicit replay to final `Qed.` in a fresh session,
  - then `safeverify` against the original source file.
