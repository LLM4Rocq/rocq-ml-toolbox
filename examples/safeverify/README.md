# SafeVerify Examples

This folder contains small Rocq files to exercise `rocq-ml-safeverify`.

## Run Commands

From repo root, either use the installed script:

```bash
rocq-ml-safeverify <source.v> <target.v> --root examples/safeverify -v
```

or run the module directly:

```bash
python -m rocq_ml_toolbox.safeverify.cli <source.v> <target.v> --root examples/safeverify -v
```

## 1) Non-trivial passing example

```bash
python -m rocq_ml_toolbox.safeverify.cli \
  examples/safeverify/nontrivial/Source.v \
  examples/safeverify/nontrivial/TargetGood.v \
  --root examples/safeverify -v
```

This uses non-trivial list proofs (`rev_app_distr_user`, `rev_involutive_user`) with induction and helper rewriting.

## 2) Statement-mismatch failing example

```bash
python -m rocq_ml_toolbox.safeverify.cli \
  examples/safeverify/nontrivial/Source.v \
  examples/safeverify/nontrivial/TargetBadStatement.v \
  --root examples/safeverify -v
```

Expected failure includes `statement_mismatch`.

## 3) Incomplete-proof failing example

```bash
python -m rocq_ml_toolbox.safeverify.cli \
  examples/safeverify/nontrivial/Source.v \
  examples/safeverify/nontrivial/TargetAdmitted.v \
  --root examples/safeverify -v
```

Expected failure includes `incomplete_proof`.

## 4) Whitelist behavior

Fail without whitelist:

```bash
python -m rocq_ml_toolbox.safeverify.cli \
  examples/safeverify/whitelist/Source.v \
  examples/safeverify/whitelist/TargetNeedsWhitelist.v \
  --root examples/safeverify -v
```

Pass with whitelist:

```bash
python -m rocq_ml_toolbox.safeverify.cli \
  examples/safeverify/whitelist/Source.v \
  examples/safeverify/whitelist/TargetNeedsWhitelist.v \
  --root examples/safeverify \
  --axiom-whitelist examples/safeverify/whitelist/whitelist.json -v
```

Whitelist files are plain JSON string lists, e.g. `["fake_oracle"]`.

## Via Inference Server

If `rocq-ml-server` is running, you can run the same check through the HTTP API:

```bash
curl -X POST http://127.0.0.1:5000/safeverify \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "examples/safeverify/nontrivial/Source.v",
    "target": "examples/safeverify/nontrivial/TargetGood.v",
    "root": "examples/safeverify",
    "axiom_whitelist": [],
    "verbose": true
  }'
```
