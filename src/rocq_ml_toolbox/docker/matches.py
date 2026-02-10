from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Key = Tuple[str, ...]  # relative logical path parts, without extension


@dataclass(frozen=True)
class MatchReport:
    """Diagnostics produced by `match_paths`."""
    output_root: Optional[Path]
    source_root: Optional[Path]
    unmatched_outputs: List[Path]
    unmatched_sources: List[Path]
    ambiguous: Dict[Path, List[Path]]  # output -> candidate sources


def match_paths(
    outputs: Sequence[Path],
    sources: Sequence[Path],
    *,
    min_unique_matches_for_root_inference: int = 3,
) -> tuple[Dict[Path, Path], MatchReport]:
    """
    Match paths from `outputs` to `sources` using both filename and directory structure.
    """
    out_root, src_root = _infer_roots(outputs, sources, min_unique_matches_for_root_inference)

    if out_root is not None and src_root is not None:
        mapping, report = _match_with_roots(outputs, sources, out_root, src_root)
        return mapping, report

    mapping, report = _match_by_suffix_similarity(outputs, sources)
    return mapping, report


# ----------------------- internals -----------------------


def _logical(p: Path) -> Path:
    """Path without its last suffix (e.g. X.v, X.vo -> X)."""
    return p.with_suffix("")


def _infer_roots(
    outputs: Sequence[Path],
    sources: Sequence[Path],
    min_unique_matches: int,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Choose (output_root, source_root) maximizing the number of *unique* matching keys.
    """
    out_root_to_counts = _root_to_key_counts(outputs)
    src_root_to_counts = _root_to_key_counts(sources)

    out_key_to_roots = _unique_key_to_roots(out_root_to_counts)
    src_key_to_roots = _unique_key_to_roots(src_root_to_counts)

    # Score root pairs by:
    #   (1) weighted sum of key lengths (prefers preserving directory structure)
    #   (2) number of matched keys
    #   (3) depth of roots (tie-breaker: ignore larger prefixes when equally good)
    scores: DefaultDict[tuple[Path, Path], List[int]] = defaultdict(lambda: [0, 0])

    for key, out_roots in out_key_to_roots.items():
        src_roots = src_key_to_roots.get(key)
        if not src_roots:
            continue
        w = len(key)
        for ro in out_roots:
            for rs in src_roots:
                s = scores[(ro, rs)]
                s[0] += w
                s[1] += 1

    if not scores:
        return None, None

    (best_out, best_src), (best_wsum, best_cnt) = max(
        scores.items(),
        key=lambda it: (
            it[1][0],  # weighted sum
            it[1][1],  # count
            len(it[0][0].parts) + len(it[0][1].parts),  # depth tie-break
        ),
    )

    if best_cnt < min_unique_matches:
        return None, None

    return best_out, best_src


def _root_to_key_counts(paths: Sequence[Path]) -> Mapping[Path, Counter[Key]]:
    """
    For each candidate root, count how many times each key appears under that root.
    """
    root_to_counts: DefaultDict[Path, Counter[Key]] = defaultdict(Counter)

    for p in paths:
        lp = _logical(p)
        for root in lp.parents:  # candidate roots are ancestor directories
            # lp is always relative_to any of its parents
            key = tuple(lp.relative_to(root).parts)
            root_to_counts[root][key] += 1

    return root_to_counts


def _unique_key_to_roots(root_to_counts: Mapping[Path, Counter[Key]]) -> Dict[Key, List[Path]]:
    """Invert root->counts into key->roots, keeping only keys unique under each root."""
    key_to_roots: DefaultDict[Key, List[Path]] = defaultdict(list)
    for root, counts in root_to_counts.items():
        for key, c in counts.items():
            if c == 1:
                key_to_roots[key].append(root)
    return dict(key_to_roots)


def _match_with_roots(
    outputs: Sequence[Path],
    sources: Sequence[Path],
    out_root: Path,
    src_root: Path
) -> tuple[Dict[Path, Path], MatchReport]:
    out_key_to_paths, out_outside = _index_under_root(outputs, out_root)
    src_key_to_paths, src_outside = _index_under_root(sources, src_root)

    mapping: Dict[Path, Path] = {}
    ambiguous: Dict[Path, List[Path]] = {}
    unmatched_outputs: List[Path] = []
    used_sources: set[Path] = set()

    for key, outs in out_key_to_paths.items():
        if len(outs) != 1:
            # leave them unmatched
            unmatched_outputs.extend(outs)
            continue

        out_path = outs[0]
        cands = src_key_to_paths.get(key, [])
        if not cands:
            unmatched_outputs.append(out_path)
            continue
        if len(cands) > 1:
            ambiguous[out_path] = list(cands)
            continue

        src_path = cands[0]
        mapping[out_path] = src_path
        used_sources.add(src_path)

    unmatched_sources = [p for p in sources if p not in used_sources and p not in src_outside]

    return mapping, MatchReport(
        output_root=out_root,
        source_root=src_root,
        unmatched_outputs=unmatched_outputs,
        unmatched_sources=unmatched_sources,
        ambiguous=ambiguous,
    )


def _index_under_root(paths: Sequence[Path], root: Path) -> tuple[Dict[Key, List[Path]], List[Path]]:
    key_to_paths: DefaultDict[Key, List[Path]] = defaultdict(list)
    outside: List[Path] = []

    for p in paths:
        lp = _logical(p)
        try:
            rel = lp.relative_to(root)
        except ValueError:
            outside.append(p)
            continue
        key_to_paths[tuple(rel.parts)].append(p)

    return dict(key_to_paths), outside


def _match_by_suffix_similarity(
    outputs: Sequence[Path],
    sources: Sequence[Path]
) -> tuple[Dict[Path, Path], MatchReport]:
    """
    Fallback: match by equal stem, then maximize common suffix length of logical parts.
    """
    src_by_stem: DefaultDict[str, List[Path]] = defaultdict(list)
    for s in sources:
        src_by_stem[_logical(s).name].append(s)

    mapping: Dict[Path, Path] = {}
    ambiguous: Dict[Path, List[Path]] = {}
    unmatched_outputs: List[Path] = []
    used_sources: set[Path] = set()

    for out in outputs:
        stem = _logical(out).name
        cands = src_by_stem.get(stem, [])
        if not cands:
            unmatched_outputs.append(out)
            continue

        best = _best_by_common_suffix(_logical(out), [_logical(c) for c in cands])
        if best is None:
            ambiguous[out] = list(cands)
            continue

        # best is a logical Path; recover original candidate with same logical
        chosen = next(c for c in cands if _logical(c) == best)
        mapping[out] = chosen
        used_sources.add(chosen)

    unmatched_sources = [s for s in sources if s not in used_sources]

    return mapping, MatchReport(
        output_root=None,
        source_root=None,
        unmatched_outputs=unmatched_outputs,
        unmatched_sources=unmatched_sources,
        ambiguous=ambiguous,
    )


def _best_by_common_suffix(out_logical: Path, cand_logicals: Sequence[Path]) -> Optional[Path]:
    """
    Pick the unique candidate maximizing the number of matching suffix components.
    Return None if the maximum is not unique.
    """
    out_parts = out_logical.parts
    best_len = -1
    best: List[Path] = []

    for c in cand_logicals:
        n = _common_suffix_len(out_parts, c.parts)
        if n > best_len:
            best_len = n
            best = [c]
        elif n == best_len:
            best.append(c)

    return best[0] if len(best) == 1 else None


def _common_suffix_len(a: Sequence[str], b: Sequence[str]) -> int:
    i = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        i += 1
    return i
