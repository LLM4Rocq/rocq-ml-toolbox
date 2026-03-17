# Zygote V1 Design (Revised)

## Scope
Design a robust V1 architecture for profile caching with:
- one **zygote pool** (templates),
- one **active pool** (serving workers),
- deterministic dedup for profile builds,
- explicit per-profile load scaling.

This is design-only. No runtime behavior is changed by this file.

## Two Critical Guarantees

### G1: Same-profile restart
If active worker `A` fails while serving profile `P`, `A` must restart from **zygote(P)**.

It must never restart from another profile as a default fallback.

### G2: Hot profiles are not single-worker bottlenecks
One zygote for profile `P` can feed multiple active workers for `P`.
Load is carried by active workers, not by zygotes.

## Non-goals (V1)
- No import "light/heavy" pruning in keying decisions.
- No cross-machine cache sharing.
- No speculative profile merging.

## Architecture

### Zygote Pool
- Role: maintain prewarmed templates by profile.
- Not request-serving.
- Typically one zygote per profile in V1 (bounded by limits).

### Active Pool
- Role: execute requests.
- Workers are assigned a single profile at a time.
- Active count is managed per-profile (autoscale within global limits).

## Profile Keying (V1)
- Profile key source: normalized header/import context.
- Canonicalization:
  - remove comments and blank lines,
  - normalize whitespace,
  - keep import order,
  - remove exact duplicates.
- Key:
  - `profile_key = "p-" + sha1(canonical_header)[:16]`

V1 rule: all normalized imports participate in key.

## Redis Keys (Proposed)

### Profile-level zygote state
- `zygote:profile:{pk}:state` JSON:
  - `status`: `MISSING|BUILDING|READY|FAILED|EVICTING`
  - `epoch`: int
  - `owner`: string
  - `updated_at`: float
  - `error`: optional string
- `zygote:profile:{pk}:lock`
- `zygote:profile:{pk}:event` (PubSub)
- `zygote:profile:{pk}:meta` JSON:
  - `canonical_header`
  - `build_ms`
  - `rss_mb`
  - `last_used_at`
  - `use_count`

### Active worker placement
- Existing keys:
  - `pet_status:{idx}`
  - `pet_profile:{idx}`
  - `generation:{idx}`
  - `pet_lock:{idx}`
- New optional counters:
  - `active:profile:{pk}:count`
  - `active:profile:{pk}:queue_depth` (best-effort)

### Optional telemetry-only import stats
- `zygote:import_stats:{hash}` JSON:
  - `count`
  - `ewma_ms`
  - `ewma_rss_delta_mb`
  - `updated_at`

## State Machines

### Zygote profile
- `MISSING -> BUILDING -> READY`
- `BUILDING -> FAILED`
- `READY -> EVICTING -> MISSING`
- `FAILED -> BUILDING` (retry with backoff)

Rules:
- `epoch` increments on each `BUILDING`.
- Only lock owner for that `epoch` may publish final result.

### Active worker
- `DOWN -> STARTING -> WARMING -> OK`
- `OK -> RESTART_NEEDED -> RESTARTING -> WARMING -> OK`
- Logical `LEASED` state is represented by held `pet_lock:{idx}`.

Rules:
- `generation` increments on each restart/rematerialization.
- Worker usable iff `status=OK` and `pet_profile` matches requested profile.

## Control Flows

### 1) Ensure zygote(profile_key)
1. Read `zygote:profile:{pk}:state`.
2. If `READY`, return.
3. Acquire `zygote:profile:{pk}:lock`.
4. Re-check state (double-check).
5. If still not `READY`, mark `BUILDING` with new epoch.
6. Build zygote template.
7. Mark `READY` or `FAILED`, publish event.
8. Release lock.

Waiters subscribe/poll until `READY` or timeout.

This guarantees dedup under concurrent same-profile requests.

### 2) Acquire active worker(profile_key)
1. Prefer `OK` + matching-profile idle workers.
2. If none, ensure zygote(profile_key).
3. If per-profile active target not reached, materialize one more active worker from zygote(profile_key).
4. Else wait briefly for matching worker availability.
5. As last resort, reprofile an idle worker (from profile Q to P) by rematerializing from zygote(P).

Selection policy in V1:
- least-loaded matching worker first,
- tie-break by oldest idle.

### 3) Active worker failure
When active idx `i` fails:
1. Read `failed_profile = pet_profile:{i}`.
2. Run `ensure_zygote(failed_profile)`.
3. Restart/rematerialize worker `i` from zygote(failed_profile).
4. Increment `generation:{i}`.
5. Return to `OK` only when profile matches and probe passes.

No default profile fallback for this path.

### 4) Per-profile scaling (simple V1 policy)
Maintain target active counts per profile:
- `min_active_per_profile` (default 0 or 1 for hot profiles),
- `max_active_per_profile`.

Scale up trigger:
- queue depth for profile > 0 and spare global capacity.

Scale down trigger:
- worker idle for `IDLE_EVICT_S` and profile has excess over min.

This avoids "one hot profile -> one overloaded worker".

## Materialization API
Hide clone implementation behind:
- `materialize_active_from_zygote(profile_key, active_idx)`

Backend options:
1. true clone/fork (if supported),
2. logical clone (bootstrap/snapshot artifact replay).

The orchestration contract is identical.

## Failure and Staleness
- Builder heartbeat while `BUILDING`.
- If build exceeds `ZYGOTE_BUILD_TIMEOUT_S` and owner stale:
  - mark failed for that epoch,
  - next waiter retries build.
- If zygote for required profile cannot be built:
  - active stays `RESTART_NEEDED`,
  - retry with exponential backoff,
  - surface profile-specific degradation in health.

## Eviction

### Zygotes
- Evict only when over zygote limits.
- Use LRU among profiles with zero active dependents.
- Never evict a zygote currently in `BUILDING`.

### Active workers
- Existing RAM/restart policy remains.
- Scale-down policy may retire idle workers above profile target.

## Config (Proposed)
- `ENABLE_ZYGOTE_POOL`
- `MAX_ZYGOTE_TOTAL`
- `MAX_ZYGOTE_PER_PROFILE`
- `MAX_RAM_PER_ZYGOTE_MB`
- `PET_ZYGOTE_START_PORT`
- `ZYGOTE_BUILD_TIMEOUT_S`
- `ZYGOTE_HEARTBEAT_TTL_S`
- `MAX_ACTIVE_TOTAL`
- `MAX_ACTIVE_PER_PROFILE`
- `MIN_ACTIVE_PER_PROFILE`
- `IDLE_EVICT_S`

## Test Matrix (Must Have)

### Concurrency
- two concurrent `ensure_zygote(P)` -> exactly one build.
- concurrent `ensure_zygote(P)` and `ensure_zygote(Q)` independent.

### Restart correctness
- worker with profile `P` crashes -> restarts from zygote `P`, not other.

### Load distribution
- hot profile with demand >1 gets >1 active workers (within limits).
- requests are spread across matching workers, not pinned to one.

### Failure recovery
- stale builder is fenced and replaced.
- failed zygote build does not corrupt other profiles.

### Eviction safety
- zygote with active dependents is not evicted.

## Why this is still a simple V1
- Single deterministic keying.
- Single dedup mechanism (per-profile lock + state).
- Explicit restart rule (same-profile only).
- Explicit scaling rule (active per profile).
- No heuristic import pruning yet.
