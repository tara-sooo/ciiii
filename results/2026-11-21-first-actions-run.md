# First GitHub Actions search — 2026-11-21

- Repository: https://github.com/tara-sooo/ciiii
- Workflow run: https://github.com/tara-sooo/ciiii/actions/runs/44910213867
- Code commit: `f3268319f755a57536c9c2131c3aecb62eba25a1`
- Result: **success**; all four jobs completed. **No exact four-distance witness**.

This run covered a single partition of the exhaustive *ordered primitive Pythagorean triple pair* enumeration with Euclid's m <= 900. **It is not exhaustive over rational plane points or denominators.**

| Shard | Ordered pairs | Three-distance seeds (unique within shard) | Additional three-distance orbit points (unique within shard) | Orbit points with reduced denominator > 50 billion | Orbit points outside unit square | Exact four-distance points |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 963,537,428 | 18 | 267 | 245 | 132 | 0 |
| 1 | 963,537,428 | 32 | 470 | 434 | 231 | 0 |
| 2 | 963,537,428 | 21 | 307 | 286 | 168 | 0 |
| 3 | 963,537,428 | 27 | 366 | 339 | 184 | 0 |
| **Sum** | **3,854,149,712** | **98** | **1,410** | **1,304** | **715** | **0** |

Counts of unique points are deduplicated *within each shard*, not globally across all shards. The run logs, exact seed JSON, compressed orbit-point JSONL and summaries were saved as GitHub Actions artifacts with 30-day retention. Artifact IDs: 14709288334 (shard 0), 14708479754 (shard 1), 14709726130 (shard 2), 14709347690 (shard 3).

The next main-branch commit `da0cbb0d968f22372faf7e1d860d3272be363710` extended the schedule into non-overlapping parameter bands. Later runs use that updated implementation and should not be attributed to the original source snapshot.

**Interpretation:** exact three-rational-distance points at very large reduced denominators have been generated and checked, but no point with all four distances rational was found in this run. Neither a null result nor extremely close approximations constitute evidence of global nonexistence.
