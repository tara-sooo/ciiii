#!/usr/bin/env python3
"""Round-aware two-point secant exploration of exact rational square distances.

Selects height-diverse anchors and prioritizes pairs containing a newly
generated point. Earlier recorded three-distance points are skipped in the
output. All intersections and square tests use exact rational/integer math.
This is a bounded search, never an exhaustive denominator theorem.
"""
import argparse
from collections import defaultdict
import gzip
from hashlib import blake2b
import json
from pathlib import Path
import sys

from secants import height, read_inputs, secant
from orbits import candidate, public_record

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def spaced(items, count):
    """Pick s-distinct, height-distributed anchors; add opposite signs last."""
    if count <= 0:
        return []
    bins = defaultdict(list)
    for item in items:
        bins[item[2]].append(item)
    primary = []
    for s in sorted(bins):
        pool = bins[s]
        # Each s gets one representative; avoid spending slots on +/-w first.
        primary.append(min(pool, key=lambda p: (height(p[2], p[3]), p[3] < 0, p[3])))
    primary.sort(key=lambda p: (height(p[2], p[3]), p[2], p[3]))
    if not primary:
        return []
    n = len(primary)
    if n <= count:
        picked = list(primary)
    else:
        # Uniform quantiles cover high- and low-height regions rather than
        # taking the first N shortest anchors, as the prior script did.
        idx = sorted({(i * (n - 1)) // (count - 1) for i in range(count)}) if count > 1 else [n // 2]
        picked = [primary[i] for i in idx]
    if len(picked) < count:
        chosen = {(v[2], v[3]) for v in picked}
        # Only after using distinct s do we allow the opposite curve sheet.
        others = sorted((v for v in items if (v[2], v[3]) not in chosen),
                        key=lambda p: (height(p[2], p[3]), p[2], p[3]))
        picked.extend(others[:count-len(picked)])
    return picked


def selected_anchors(base, fresh, cap):
    """Balanced selection with at least one fresh point and distinct s."""
    if not fresh:
        return [], []
    fresh_quota = max(1, cap // 2)
    recent = spaced(fresh, fresh_quota)
    old = spaced(base, cap - len(recent))
    if len(old) + len(recent) < cap:
        # More novel choices when the curve has few old anchors.
        chosen = {(p[2], p[3]) for p in recent}
        remaining = [p for p in fresh if (p[2], p[3]) not in chosen]
        recent.extend(spaced(remaining, cap - len(old) - len(recent)))
    return recent, old


def pair_candidates(recent, old, limit):
    """Interleave high/low-height cross-generation and new/new pairs."""
    crosses = [(p, q, "cross") for p in recent for q in old if p[2] != q[2]]
    novel = [(p, q, "fresh") for i, p in enumerate(recent) for q in recent[:i]
             if p[2] != q[2]]
    def rank(pair):
        p, q, _ = pair
        hp = height(p[2], p[3])
        hq = height(q[2], q[3])
        return (-abs(hp-hq), hp + hq, p[2], q[2])
    crosses.sort(key=rank)
    novel.sort(key=rank)
    # Reserve roughly one quarter of the trial budget for fresh/fresh.
    cross_budget = min(len(crosses), (limit * 3 + 3) // 4)
    novel_budget = min(len(novel), limit-cross_budget)
    if cross_budget + novel_budget < limit:
        cross_budget = min(len(crosses), limit - novel_budget)
        novel_budget = min(len(novel), limit - cross_budget)
    return crosses[:cross_budget] + novel[:novel_budget]


def fingerprint(a, b, p, q):
    """Stable hash of the unordered exact secant inputs (for future rounds)."""
    x = (p[2].numerator, p[2].denominator, p[3].numerator, p[3].denominator)
    y = (q[2].numerator, q[2].denominator, q[3].numerator, q[3].denominator)
    if y < x:
        x, y = y, x
    payload = repr((a, b, x, y)).encode("ascii")
    return blake2b(payload, digest_size=20).hexdigest()


def read_point_keys(paths):
    known = set()
    total = 0
    for filename in paths:
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                known.add((str(p["x"]), str(p["y"]), str(p["q"])))
                total += 1
    return known, total


def run(seeds, base_files, fresh_files, prior_pair_files, output, records,
        pair_log, source_bits, source_q_bits, per_curve, pairs_per_curve,
        max_attempts, bit_cap):
    base_groups, base_stats = read_inputs(seeds, base_files, 0,
                                          source_bits, source_q_bits)
    fresh_groups, fresh_stats = read_inputs([], fresh_files, 0,
                                            source_bits, source_q_bits)
    known, prior_rows = read_point_keys(base_files + fresh_files)
    # The seed stage is not compressed; its embedded point is in source_s
    # and is therefore omitted without requiring a second JSON read.
    prior_pairs = set()
    for filename in prior_pair_files:
        with gzip.open(filename, "rt", encoding="ascii") as f:
            for line in f:
                prior_pairs.add(line.strip())
    groups = set(base_groups) | set(fresh_groups)
    stats = {
        "base_rows": base_stats["orbit_rows"],
        "fresh_rows": fresh_stats["orbit_rows"],
        "prior_rows": prior_rows,
        "prior_distinct_points": len(known),
        "curves": len(groups), "curves_with_fresh": 0,
        "pairs": 0, "cross_pairs": 0, "fresh_pairs": 0,
        "previous_pairs_skipped": 0, "intersections": 0,
        "degenerate": 0, "source_hits": 0,
        "known_point_skipped": 0, "duplicates": 0,
        "denominator_skipped": 0, "three_distance_points": 0,
        "max_q_bits": 0, "found_four": 0,
        "stopped_at_attempt_cap": False,
    }
    seen = set()
    pair_seen = set(prior_pairs)
    best, solutions = [], []
    for path in (output, records, pair_log):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    stopping = False
    with gzip.open(records, "wt", encoding="utf-8") as dest, \
         gzip.open(pair_log, "wt", encoding="ascii") as pair_dest:
        for a, b in sorted(groups):
            old = base_groups.get((a,b), {})
            newer = fresh_groups.get((a,b), {})
            # A point already found in an old source is not a new anchor.
            pure_fresh = [v for k, v in newer.items() if k not in old]
            if not pure_fresh:
                continue
            stats["curves_with_fresh"] += 1
            recent, older = selected_anchors(list(old.values()), pure_fresh, per_curve)
            if not older and len(recent) < 2:
                continue
            h2, cf = (recent or older)[0][:2]
            source_s = {key[0] for key in old} | {key[0] for key in newer}
            for p, q, kind in pair_candidates(recent, older, pairs_per_curve):
                if stats["intersections"] + 2 > max_attempts:
                    stopping = True
                    break
                code = fingerprint(a, b, p, q)
                if code in pair_seen:
                    stats["previous_pairs_skipped"] += 1
                    continue
                pair_seen.add(code)
                pair_dest.write(code + "\n")
                stats["pairs"] += 1
                stats["cross_pairs" if kind == "cross" else "fresh_pairs"] += 1
                for lead in (1, -1):
                    stats["intersections"] += 1
                    result = secant(p[2], p[3], q[2], q[3], cf, lead)
                    if result is None:
                        stats["degenerate"] += 1
                        continue
                    s, w = result
                    if s in source_s:
                        stats["source_hits"] += 1
                        continue
                    if height(s, w) > bit_cap * 2:
                        stats["denominator_skipped"] += 1
                        continue
                    c = candidate(s, w, a, b, h2, cf, bit_cap)
                    if c is None:
                        continue
                    if c == "over":
                        stats["denominator_skipped"] += 1
                        continue
                    key = (str(c["x"]), str(c["y"]), str(c["q"]))
                    if key in known:
                        stats["known_point_skipped"] += 1
                        continue
                    if key in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(key)
                    stats["three_distance_points"] += 1
                    stats["max_q_bits"] = max(stats["max_q_bits"], c["q_bits"])
                    rec = public_record(c)
                    dest.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    if c["gap"] == 0:
                        stats["found_four"] += 1
                        solutions.append(rec)
                        print("POTENTIAL FOUR-DISTANCE SOLUTION!", flush=True)
                    else:
                        best.append((c["gap_log10"], rec))
                        best.sort(key=lambda x: x[0])
                        if len(best) > 10:
                            best.pop()
            if stopping:
                break
    stats["stopped_at_attempt_cap"] = stopping
    obj = {
        "method": "height-diverse-fresh-cross-generation-secants-v1",
        "scope": "bounded selection; intersections and distance identities are exact; not exhaustive",
        "parameters": dict(source_bits=source_bits, source_q_bits=source_q_bits,
                           per_curve=per_curve, pairs_per_curve=pairs_per_curve,
                           max_attempts=max_attempts, max_q_bits=bit_cap),
        "stats": stats,
        "solutions": solutions,
        "best_integer_root_near_misses": [rec for _, rec in best],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"curves={stats['curves']} fresh_curves={stats['curves_with_fresh']} "
          f"pairs={stats['pairs']} fresh_pairs={stats['fresh_pairs']} "
          f"3-distance={stats['three_distance_points']} FOUR={stats['found_four']} "
          f"max_q_bits={stats['max_q_bits']}")
    return obj


def self_test():
    from tempfile import TemporaryDirectory
    from fractions import Fraction as F
    from secants import embed_record
    from orbits import seeds_to_initial_point, tangent
    p = {"x": "45", "y": "24", "q": "52", "d": ["51", "25", "53"]}
    a, b, h2, cf, s, w = seeds_to_initial_point(p)
    nxt = tangent(s, w, cf, -1)
    assert nxt is not None
    fresh_c = candidate(*nxt, a, b, h2, cf, 1024)
    assert isinstance(fresh_c, dict)
    assert embed_record(public_record(fresh_c)) is not None
    assert len(spaced([(h2,cf,s,w), (h2,cf,s,-w)], 2)) == 2
    pair = pair_candidates([(h2,cf,*nxt)], [(h2,cf,s,w)], 4)
    assert len(pair) == 1 and pair[0][2] == "cross"
    assert fingerprint(a,b,pair[0][0],pair[0][1]) == fingerprint(a,b,pair[0][1],pair[0][0])
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        seeds = directory / "seeds.json"
        fresh = directory / "fresh.jsonl.gz"
        summary = directory / "summary.json"
        points = directory / "points.jsonl.gz"
        pair_log = directory / "pairlog.gz"
        seeds.write_text(json.dumps({"seeds":[p]}), encoding="utf-8")
        with gzip.open(fresh, "wt", encoding="utf-8") as f:
            f.write(json.dumps(public_record(fresh_c)) + "\n")
        out = run([str(seeds)], [], [str(fresh)], [],
                  str(summary), str(points), str(pair_log),
                  source_bits=512, source_q_bits=512, per_curve=8,
                  pairs_per_curve=24, max_attempts=100, bit_cap=1024)
        assert out["stats"]["cross_pairs"] > 0
        assert out["stats"]["three_distance_points"] > 0
        assert out["stats"]["found_four"] == 0
        with gzip.open(points, "rt", encoding="utf-8") as f:
            pts = [json.loads(line) for line in f]
        assert len(pts) == out["stats"]["three_distance_points"]
        with gzip.open(pair_log, "rt", encoding="ascii") as f:
            ids = [line.strip() for line in f]
        assert len(ids) == out["stats"]["pairs"]
        assert len(set(ids)) == len(ids)
        # Ledger prevents re-evaluation of the same trial.
        out2 = run([str(seeds)], [], [str(fresh)], [str(pair_log)],
                   str(directory / "summary2.json"), str(directory / "points2.gz"),
                   str(directory / "pairlog2.gz"),
                   source_bits=512, source_q_bits=512, per_curve=8,
                   pairs_per_curve=24, max_attempts=100, bit_cap=1024)
        assert out2["stats"]["pairs"] == 0
        assert out2["stats"]["previous_pairs_skipped"] == len(ids)
    print("self-test passed: fresh/old pairs, exact new points, and ledger skip")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--fresh", nargs="*", default=[])
    ap.add_argument("--prior-pairs", nargs="*", default=[])
    ap.add_argument("--output", default="local-secants/round3-summary.json")
    ap.add_argument("--records", default="local-secants/round3-points.jsonl.gz")
    ap.add_argument("--pair-log", default="local-secants/round3-pairs.txt.gz")
    ap.add_argument("--source-bits", type=int, default=1024)
    ap.add_argument("--source-q-bits", type=int, default=2048)
    ap.add_argument("--per-curve", type=int, default=32)
    ap.add_argument("--pairs-per-curve", type=int, default=256)
    ap.add_argument("--max-attempts", type=int, default=2500000)
    ap.add_argument("--max-q-bits", type=int, default=8192)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        sys.exit(0)
    if not args.fresh or (not args.seeds and not args.base):
        ap.error("provide --fresh and --seeds and/or --base")
    if not (64 <= args.source_bits <= 4096
            and 64 <= args.source_q_bits <= 4096
            and 4 <= args.per_curve <= 100
            and 1 <= args.pairs_per_curve <= 2000
            and 2 <= args.max_attempts <= 4000000
            and 128 <= args.max_q_bits <= 8192):
        ap.error("resource limits exceeded")
    run(args.seeds, args.base, args.fresh, args.prior_pairs,
        args.output, args.records, args.pair_log,
        args.source_bits, args.source_q_bits, args.per_curve,
        args.pairs_per_curve, args.max_attempts, args.max_q_bits)
