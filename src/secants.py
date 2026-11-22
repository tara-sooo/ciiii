#!/usr/bin/env python3
"""Exact secant intersections on the three-distance genus-one quartics.

Combines points from the C++ seeds and optionally the existing tangent-walk
records. Unlike the tangent-only walk, a quadratic through TWO distinct
rational points supplies a third rational intersection. This is a finite,
bounded experiment, not a complete rational-point search.
"""
import argparse
from collections import defaultdict, deque
from fractions import Fraction as F
import gzip
import json
from math import gcd, isqrt
from pathlib import Path
import sys

from orbits import candidate, public_record, qpoly_coeff, quartic, seeds_to_initial_point, tangent

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def secant(s1, w1, s2, w2, cf, lead):
    """Third intersection of w = lead*s**2 + beta*s + gamma and w**2=f(s).

    The quartic terms cancel because lead is +/-1; the remaining cubic
    already has rational roots s1 and s2, hence its third root is rational.
    """
    if s1 == s2:
        return None
    beta = (w2 - w1) / (s2 - s1) - lead * (s1 + s2)
    gamma = w1 - lead * s1 * s1 - beta * s1
    g3 = cf[1] - 2 * lead * beta
    if not g3:
        return None
    g2 = cf[2] - beta * beta - 2 * lead * gamma
    s3 = -g2 / g3 - s1 - s2
    w3 = lead * s3 * s3 + beta * s3 + gamma
    if w3 * w3 != quartic(s3, cf):
        raise AssertionError("secant failed exact quartic identity")
    return s3, w3


def embed_record(p):
    """Recover an exact quartic point from any three-distance record.

    Unlike the seed-only embedding, this also accepts exterior points with
    negative X,Y (but the source slope must have same-sign nonzero legs).
    """
    x, y, q = (int(p[k]) for k in ("x", "y", "q"))
    d = [int(v) for v in p["d"]]
    if q <= 0 or x == 0 or y == 0 or (x > 0) != (y > 0):
        return None
    if gcd(gcd(abs(x), abs(y)), q) != 1:
        raise AssertionError("unreduced three-distance record")
    if (x*x+y*y != d[0]*d[0] or (q-x)**2+y*y != d[1]*d[1]
            or x*x+(q-y)**2 != d[2]*d[2]):
        raise AssertionError("invalid three-distance record")
    g = gcd(abs(x), abs(y))
    a, b = abs(x)//g, abs(y)//g
    h2 = a*a+b*b
    if isqrt(h2)**2 != h2 or gcd(a, b) != 1:
        raise AssertionError("record lacks a primitive Pythagorean slope")
    signed_g = g if x > 0 else -g
    s = F(d[1]-q, signed_g)
    D = h2 - s*s
    if not D or 2*(s+a)/D != F(signed_g, q):
        raise AssertionError("invalid conic inversion")
    w = F(d[2], q) * D
    cf = qpoly_coeff(a, b, h2)
    if w*w != quartic(s, cf):
        raise AssertionError("invalid quartic embedding")
    return a, b, h2, cf, s, w


def height(s, w):
    return max(s.numerator.bit_length(), s.denominator.bit_length(),
               w.numerator.bit_length(), w.denominator.bit_length())


def read_inputs(seed_files, orbit_files, tangent_depth, source_bits, source_q_bits):
    groups = defaultdict(dict)
    stats = dict(seed_files=len(seed_files), orbit_files=len(orbit_files),
                 input_seeds=0, orbit_rows=0, orbit_rows_skipped_q=0,
                 anchor_states=0, anchor_skipped_bits=0)
    for filename in seed_files:
        with open(filename, encoding="utf-8") as f:
            data = json.load(f)
        for p in data["seeds"]:
            a, b, h2, cf, s, w = seeds_to_initial_point(p)
            stats["input_seeds"] += 1
            queue = deque([(s, w, 0), (s, -w, 0)])
            local_seen = set()
            while queue:
                ss, ww, depth = queue.popleft()
                if height(ss, ww) > source_bits:
                    stats["anchor_skipped_bits"] += 1
                    continue
                if (ss, ww) in local_seen:
                    continue
                local_seen.add((ss, ww))
                groups[(a, b)][(ss, ww)] = (h2, cf, ss, ww)
                if depth >= tangent_depth:
                    continue
                for lead in (1, -1):
                    nxt = tangent(ss, ww, cf, lead)
                    if nxt is not None:
                        queue.append((*nxt, depth + 1))
    for filename in orbit_files:
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                stats["orbit_rows"] += 1
                if p["q_bits"] > source_q_bits:
                    stats["orbit_rows_skipped_q"] += 1
                    continue
                result = embed_record(p)
                if result is None:
                    continue
                a, b, h2, cf, s, w = result
                for ww in (w, -w):
                    if height(s, ww) <= source_bits:
                        groups[(a, b)][(s, ww)] = (h2, cf, s, ww)
                    else:
                        stats["anchor_skipped_bits"] += 1
    stats["anchor_states"] = sum(map(len, groups.values()))
    return groups, stats


def search(seed_files, orbit_files, output, records, tangent_depth,
           source_bits, source_q_bits, per_curve, pairs_per_curve,
           max_attempts, bit_cap):
    groups, stats = read_inputs(seed_files, orbit_files, tangent_depth,
                               source_bits, source_q_bits)
    stats.update(curves=len(groups), curves_with_pairs=0, pairs=0,
                 intersections=0, degenerate=0, source_hits=0,
                 denominator_skipped=0, duplicates=0,
                 three_distance_points=0, max_q_bits=0, found_four=0,
                 stopped_at_attempt_cap=False)
    seen_points = set()
    best, solutions = [], []
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(records).parent.mkdir(parents=True, exist_ok=True)
    stopped = False
    with gzip.open(records, "wt", encoding="utf-8") as dest:
        for curve in sorted(groups):
            points = sorted(groups[curve].values(),
                            key=lambda t: (height(t[2], t[3]), t[2], t[3]))
            points = points[:per_curve]
            source_s = {t[2] for t in groups[curve].values()}
            if len(source_s) < 2:
                continue
            stats["curves_with_pairs"] += 1
            a, b = curve
            h2, cf = points[0][:2]
            count = 0
            for i in range(len(points)):
                if stopped or count >= pairs_per_curve:
                    break
                for j in range(i):
                    if count >= pairs_per_curve:
                        break
                    s1, w1 = points[i][2:]
                    s2, w2 = points[j][2:]
                    if s1 == s2:
                        continue
                    count += 1
                    stats["pairs"] += 1
                    for lead in (1, -1):
                        if stats["intersections"] >= max_attempts:
                            stopped = True
                            break
                        stats["intersections"] += 1
                        result = secant(s1, w1, s2, w2, cf, lead)
                        if result is None:
                            stats["degenerate"] += 1
                            continue
                        s3, w3 = result
                        if s3 in source_s:
                            stats["source_hits"] += 1
                            continue
                        # A large s can make rational arithmetic unbounded;
                        # the endpoint cap prevents its expensive square tests.
                        if height(s3, w3) > bit_cap * 2:
                            stats["denominator_skipped"] += 1
                            continue
                        c = candidate(s3, w3, a, b, h2, cf, bit_cap)
                        if c is None:
                            continue
                        if c == "over":
                            stats["denominator_skipped"] += 1
                            continue
                        key = (c["x"], c["y"], c["q"])
                        if key in seen_points:
                            stats["duplicates"] += 1
                            continue
                        seen_points.add(key)
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
                            best.sort(key=lambda v: v[0])
                            if len(best) > 10:
                                best.pop()
                    if stopped:
                        break
            if stopped:
                break
    stats["stopped_at_attempt_cap"] = stopped
    summary = {
        "method": "exact-quartic-two-point-secant-v1",
        "scope": "finite bounded secants of exact three-distance anchors; not exhaustive",
        "parameters": dict(tangent_depth=tangent_depth, source_bits=source_bits,
                           source_q_bits=source_q_bits, per_curve=per_curve,
                           pairs_per_curve=pairs_per_curve,
                           max_attempts=max_attempts, max_q_bits=bit_cap),
        "stats": stats,
        "best_integer_root_near_misses": [rec for _, rec in best],
        "solutions": solutions,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"curves={stats['curves']} pairs={stats['pairs']} "
          f"3-distance={stats['three_distance_points']} "
          f"FOUR={stats['found_four']} max_q_bits={stats['max_q_bits']}")
    return summary


def self_test():
    p = {"x": "45", "y": "24", "q": "52", "d": ["51", "25", "53"]}
    a, b, h2, cf, s, w = seeds_to_initial_point(p)
    assert embed_record(p) == (a, b, h2, cf, s, w)
    nxt = tangent(s, w, cf, -1)
    assert nxt is not None
    assert secant(s, w, s, -w, cf, 1) is None
    z = secant(s, w, *nxt, cf, 1)
    assert z is not None and z[0] == F(-203, 11)
    c = candidate(*z, a, b, h2, cf, 8192)
    assert isinstance(c, dict) and c["gap"] != 0
    for lead in (1, -1):
        for first in ((s, w), (s, -w), nxt):
            for second in ((s, w), (s, -w), nxt):
                if first[0] == second[0]:
                    continue
                zz = secant(*first, *second, cf, lead)
                if zz is not None:
                    assert zz[1]*zz[1] == quartic(zz[0], cf)
                    cc = candidate(*zz, a, b, h2, cf, 8192)
                    assert cc is None or cc == "over" or isinstance(cc, dict)
    # End-to-end fixture: seed JSON + compressed tangent-orbit JSONL.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        seed_path = root / "seeds.json"
        orbit_path = root / "orbit.jsonl.gz"
        output_path = root / "summary.json"
        points_path = root / "points.jsonl.gz"
        seed_path.write_text(json.dumps({"seeds": [p]}), encoding="utf-8")
        orbit_candidate = candidate(*nxt, a, b, h2, cf, 8192)
        assert isinstance(orbit_candidate, dict)
        with gzip.open(orbit_path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(public_record(orbit_candidate)) + "\n")
        result = search([str(seed_path)], [str(orbit_path)],
                        str(output_path), str(points_path),
                        tangent_depth=1, source_bits=512, source_q_bits=256,
                        per_curve=12, pairs_per_curve=32,
                        max_attempts=100, bit_cap=512)
        stats = result["stats"]
        assert stats["input_seeds"] == 1 and stats["orbit_rows"] == 1
        assert stats["pairs"] > 0 and stats["three_distance_points"] > 0
        assert stats["found_four"] == 0
        assert json.loads(output_path.read_text())["stats"] == stats
        with gzip.open(points_path, "rt", encoding="utf-8") as f:
            assert sum(1 for _ in f) == stats["three_distance_points"]
    print("self-test passed: secant identities and end-to-end seed/orbit search")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--orbits", nargs="*", default=[])
    ap.add_argument("--output", default="secants-summary.json")
    ap.add_argument("--records", default="secants.jsonl.gz")
    ap.add_argument("--tangent-depth", type=int, default=2)
    ap.add_argument("--source-bits", type=int, default=512)
    ap.add_argument("--source-q-bits", type=int, default=256)
    ap.add_argument("--per-curve", type=int, default=12)
    ap.add_argument("--pairs-per-curve", type=int, default=32)
    ap.add_argument("--max-attempts", type=int, default=250000)
    ap.add_argument("--max-q-bits", type=int, default=8192)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        sys.exit(0)
    if not args.seeds and not args.orbits:
        ap.error("provide --seeds and/or --orbits")
    if not (0 <= args.tangent_depth <= 4 and 64 <= args.source_bits <= 2048
            and 64 <= args.source_q_bits <= 2048 and 2 <= args.per_curve <= 100
            and 1 <= args.pairs_per_curve <= 1000 and 1 <= args.max_attempts <= 2000000
            and 128 <= args.max_q_bits <= 8192):
        ap.error("resource limits exceeded")
    search(args.seeds, args.orbits, args.output, args.records,
           args.tangent_depth, args.source_bits, args.source_q_bits,
           args.per_curve, args.pairs_per_curve, args.max_attempts,
           args.max_q_bits)
