#!/usr/bin/env python3
"""Bounded exact three-distance exploration with changing reference corners.

Known corners: A=(0,0), B=(1,0), C=(0,1). For each point, fix the
primitive slope at A, B or C, then generate rational intersections on
the corresponding quartic. Exterior points may be used as intermediates.
Only interior four-distance witnesses trigger the success alert.
"""
import argparse
from collections import deque
from fractions import Fraction as F
import gzip
import json
from math import gcd, isqrt, lcm
from pathlib import Path
import sys

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def size(*values):
    return max(max(v.numerator.bit_length(), v.denominator.bit_length())
               for v in values)


def squared(X, Y, Q):
    return (X*X+Y*Y, (Q-X)**2+Y*Y, X*X+(Q-Y)**2,
            (Q-X)**2+(Q-Y)**2)


def certify(X, Y, Q):
    assert Q > 0 and gcd(gcd(X, Y), Q) == 1
    ns = squared(X, Y, Q)
    roots = tuple(isqrt(n) for n in ns)
    if any(r*r != n for r, n in zip(roots[:3], ns[:3])):
        raise ArithmeticError("three-distance identity failed")
    inside = 0 < X < Q and 0 < Y < Q
    full = roots[3]**2 == ns[3]
    nearest = min((roots[3], roots[3]+1), key=lambda r: abs(ns[3]-r*r))
    return dict(x=str(X), y=str(Y), q=str(Q),
                d=[str(r) for r in roots[:3]],
                fourth_squared=str(ns[3]), nearest=str(nearest),
                gap=str(abs(ns[3]-nearest*nearest)), q_bits=Q.bit_length(),
                inside=inside, four_distance=full)


def poly(s, c):
    value = 0
    for v in c:
        value = value*s+v
    return value


def derivative(s, c):
    return ((4*c[0]*s+3*c[1])*s+2*c[2])*s+c[3]


def curve(point, anchor):
    X, Y, Q = point
    ns = squared(X, Y, Q)
    A, B, C = (isqrt(n) for n in ns[:3])
    # In transformed coordinates the fixed-distance corner is (0,0),
    # the conic corner (1,0); remaining corner is (0,1) or (1,1).
    if anchor == 0:
        U, V, adjacent, other, diagonal = X, Y, B, C, False
    elif anchor == 1:
        U, V, adjacent, other, diagonal = Q-X, Y, A, C, True
    else:
        U, V, adjacent, other, diagonal = Q-Y, X, A, B, True
    if not U or not V or not adjacent:
        return None
    g = gcd(U, V)
    a, b = U//g, V//g
    H = a*a+b*b
    if isqrt(H)**2 != H:
        raise ArithmeticError("non-Pythagorean anchor slope")
    L, k = (a+b, 2) if diagonal else (b, 1)
    # f(s)=k D^2-4 L(s+a)D+4 H(s+a)^2, D=H-s^2.
    cf = (k, 4*L, -2*k*H+4*L*a+4*H,
          -4*L*H+8*H*a, k*H*H-4*L*a*H+4*H*a*a)
    states = []
    for sign in (1, -1):
        s = F(sign*adjacent-Q, g)
        D = H-s*s
        if not D:
            continue
        w = F(other, Q)*D
        if 2*(s+a)/D != F(g, Q) or w*w != poly(s, cf):
            raise ArithmeticError("quartic embedding failed")
        states.append((s, w))
    return a, b, H, cf, states


def tangent_monic(s, w, cf, lead):
    """At anchor A, the quartic leading coefficient is a square."""
    if not w:
        return None
    d = derivative(s, cf)/(2*w)
    beta = d-2*lead*s
    gamma = w+lead*s*s-d*s
    c3 = cf[1]-2*lead*beta
    if not c3:
        return None
    u = -(cf[2]-beta*beta-2*lead*gamma)/c3-2*s
    v = lead*u*u+beta*u+gamma
    return u, v


def tangent_through(s, w, r, z, cf):
    """Quadratic tangent at (s,w) and passing through (r,z).
    The difference f-q^2 has roots s,s,r; Vieta gives the fourth.
    """
    if s == r or not w:
        return None
    d = derivative(s, cf)/(2*w)
    alpha = (z-w-d*(r-s))/(r-s)**2
    beta = d-2*alpha*s
    gamma = w-alpha*s*s-beta*s
    c4 = cf[0]-alpha*alpha
    if not c4:
        return None
    u = -(cf[1]-2*alpha*beta)/c4-2*s-r
    v = (alpha*u+beta)*u+gamma
    return u, v


def decode(u, v, a, b, H, cf, anchor, bit_cap):
    if v*v != poly(u, cf):
        raise ArithmeticError("intersection failed exact quartic identity")
    D = H-u*u
    if not D:
        return None
    t = 2*(u+a)/D
    if not t:
        return None
    x, y = a*t, b*t
    if anchor == 1:
        x = 1-x
    elif anchor == 2:
        x, y = y, 1-x
    Q = lcm(x.denominator, y.denominator)
    if Q.bit_length() > bit_cap:
        return None
    return int(x*Q), int(y*Q), Q


def neighbors(point, anchors, bit_cap, stats):
    for anchor in anchors:
        data = curve(point, anchor)
        if data is None:
            continue
        a, b, H, cf, states = data
        if not states:
            continue
        if anchor == 0:
            trials = (tangent_monic(s, w, cf, lead)
                      for s, w in states for lead in (1, -1))
        elif len(states) == 2:
            (s, w), (r, z) = states
            trials = (tangent_through(p, q, u, sign*v, cf)
                      for p, q, u, v in ((s,w,r,z),(r,z,s,w))
                      for sign in (1, -1))
        else:
            continue
        for result in trials:
            stats["intersections"] += 1
            if result is None:
                stats["degenerate"] += 1
                continue
            u, v = result
            if size(u, v) > 4*bit_cap:
                stats["height_skipped"] += 1
                continue
            nxt = decode(u, v, a, b, H, cf, anchor, bit_cap)
            if nxt is None:
                stats["decode_skipped"] += 1
                continue
            yield nxt, anchor


def input_rows(seed_files, point_files):
    for path in seed_files:
        with open(path, encoding="utf-8") as f:
            yield from json.load(f)["seeds"]
    for path in point_files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def search(seed_files, point_files, output, records, depth, source_bits,
           bit_cap, max_inputs, max_points, anchors=(0,1,2)):
    stats = dict(input_rows=0, input_accepted=0, input_duplicate=0,
                 source_height_skipped=0, intersections=0, degenerate=0,
                 height_skipped=0, decode_skipped=0, duplicate=0,
                 three_distance_points=0, inside=0, outside=0,
                 found_four=0, exterior_four=0, max_q_bits=0,
                 generated_by_anchor={"A":0,"B":0,"C":0},
                 stopped_at_input_cap=False, stopped_at_point_cap=False)
    seen = set()
    solutions = []
    out = Path(output)
    rec = Path(records)
    if out.resolve() == rec.resolve():
        raise ValueError("summary and records must be different paths")
    inputs = {Path(p).resolve() for p in seed_files+point_files}
    if out.resolve() in inputs or rec.resolve() in inputs:
        raise ValueError("output would overwrite an input")
    for p in (out, rec):
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    part = rec.with_name(rec.name+".partial")
    summary_part = out.with_name(out.name+".partial")
    with gzip.open(part, "wt", encoding="utf-8") as dest:
        def save(point, record, anchor, level):
            record["anchor"] = "ABC"[anchor]
            record["depth"] = level
            stats["three_distance_points"] += 1
            stats["inside" if record["inside"] else "outside"] += 1
            stats["max_q_bits"] = max(stats["max_q_bits"], record["q_bits"])
            stats["generated_by_anchor"]["ABC"[anchor]] += 1
            dest.write(json.dumps(record, separators=(",", ":"))+"\n")
            if record["four_distance"]:
                if record["inside"]:
                    stats["found_four"] += 1
                    solutions.append(record)
                    dest.flush()
                    print("POTENTIAL FOUR-DISTANCE SOLUTION!", flush=True)
                    print(json.dumps(record), flush=True)
                else:
                    stats["exterior_four"] += 1
        for row in input_rows(seed_files, point_files):
            if stats["input_accepted"] >= max_inputs:
                stats["stopped_at_input_cap"] = True
                break
            if stats["three_distance_points"] >= max_points:
                stats["stopped_at_point_cap"] = True
                break
            stats["input_rows"] += 1
            if row.get("q_bits", 0) > source_bits:
                stats["source_height_skipped"] += 1
                continue
            p = tuple(int(row[k]) for k in ("x","y","q"))
            if max(abs(v).bit_length() for v in p) > source_bits:
                stats["source_height_skipped"] += 1
                continue
            if p in seen:
                stats["input_duplicate"] += 1
                continue
            initial = certify(*p)
            if "d" in row and list(map(int,row["d"])) != list(map(int,initial["d"])):
                raise ArithmeticError("source distances disagree")
            seen.add(p)
            stats["input_accepted"] += 1
            if initial["four_distance"] and initial["inside"]:
                stats["found_four"] += 1
                solutions.append(initial)
                print("POTENTIAL FOUR-DISTANCE SOLUTION!", flush=True)
                print(json.dumps(initial), flush=True)
            queue = deque([(p,0)])
            while queue:
                parent, level = queue.popleft()
                if level >= depth:
                    continue
                for nxt, anchor in neighbors(parent, anchors, bit_cap, stats):
                    if stats["three_distance_points"] >= max_points:
                        stats["stopped_at_point_cap"] = True
                        break
                    if nxt in seen:
                        stats["duplicate"] += 1
                        continue
                    record = certify(*nxt)
                    seen.add(nxt)
                    save(nxt,record,anchor,level+1)
                    if level+1 < depth:
                        queue.append((nxt,level+1))
                if stats["stopped_at_point_cap"]:
                    break
            if stats["input_accepted"] % 100 == 0:
                print(f"inputs={stats['input_accepted']} points={stats['three_distance_points']} "
                      f"inside={stats['inside']} FOUR={stats['found_four']}", flush=True)
    result = dict(method="exact-changing-anchor-quartics-v1",
                  scope="bounded per-input walk; exact reduced coordinates; not exhaustive",
                  parameters=dict(depth=depth, source_bits=source_bits,
                                  max_q_bits=bit_cap, max_inputs=max_inputs,
                                  max_points=max_points, anchors=["ABC"[a] for a in anchors]),
                  seed_files=list(seed_files), point_files=list(point_files),
                  stats=stats, solutions=solutions)
    summary_part.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    part.replace(rec)
    summary_part.replace(out)
    print(json.dumps(stats), flush=True)
    return result


def self_test():
    from tempfile import TemporaryDirectory
    p = (45,24,52)
    c = certify(*p)
    assert c["d"] == ["51","25","53"] and c["gap"] != "0"
    stats = dict(intersections=0, degenerate=0, height_skipped=0, decode_skipped=0)
    generated = list(neighbors(p, (0,1,2), 8192, stats))
    expected = (23824816483171330179,44335350636984317928,36755960418958422908)
    assert (expected,1) in generated, generated
    for q, _ in generated:
        assert not certify(*q)["four_distance"]
    with TemporaryDirectory() as folder:
        f = Path(folder)
        seed = f/"seeds.json"
        seed.write_text(json.dumps({"seeds":[c]}))
        points = f/"input.jsonl.gz"
        with gzip.open(points,"wt") as dest:
            dest.write(json.dumps(c)+"\n")
        out = search([str(seed)],[str(points)],f/"summary.json",f/"points.gz",
                     2,256,8192,10,1000)
        assert out["stats"]["input_accepted"] == 1
        assert out["stats"]["input_duplicate"] == 1
        assert out["stats"]["generated_by_anchor"]["B"] > 0
        assert out["stats"]["generated_by_anchor"]["C"] > 0
        assert out["stats"]["found_four"] == 0
        with gzip.open(f/"points.gz","rt") as src:
            rows = [json.loads(line) for line in src]
        assert len(rows) == out["stats"]["three_distance_points"]
        assert len({(r["x"],r["y"],r["q"]) for r in rows}) == len(rows)
        for r in rows:
            assert certify(*(int(r[k]) for k in ("x","y","q")))["d"] == r["d"]
    print("self-test passed: A/B/C quartics, exact intersections, gzip and dedupe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--seeds",nargs="*",default=[])
    ap.add_argument("--points",nargs="*",default=[])
    ap.add_argument("--output",default="local-reanchor/summary.json")
    ap.add_argument("--records",default="local-reanchor/points.jsonl.gz")
    ap.add_argument("--depth",type=int,default=2)
    ap.add_argument("--source-bits",type=int,default=256)
    ap.add_argument("--max-q-bits",type=int,default=8192)
    ap.add_argument("--max-inputs",type=int,default=10000)
    ap.add_argument("--max-points",type=int,default=200000)
    ap.add_argument("--anchors",default="ABC",choices=["ABC","BC","A","B","C"])
    args = ap.parse_args()
    if args.self_test:
        self_test()
        sys.exit(0)
    if not args.seeds and not args.points:
        ap.error("provide --seeds and/or --points")
    if not (1 <= args.depth <= 5 and 8 <= args.source_bits <= args.max_q_bits
            and 128 <= args.max_q_bits <= 8192 and 1 <= args.max_inputs <= 1000000
            and 1 <= args.max_points <= 1000000):
        ap.error("invalid resource limits")
    search(args.seeds,args.points,args.output,args.records,args.depth,
           args.source_bits,args.max_q_bits,args.max_inputs,args.max_points,
           tuple("ABC".index(a) for a in args.anchors))
