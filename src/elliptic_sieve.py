#!/usr/bin/env python3
"""Exact elliptic-curve / finite-field sieve for the rational square problem.

Fix a=A/Q, b=B/Q with A+B=Q, gcd(A,B)=1.
On E: v^2=u(u+A^2)(u+B^2), the rational function
f(R)=(u^2-A^2 B^2)/(2v) gives the signed integer-scale height of 2R.
Explore labels +/-f(nR), +/-AB/f(nR). Sieve their complementary
heights modulo good primes before computing any large rational multiples.
The pair search is bounded to the chosen labels; the direct sieve also
checks each label without requiring its complement to be in that list.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
import json
from math import gcd, isqrt, lcm
from pathlib import Path
import sys

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def coeff(A, B):
    return A*A+B*B, (A*B)**2


def on_curve(P, c2, c4, p=None):
    if P is None:
        return True
    x,y=P
    z=y*y-x*(x*x+c2*x+c4)
    return z == 0 if p is None else z % p == 0


def add(P, R, c2, c4, p=None):
    if P is None:
        return R
    if R is None:
        return P
    x,y=P
    u,v=R
    if x == u:
        if (y+v == 0 if p is None else (y+v)%p == 0):
            return None
        if y != v:
            raise ArithmeticError("inconsistent points")
        num,den=3*x*x+2*c2*x+c4,2*y
    else:
        num,den=v-y,u-x
    m=F(num,den) if p is None else (num*pow(den,-1,p))%p
    w=m*m-c2-x-u
    z=m*(x-w)-y
    return (w,z) if p is None else (w%p,z%p)


def bits(P):
    return 0 if P is None else max(
        max(z.numerator.bit_length(),z.denominator.bit_length()) for z in P)


class WorkLimit(Exception):
    pass


def multiply(n, R, c2, c4, cap):
    P=None
    while n:
        if n & 1:
            P=add(P,R,c2,c4)
            if bits(P)>cap:
                raise WorkLimit
        n//=2
        if n:
            R=add(R,R,c2,c4)
            if bits(R)>cap:
                raise WorkLimit
    return P


def half_point(y, r, s, A, B):
    """Construct a half of P=(y^2, y*r*s) from the three square roots."""
    c2,c4=coeff(A,B)
    target=(y*y,y*r*s)
    choices=[]
    for i,j,k in ((-1,1,1),(1,-1,1),(1,1,-1),(-1,-1,-1)):
        h1,h2,h3=i*y,j*r,k*s
        d=h1*h2+h2*h3+h3*h1
        R=(target[0]+d,-target[1]-(h1+h2+h3)*d)
        if not on_curve(R,c2,c4) or add(R,R,c2,c4)!=target:
            raise ArithmeticError("halving identity failed")
        # Both signs of f will be enumerated, so normalize the sign of v.
        choices.append((R[0],abs(R[1])))
    return min(choices,key=lambda R:(bits(R),R))


def reduce_point(R,p):
    if R is None:
        return None
    if any(v.denominator%p==0 for v in R):
        # At good reduction a rational point with nonintegral x reduces to O.
        if R[0].denominator%p!=0:
            raise ArithmeticError("unexpected nonintegral affine point")
        return None
    return tuple((v.numerator*pow(v.denominator,-1,p))%p for v in R)


def heights(P,A,B,p=None):
    """Four projective values: f,-f,AB/f,-AB/f. None is infinity."""
    if P is None:
        return (None,None,0,0) if p is not None else (None,None,F(0),F(0))
    u,v=P
    ab=A*B
    num,den=u*u-ab*ab,2*v
    if p is not None:
        num,den,ab=num%p,den%p,ab%p
    if not den and not num:
        raise ArithmeticError("indeterminate height at good reduction")
    if p is None:
        y=F(num,den) if den else None
        z=F(ab*den,num) if num else None
        return y, -y if y is not None else None, z, -z if z is not None else None
    y=num*pow(den,-1,p)%p if den else None
    z=ab*den*pow(num,-1,p)%p if num else None
    return y, (-y)%p if y is not None else None, z, (-z)%p if z is not None else None


def primes_up_to(n):
    return [p for p in range(3,n+1,2)
            if all(p%d for d in range(3,isqrt(p)+1,2))]


def good_primes(A,B,primes):
    return [p for p in primes if A%p and B%p and (A-B)%p and (A+B)%p]


def mod_heights(R,A,B,p,N):
    c2,c4=coeff(A,B)
    c2,c4=c2%p,c4%p
    R=reduce_point(R,p)
    if not on_curve(R,c2,c4,p):
        raise ArithmeticError("bad reduced generator")
    # The subgroup often repeats after only a few dozen multiples.
    # Cache its entire cycle when it closes before N.
    P=None
    values=[]
    for n in range(1,N+1):
        P=add(P,R,c2,c4,p)
        values.extend(heights(P,A,B,p))
        if P is None:
            cycle=values[:]
            return (cycle*((N+n-1)//n))[:4*N]
    return values


def complement(y,Q,p):
    return None if y is None else (Q-y)%p


def direct_mask(values,A,B,p):
    Q=A+B
    squares={i*i%p for i in range(p)}
    good={t for t in range(p)
          if ((Q-t)**2+A*A)%p in squares
          and ((Q-t)**2+B*B)%p in squares}
    # A pole is allowed: rational heights with denominator divisible
    # by p cannot be rejected just because an affine residue is absent.
    return {i for i,y in enumerate(values) if y is None or y in good}


def sieve(R,A,B,N,primes,mode):
    Q=A+B
    active=set(range(4*N))
    tables=[]
    counts=[]
    for p in good_primes(A,B,primes):
        values=mod_heights(R,A,B,p,N)
        active.intersection_update(direct_mask(values,A,B,p))
        counts.append([p,len(active)])
        if mode in ("pairs","both"):
            tables.append((p,values))
        if not active:
            break
    direct=sorted(active)
    # Bit masks encode compatible partners, avoiding quadratic enumeration
    # of labels. The direct filter is necessary for either member of a pair.
    pair_count=0
    paired=set()
    if mode in ("pairs","both") and direct:
        masks=[]
        for p,values in tables:
            groups=defaultdict(int)
            for j,index in enumerate(direct):
                groups[values[index]] |= 1<<j
            masks.append((p,values,groups))
        partner_union=0
        for j,index in enumerate(direct):
            possible=((1<<len(direct))-1)^((1<<j)-1)
            for p,values,groups in masks:
                # The homogeneous equality also permits two poles.
                possible &= groups.get(complement(values[index],Q,p),0)
                if not possible:
                    break
            if possible:
                pair_count+=possible.bit_count()
                paired.add(index)
                partner_union |= possible
        while partner_union:
            low=partner_union&-partner_union
            paired.add(direct[low.bit_length()-1])
            partner_union^=low
    chosen=sorted(paired) if mode=="pairs" else direct
    return direct,paired,chosen,pair_count,counts


def exact_check(A,B,h):
    if h is None or not 0<h<A+B:
        return None
    Q=A+B
    x=F(A,Q)
    y=h/Q
    q=lcm(x.denominator,y.denominator)
    X,Y=int(x*q),int(y*q)
    ns=(X*X+Y*Y,(q-X)**2+Y*Y,X*X+(q-Y)**2,(q-X)**2+(q-Y)**2)
    ds=tuple(isqrt(n) for n in ns)
    if any(r*r!=n for r,n in zip(ds[:2],ns[:2])):
        raise ArithmeticError("elliptic height failed distance identity")
    if any(r*r!=n for r,n in zip(ds[2:],ns[2:])):
        return False
    return dict(x=str(X),y=str(Y),q=str(q),d=[str(d) for d in ds])


def seeds_and_points(seeds,points):
    for path in seeds:
        with open(path,encoding="utf-8") as f:
            yield from json.load(f)["seeds"]
    for path in points:
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def models(row,source_bits,axes):
    X,Y,Q=map(int,(row["x"],row["y"],row["q"]))
    if max(abs(z).bit_length() for z in (X,Y,Q))>source_bits:
        return
    if Q<=0 or gcd(gcd(X,Y),Q)!=1:
        raise ArithmeticError("invalid source coordinates")
    ns=(X*X+Y*Y,(Q-X)**2+Y*Y,X*X+(Q-Y)**2)
    ds=tuple(isqrt(n) for n in ns)
    if any(r*r!=n for r,n in zip(ds,ns)):
        raise ArithmeticError("invalid source three-distance point")
    for axis in axes:
        x,y,r,s=(X,Y,ds[0],ds[1]) if axis=="x" else (Y,X,ds[0],ds[2])
        if not 0<x<Q or 2*x==Q or not y:
            continue
        g=gcd(x,Q)
        A,B=x//g,(Q-x)//g
        if A>B:
            A,B=B,A
            r,s=s,r
        R=half_point(F(abs(y),g),F(r,g),F(s,g),A,B)
        yield A,B,R


def search(seeds,points,output,records,N,prime_limit,source_bits,
           max_curves,max_exact_bits,mode,axes,max_exact_labels):
    out,rec=Path(output),Path(records)
    inputs={Path(p).resolve() for p in seeds+points}
    if out.resolve()==rec.resolve() or {out.resolve(),rec.resolve()}&inputs:
        raise ValueError("output paths conflict with inputs or each other")
    for path in (out,rec):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    part=rec.with_name(rec.name+".partial")
    op=out.with_name(out.name+".partial")
    stats=Counter()
    for key in ("input_rows","models","duplicate_generators","curves",
                "labels","pairs_before","pairs_after","direct_survivors",
                "paired_labels","exact_requested","exact_tested","outside_or_pole",
                "work_limit_deferred","label_limit_deferred","found_four",
                "stopped_at_curve_cap","good_prime_tests"):
        stats[key]=0
    seen=set()
    curves=set()
    solutions=[]
    primes=primes_up_to(prime_limit)
    with gzip.open(part,"wt",encoding="utf-8") as dest:
        stop=False
        for row in seeds_and_points(seeds,points):
            stats["input_rows"]+=1
            for A,B,R in models(row,source_bits,axes):
                key=(A,B,R)
                if key in seen:
                    stats["duplicate_generators"]+=1
                    continue
                if stats["models"]>=max_curves:
                    stats["stopped_at_curve_cap"]=1
                    stop=True
                    break
                seen.add(key)
                curves.add((A,B))
                stats["models"]+=1
                stats["curves"]=len(curves)
                L=4*N
                stats["labels"]+=L
                if mode!="direct":
                    stats["pairs_before"]+=L*(L+1)//2
                direct,paired,chosen,pair_count,counts=sieve(R,A,B,N,primes,mode)
                stats["direct_survivors"]+=len(direct)
                stats["paired_labels"]+=len(paired)
                stats["pairs_after"]+=pair_count
                stats["good_prime_tests"]+=len(counts)
                stats["exact_requested"]+=len(chosen)
                c2,c4=coeff(A,B)
                cache={}
                deferred=[]
                for i,label in enumerate(chosen):
                    n,kind=label//4+1,label%4
                    if i>=max_exact_labels:
                        stats["label_limit_deferred"]+=1
                        deferred.append([n,kind,"label_cap"])
                        continue
                    if n not in cache:
                        try:
                            cache[n]=multiply(n,R,c2,c4,max_exact_bits)
                        except WorkLimit:
                            cache[n]="deferred"
                    if cache[n]=="deferred":
                        stats["work_limit_deferred"]+=1
                        deferred.append([n,kind,"work_bits"])
                        continue
                    y=heights(cache[n],A,B)[kind]
                    if y is not None and bits((y,))>max_exact_bits:
                        stats["work_limit_deferred"]+=1
                        deferred.append([n,kind,"height_bits"])
                        continue
                    result=exact_check(A,B,y)
                    if result is None:
                        stats["outside_or_pole"]+=1
                    else:
                        stats["exact_tested"]+=1
                    if result:
                        if result not in solutions:
                            stats["found_four"]+=1
                            solutions.append(result)
                            print("POTENTIAL FOUR-DISTANCE SOLUTION!",flush=True)
                            print(json.dumps(result),flush=True)
                dest.write(json.dumps(dict(A=str(A),B=str(B),R=[str(z) for z in R],
                    direct_survivors=len(direct),pair_survivors=pair_count,
                    paired_labels=len(paired),prime_counts=counts,
                    deferred=deferred),separators=(",",":"))+"\n")
                if stats["models"]%25==0:
                    print(f"models={stats['models']} labels={stats['labels']} "
                          f"direct={stats['direct_survivors']} pairs={stats['pairs_after']} "
                          f"exact={stats['exact_tested']} FOUR={stats['found_four']}",flush=True)
            if stop:
                break
    summary=dict(method="elliptic-good-reduction-sieve-v1",
        scope="bounded cyclic subgroups and inversion/sign labels; not a rank computation",
        parameters=dict(max_multiple=N,prime_limit=prime_limit,source_bits=source_bits,
            max_curves=max_curves,max_exact_bits=max_exact_bits,mode=mode,
            axes=axes,max_exact_labels=max_exact_labels),
        seeds=seeds,points=points,stats=dict(stats),solutions=solutions)
    op.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    part.replace(rec)
    op.replace(out)
    print(json.dumps(dict(stats)),flush=True)
    return summary


def self_test():
    from tempfile import TemporaryDirectory
    A,B=7,45
    c2,c4=coeff(A,B)
    R=half_point(F(24),F(25),F(51),A,B)
    assert on_curve(R,c2,c4)
    P=add(R,R,c2,c4)
    assert P[0]==576 and abs(P[1])==30600
    assert F(24) in heights(R,A,B)
    assert F(105,8) in heights(R,A,B)
    # Compare exact reductions to finite-field group arithmetic, including
    # poles (denominator divisible by p) and the inversion branch.
    primes=good_primes(A,B,primes_up_to(83))
    N=8
    rational=[]
    for n in range(1,N+1):
        vals=heights(multiply(n,R,c2,c4,100000),A,B)
        rational.extend(vals)
        for y in vals:
            if y is not None:
                for leg in (A,B):
                    z=y*y+leg*leg
                    assert isqrt(z.numerator)**2==z.numerator
                    assert isqrt(z.denominator)**2==z.denominator
    for p in primes:
        expected=[None if y is None or y.denominator%p==0 else
                  y.numerator*pow(y.denominator,-1,p)%p for y in rational]
        assert mod_heights(R,A,B,p,N)==expected,(p,expected,mod_heights(R,A,B,p,N))
    # Independent brute-force check, including surviving pairs and poles.
    for subset in ([],primes[:1],primes):
        direct,paired,chosen,num,counts=sieve(R,A,B,N,subset,"both")
        tables=[(p,mod_heights(R,A,B,p,N)) for p in subset]
        brute=[(i,j) for i in range(4*N) for j in range(i,4*N)
               if all(t[j]==complement(t[i],A+B,p) for p,t in tables)]
        assert num==len(brute)
        assert paired=={i for pair in brute for i in pair}
        assert set(chosen)==set(direct)
    assert exact_check(A,B,F(24)) is False
    assert exact_check(A,B,None) is None
    assert exact_check(A,B,F(-24)) is None
    # Seed and gzip input integration, exact survivor check, output format.
    with TemporaryDirectory() as root:
        f=Path(root)
        seed=f/"seeds.json"
        row=dict(x="45",y="24",q="52",d=["51","25","53"])
        seed.write_text(json.dumps(dict(seeds=[row])))
        pts=f/"points.gz"
        with gzip.open(pts,"wt") as out:
            out.write(json.dumps(row)+"\n")
        s=search([str(seed)],[str(pts)],f/"summary.json",f/"curves.gz",
                 30,83,256,10,100000,"both","xy",100)
        assert s["stats"]["models"]==2
        assert s["stats"]["duplicate_generators"]==2
        assert s["stats"]["found_four"]==0
        assert s["stats"]["labels"]==240
        assert json.loads((f/"summary.json").read_text())==s
        with gzip.open(f/"curves.gz","rt") as inp:
            assert len(list(inp))==2
    print("self-test passed: halving, exact heights, projective reduction, pair sieve, input/output")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--seeds",nargs="*",default=[])
    ap.add_argument("--points",nargs="*",default=[])
    ap.add_argument("--max-multiple",type=int,default=500)
    ap.add_argument("--prime-limit",type=int,default=251)
    ap.add_argument("--source-bits",type=int,default=256)
    ap.add_argument("--max-curves",type=int,default=2000,
                    help="maximum distinct (curve, generator) models")
    ap.add_argument("--max-exact-bits",type=int,default=32768)
    ap.add_argument("--max-exact-labels",type=int,default=100)
    ap.add_argument("--mode",choices=("direct","pairs","both"),default="both")
    ap.add_argument("--axes",choices=("x","y","xy"),default="xy")
    ap.add_argument("--output",default="local-elliptic/summary.json")
    ap.add_argument("--records",default="local-elliptic/curves.jsonl.gz")
    args=ap.parse_args()
    if args.self_test:
        self_test()
        sys.exit(0)
    if not args.seeds and not args.points:
        ap.error("provide --seeds and/or --points")
    if not (1<=args.max_multiple<=100000 and 3<=args.prime_limit<=10000
            and 8<=args.source_bits<=8192 and 1<=args.max_curves<=100000
            and 128<=args.max_exact_bits<=1000000
            and 1<=args.max_exact_labels<=100000):
        ap.error("invalid resource limits")
    if args.mode != "direct" and 4*args.max_multiple*len(primes_up_to(args.prime_limit))>4000000:
        ap.error("pair table budget exceeded; reduce multiples or prime limit, or use direct mode")
    search(args.seeds,args.points,args.output,args.records,args.max_multiple,
        args.prime_limit,args.source_bits,args.max_curves,args.max_exact_bits,
        args.mode,args.axes,args.max_exact_labels)
