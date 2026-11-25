#!/usr/bin/env python3
"""Unbounded multiplier congruence analysis of a chosen elliptic subgroup.

For each good prime compute the full period of R in E(F_p), and the
allowed residues for f(nR) and AB/f(nR). Both signs are covered by
n ranging over all integers. All constraints are necessary, not
sufficient. The ordinary branch always admits the common zero residue.
An empty inverse branch is a rigorous exclusion of that branch for
ALL integer multiples of this generator, not of the entire curve.
"""
import argparse
from collections import Counter, defaultdict, deque
from fractions import Fraction as F
import gzip
import json
from math import gcd, isqrt, lcm
from pathlib import Path
import random

from elliptic_sieve import (add, coeff, direct_mask, good_primes, heights,
                            models, on_curve, primes_up_to, reduce_point,
                            seeds_and_points)


def period_constraints(R, A, B, p):
    c2,c4=coeff(A,B)
    c2,c4=c2%p,c4%p
    R=reduce_point(R,p)
    if not on_curve(R,c2,c4,p):
        raise ArithmeticError("invalid reduction")
    P=None
    values=[]
    bound=p+1+isqrt(4*p)
    for n in range(bound):
        h=heights(P,A,B,p)
        values.extend((h[0],h[2]))
        P=add(P,R,c2,c4,p)
        if P is None:
            m=n+1
            mask=direct_mask(values,A,B,p)
            domains=[{j//2 for j in mask if j%2==kind} for kind in (0,1)]
            if 0 not in domains[0]:
                raise ArithmeticError("ordinary branch lost zero residue")
            return m,domains
    raise ArithmeticError("period exceeds Hasse bound")


def shrink(m,S):
    """Replace a periodic set by its least period (same integers)."""
    S=set(S)
    if not S:
        return 1,set()
    if len(S)==m:
        return 1,{0}
    divisors=sorted({d for i in range(1,isqrt(m)+1) if m%i==0
                     for d in (i,m//i)})
    for d in divisors:
        if len(S)%(m//d):
            continue
        T={x%d for x in S}
        if len(T)*(m//d)==len(S):
            return d,T
    raise ArithmeticError("period reduction failed")


def normalize(constraints):
    """Intersect equal periods and discard tautologies."""
    by_mod={}
    for m,S in constraints:
        m,S=shrink(m,S)
        if not S:
            return [(1,set())]
        if m==1:
            continue
        if m in by_mod:
            by_mod[m]&=S
        else:
            by_mod[m]=S
        if not by_mod[m]:
            return [(1,set())]
    return sorted(by_mod.items())


def graph(moduli):
    links=[[] for _ in moduli]
    for i,m in enumerate(moduli):
        for j in range(i):
            g=gcd(m,moduli[j])
            if g>1:
                links[i].append((j,g))
                links[j].append((i,g))
    return links


def propagate(domains,links,queue=None):
    """Arc consistency under generalized CRT pair compatibility."""
    if any(not d for d in domains):
        return False
    q=deque((i,j,g) for i,L in enumerate(links) for j,g in L) if queue is None else deque(queue)
    while q:
        i,j,g=q.popleft()
        residues={b%g for b in domains[j]}
        old=domains[i]
        new={a for a in old if a%g in residues}
        if len(new)!=len(old):
            domains[i]=new
            if not new:
                return False
            q.extend((k,i,h) for k,h in links[i] if k!=j)
    return True


def crt_pair(a,m,b,n):
    g=gcd(m,n)
    if (b-a)%g:
        return None
    k=n//g
    t=((b-a)//g*pow(m//g,-1,k))%k if k>1 else 0
    M=m*k
    return (a+m*t)%M,M


def combine_choice(moduli,choice):
    a,m=0,1
    for n,b in zip(moduli,choice):
        result=crt_pair(a,m,b,n)
        if result is None:
            return None
        a,m=result
    return a,m


def solve_domains(moduli,domains,links,node_cap):
    """Complete if budget permits; return sat, unsat, or unknown."""
    nodes=0
    def visit(D):
        nonlocal nodes
        if nodes>=node_cap:
            return "unknown",None
        nodes+=1
        choice=[min(s) for s in D]
        trial=combine_choice(moduli,choice)
        if trial is not None:
            return "sat",trial
        # A conflict must involve a domain with >1 residue after propagation.
        bad=set()
        for i,L in enumerate(links):
            for j,g in L:
                if choice[i]%g!=choice[j]%g:
                    bad.update((i,j))
        options=[i for i in bad if len(D[i])>1]
        if not options:
            return "unsat",None
        i=min(options,key=lambda i:(len(D[i]),-len(links[i]),i))
        unknown=False
        # Values with the same projection onto all neighbor moduli are
        # interchangeable for satisfiability, so branch only once per class.
        relevant=lcm(*(g for _,g in links[i]))
        reps={}
        for a in sorted(D[i]):
            reps.setdefault(a%relevant,a)
        for a in reps.values():
            if nodes>=node_cap:
                return "unknown",None
            child=D[:]
            child[i]={a}
            if not propagate(child,links,[(j,i,g) for j,g in links[i]]):
                continue
            status,witness=visit(child)
            if status=="sat":
                return status,witness
            unknown|=status=="unknown"
        return ("unknown" if unknown else "unsat"),None
    status,witness=visit(domains)
    return status,witness,nodes


def enumerate_classes(moduli,domains,cap):
    """Materialize exact CRT intersection only within a class-count budget."""
    M=1
    S={0}
    pending=list(zip(moduli,domains))
    while pending:
        best=None
        for i,(n,T) in enumerate(pending):
            g=gcd(M,n)
            ca=Counter(a%g for a in S)
            cb=Counter(b%g for b in T)
            count=sum(v*cb.get(k,0) for k,v in ca.items())
            if count>cap:
                continue
            score=(count,F(count,lcm(M,n)),i)
            if best is None or score<best[0]:
                best=(score,i,g)
        if best is None:
            break
        _,i,g=best
        n,T=pending.pop(i)
        groups=defaultdict(list)
        for b in T:
            groups[b%g].append(b)
        M2=lcm(M,n)
        inv=pow(M//g,-1,n//g) if n//g>1 else 0
        S={ (a+M*(((b-a)//g*inv)%(n//g)))%M2
            for a in S for b in groups[a%g] }
        M=M2
        if not S:
            return M,S,True
    return M,S,not pending


def analyze_branch(constraints,node_cap,class_cap):
    normalized=normalize(constraints)
    moduli=[m for m,_ in normalized]
    D=[set(S) for _,S in normalized]
    global_period=lcm(*(m for m,_ in constraints))
    links=graph(moduli)
    if not propagate(D,links):
        return dict(status="excluded",method="arc_consistency",
                    period=str(global_period),nodes=0,
                    exact_classes=True,merged_period="1",
                    surviving_classes=0,residues=[],witness=None)
    status,witness,nodes=solve_domains(moduli,D,links,node_cap)
    if status=="unsat":
        return dict(status="excluded",method="crt_search",
                    period=str(global_period),nodes=nodes,
                    exact_classes=True,merged_period="1",
                    surviving_classes=0,residues=[],witness=None)
    M,S,complete=enumerate_classes(moduli,D,class_cap)
    if not S:
        status="unsat"
    elif complete and status=="unknown":
        status="sat"
        witness=(min(S),M)
    if status=="sat":
        if witness is None:
            raise ArithmeticError("missing CRT witness")
        a,wmod=witness
        if any(a%m not in T for m,T in constraints):
            raise ArithmeticError("invalid CRT witness")
        # The class may be zero; this means POSITIVE multiples of the
        # full period also survive. Never interpret it as only n=0.
        witness=str(a%global_period or global_period)
    return dict(status={"sat":"survives","unsat":"excluded",
                        "unknown":"budget_limited"}[status],
                method="crt_search",period=str(global_period),nodes=nodes,
                exact_classes=complete or not S,merged_period=str(M),
                surviving_classes=len(S),residues=sorted(S),witness=witness,
                density_upper=str(F(len(S),M)),
                min_abs_n_lower_bound=str(min((min(r,M-r) or M for r in S),default=0)),
                reduced_constraints=[[m,sorted(T)] for m,T in zip(moduli,D)])


def analyze(R,A,B,prime_limit,node_cap,class_cap):
    raw=[]
    for p in good_primes(A,B,primes_up_to(prime_limit)):
        m,D=period_constraints(R,A,B,p)
        raw.append([p,m,sorted(D[0]),sorted(D[1])])
    branches={}
    for k,name in enumerate(("ordinary","inverse")):
        constraints=[(m,set(row[2+k])) for row in raw for m in [row[1]]]
        branches[name]=analyze_branch(constraints,node_cap,class_cap)
    if branches["ordinary"]["status"]!="survives":
        raise ArithmeticError("ordinary common zero residue must survive")
    return dict(A=str(A),B=str(B),R=[str(z) for z in R],
                primes=raw,branches=branches)


def search(seeds,points,prior,output,records,prime_limit,source_bits,
           max_models,node_cap,class_cap,axes):
    out,rec=Path(output),Path(records)
    inputs={Path(p).resolve() for p in seeds+points+prior}
    if out.resolve()==rec.resolve() or {out.resolve(),rec.resolve()}&inputs:
        raise ValueError("output path conflicts")
    for path in (out,rec):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    part=rec.with_name(rec.name+".partial")
    tmp=out.with_name(out.name+".partial")
    stats=Counter({k:0 for k in ("input_rows","models","curves",
        "duplicate_generators","prime_tests","ordinary_survives",
        "inverse_survives","inverse_excluded","inverse_budget_limited",
        "inverse_excluded_arc_consistency","inverse_excluded_crt_search",
        "ordinary_exact_classes","inverse_exact_classes","stopped_at_model_cap",
        "max_period_bits")})
    seen=set()
    curves=set()
    def sources():
        for path in prior:
            with gzip.open(path,"rt",encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    A,B=int(row["A"]),int(row["B"])
                    R=tuple(F(v) for v in row["R"])
                    if not (0<A<B and gcd(A,B)==1 and len(R)==2
                            and on_curve(R,*coeff(A,B))):
                        raise ArithmeticError("invalid prior model")
                    yield [(A,B,R)]
        for row in seeds_and_points(seeds,points):
            yield models(row,source_bits,axes)
    with gzip.open(part,"wt",encoding="utf-8") as dest:
        stop=False
        for group in sources():
            stats["input_rows"]+=1
            for A,B,R in group:
                key=(A,B,R)
                if key in seen:
                    stats["duplicate_generators"]+=1
                    continue
                if stats["models"]>=max_models:
                    stats["stopped_at_model_cap"]=1
                    stop=True
                    break
                seen.add(key)
                curves.add((A,B))
                result=analyze(R,A,B,prime_limit,node_cap,class_cap)
                stats["models"]+=1
                stats["curves"]=len(curves)
                stats["prime_tests"]+=len(result["primes"])
                for name,branch in result["branches"].items():
                    stats[name+"_"+branch["status"]]+=1
                    stats[name+"_exact_classes"]+=int(branch["exact_classes"])
                    stats["max_period_bits"]=max(stats["max_period_bits"],
                                                 int(branch["period"]).bit_length())
                    if branch["status"]=="excluded":
                        stats[name+"_excluded_"+branch["method"]]+=1
                dest.write(json.dumps(result,separators=(",",":"))+"\n")
                if stats["models"]%25==0:
                    print(f"models={stats['models']} inverse_excluded={stats['inverse_excluded']} "
                          f"inverse_survives={stats['inverse_survives']} "
                          f"unknown={stats['inverse_budget_limited']}",flush=True)
            if stop:
                break
    result=dict(method="unbounded-multiplier-crt-v1",
                scope="all integer multiples of each selected generator; necessary congruences only",
                parameters=dict(prime_limit=prime_limit,source_bits=source_bits,
                    max_models=max_models,node_cap=node_cap,class_cap=class_cap,axes=axes),
                seeds=seeds,points=points,prior_models=prior,stats=dict(stats))
    tmp.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    part.replace(rec)
    tmp.replace(out)
    print(json.dumps(dict(stats)),flush=True)
    return result


def self_test():
    from tempfile import TemporaryDirectory
    from elliptic_sieve import half_point, mod_heights
    # Synthetic unsatisfiable 3-cycle: equality across (2,3), (2,5)
    # and inequality across (3,5). All pairwise projections agree.
    contradiction=[(6,{0,1}),(10,{0,1}),(15,{6,10})]
    assert analyze_branch(contradiction,100,100)["status"]=="excluded"
    # Random small CRT systems, compare against exhaustive integer residues.
    rng=random.Random(42)
    for _ in range(300):
        raw=[]
        for m in rng.sample(range(2,13),rng.randrange(1,5)):
            raw.append((m,{r for r in range(m) if rng.randrange(2)}))
        period=lcm(*(m for m,_ in raw))
        brute={n for n in range(period) if all(n%m in S for m,S in raw)}
        b=analyze_branch(raw,10000,100000)
        assert (b["status"]=="survives")==bool(brute),(raw,b)
        assert b["exact_classes"]
        if brute:
            M=int(b["merged_period"])
            assert {n for n in range(period) if n%M in b["residues"]}==brute
            assert int(b["witness"])%period in brute
            assert int(b["min_abs_n_lower_bound"])==min(min(n,period-n) or period for n in brute)
    A,B=7,45
    R=half_point(F(24),F(25),F(51),A,B)
    for p in good_primes(A,B,primes_up_to(101)):
        m,D=period_constraints(R,A,B,p)
        values=mod_heights(R,A,B,p,2*m+5)
        mask=direct_mask(values,A,B,p)
        for n in range(1,2*m+6):
            for k in (0,1):
                assert ((n%m in D[k])==(4*(n-1)+2*k in mask))
    r=analyze(R,A,B,251,10000,1024)
    assert r["branches"]["ordinary"]["status"]=="survives"
    print("pilot",json.dumps({k:{j:v for j,v in b.items() if j in
          ("status","method","period","nodes","exact_classes","merged_period",
           "surviving_classes","density_upper","min_abs_n_lower_bound","witness")}
          for k,b in r["branches"].items()}))
    with TemporaryDirectory() as root:
        f=Path(root)
        seed=f/"seed.json"
        seed.write_text(json.dumps({"seeds":[dict(x="45",y="24",q="52")]}))
        s=search([str(seed)],[],[],f/"summary.json",f/"records.gz",101,256,10,1000,128,"xy")
        assert s["stats"]["models"]==2 and s["stats"]["ordinary_survives"]==2
        with gzip.open(f/"records.gz","rt") as inp:
            assert len(list(inp))==2
        t=search([],[],[str(f/"records.gz")],f/"summary2.json",f/"records2.gz",101,256,10,1000,128,"xy")
        assert t["stats"]["input_rows"]==2 and s["stats"]["input_rows"]==1
    assert {k:v for k,v in t["stats"].items() if k!="input_rows"} == {
        k:v for k,v in s["stats"].items() if k!="input_rows"},(t,s)
    print("self-test passed: full periods, CRT vs exhaustive checks, prior model input")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--seeds",nargs="*",default=[])
    ap.add_argument("--points",nargs="*",default=[])
    ap.add_argument("--prior-models",nargs="*",default=[])
    ap.add_argument("--prime-limit",type=int,default=251)
    ap.add_argument("--source-bits",type=int,default=256)
    ap.add_argument("--max-models",type=int,default=2000)
    ap.add_argument("--node-cap",type=int,default=10000)
    ap.add_argument("--class-cap",type=int,default=1024)
    ap.add_argument("--axes",choices=("x","y","xy"),default="xy")
    ap.add_argument("--output",default="local-periodic/summary.json")
    ap.add_argument("--records",default="local-periodic/periods.jsonl.gz")
    a=ap.parse_args()
    if a.self_test:
        self_test()
        raise SystemExit(0)
    if not (a.seeds or a.points or a.prior_models):
        ap.error("provide --seeds, --points or --prior-models")
    if not (3<=a.prime_limit<=5000 and 8<=a.source_bits<=8192 and
            1<=a.max_models<=100000 and 1<=a.node_cap<=1000000 and
            1<=a.class_cap<=100000):
        ap.error("invalid resource limits")
    search(a.seeds,a.points,a.prior_models,a.output,a.records,a.prime_limit,
           a.source_bits,a.max_models,a.node_cap,a.class_cap,a.axes)
