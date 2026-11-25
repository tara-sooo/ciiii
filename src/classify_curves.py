#!/usr/bin/env python3
"""Certified rank bounds and fixed-coordinate exclusion for the rational square.

Input: model records of elliptic_sieve.py or periodic_sieve.py.
All proofs use integer/Fraction arithmetic, standard 2-isogeny descent,
canonical height quadraticity, and good reduction of rational torsion.
A failed resource limit or incomplete descent/saturation is NOT exclusion.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
from functools import lru_cache
import gzip
import json
from math import gcd, isqrt
from pathlib import Path
from elliptic_sieve import (add, bits, coeff, exact_check, good_primes,
    half_point, multiply, on_curve, primes_up_to, WorkLimit)
from periodic_sieve import analyze as periodic_analyze


def sq(n):
    return n>=0 and isqrt(n)**2==n


def ceilf(x):
    return -(-x.numerator//x.denominator)


def prime64(n):
    """Deterministic Miller-Rabin for n < 2**64 only."""
    if n<2:
        return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n%p==0:
            return n==p
    d,s=n-1,0
    while d%2==0:
        d//=2
        s+=1
    for a in (2,325,9375,28178,450775,9780504,1795265022):
        if a%n==0:
            continue
        x=pow(a,d,n)
        if x in (1,n-1):
            continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1:
                break
        else:
            return False
    return True


@lru_cache(maxsize=8192)
def factor(n,limit):
    if n<=0:
        raise ValueError("positive factorization input required")
    out={}
    if n<2**64 and prime64(n):
        return {n:1}
    for p in trial_primes(limit):
        if p*p>n:
            break
        if n%p==0:
            e=0
            while n%p==0:
                n//=p
                e+=1
            out[p]=e
            if n==1 or (n<2**64 and prime64(n)):
                break
    if n>1:
        if n>=2**64 or not prime64(n):
            return None
        out[n]=out.get(n,0)+1
    return out


@lru_cache(maxsize=8)
def trial_primes(limit):
    sieve=bytearray(b"\1")*(limit+1)
    sieve[:2]=b"\0\0"
    for p in range(2,isqrt(limit)+1):
        if sieve[p]:
            sieve[p*p::p]=b"\0"*((limit-p*p)//p+1)
    return tuple(p for p in range(2,limit+1) if sieve[p])


def factor_inputs(A,B,limit):
    out={}
    for name,n in (("A",A),("B",B),("sum",A+B),("difference",B-A)):
        f=factor(n,limit)
        if f is None:
            return None
        prod=1
        for p,e in f.items():
            prod*=p**e
        if prod!=n:
            raise ArithmeticError("factorization mismatch")
        out[name]=f
    return out


def classes(primes,cap):
    if 2**(len(primes)+1)>cap:
        return None
    D=[1]
    for p in sorted(primes):
        D += [p*d for d in D]
    return sorted(D+[-d for d in D])


@lru_cache(maxsize=256)
def local_data(mod):
    p=next(p for p in range(2,mod+1) if mod%p==0)
    return ({i*i%mod for i in range(mod)},
            tuple((i*i%mod,pow(i,4,mod)) for i in range(mod)),
            tuple((i*i%mod,pow(i,4,mod)) for i in range(0,mod,p)))


def locally_possible(d,a,b,mod):
    # A primitive solution of w^2=d*i^4+a*i^2*j^2+(b/d)*j^4
    # can normalize either i=1 or j=1 with i divisible by p.
    S,all_j,nonunit_i=local_data(mod)
    d0,a0,e0=d%mod,a%mod,(b//d)%mod
    return (any((d0+a0*j2+e0*j4)%mod in S for j2,j4 in all_j)
            or any((d0*i4+a0*i2+e0)%mod in S for i2,i4 in nonunit_i))


def descent_side(a,b,primes,mods,cap,positive=False):
    D=classes(primes,cap)
    if D is None:
        return None
    rejected=[]
    keep=[]
    for d in D:
        if positive and d<0:
            # All real points on the dual curve have nonnegative x.
            rejected.append([d,"real"])
            continue
        for mod in mods:
            if not locally_possible(d,a,b,mod):
                rejected.append([d,mod])
                break
        else:
            keep.append(d)
    if 1 not in keep:
        raise ArithmeticError("identity lost in descent")
    return dict(initial=len(D),survivors=keep,rejected=rejected,
                dimension_upper=len(keep).bit_length()-1)



def binary_kernel(rows,n):
    """Homogeneous GF(2) kernel, represented by integer bit masks."""
    pivots={}
    for row in rows:
        while row:
            k=row.bit_length()-1
            if k in pivots:
                row^=pivots[k]
            else:
                pivots[k]=row
                break
    basis=[]
    for j in range(n):
        if j in pivots:
            continue
        v=1<<j
        for k,row in sorted(pivots.items()):
            if (row&v).bit_count()%2:
                v^=1<<k
        basis.append(v)
    return basis


def class_mask(d,gens):
    return sum(1<<i for i,g in enumerate(gens)
               if (d<0 if g==-1 else d%g==0))


def mask_class(mask,gens):
    d=1
    for i,g in enumerate(gens):
        if mask>>i&1:
            d*=g
    return d


def legendre_row(gens,p):
    if any(g%p==0 for g in gens):
        raise ArithmeticError("Legendre symbol requested for nonunit")
    return sum(1<<i for i,g in enumerate(gens)
               if pow(g%p,(p-1)//2,p)==p-1)


def local_two_code(d):
    """Coordinates of Q_2*/Q_2*^2: v2 parity, -1, 5."""
    e=0
    while d%2==0:
        d//=2
        e^=1
    return e | ({1:0,7:2,5:4,3:6}[d%8])


def two_adic_rows(a,b,gens,mod=256):
    # A unit square rescales i and w without changing primitive local
    # solubility. Test representatives of all possible Q_2 squareclasses
    # modulo mod, using the odd part inverse in b/d. A class that fails
    # this necessary test cannot be a rational descent class.
    S,all_j,nonunit_i=local_data(mod)
    allowed=[]
    for d in (1,3,5,7,2,6,10,14):
        e=d%2==0
        if e and b%2:
            continue
        d0=d%mod
        e0=(b//(2 if e else 1))*pow(d//(2 if e else 1),-1,mod)%mod
        a0=a%mod
        if (any((d0+a0*j2+e0*j4)%mod in S for j2,j4 in all_j)
            or any((d0*i4+a0*i2+e0)%mod in S for i2,i4 in nonunit_i)):
            allowed.append(local_two_code(d))
    # Every vector orthogonal to all allowed codes vanishes on the true
    # image. Using their span is safe even if the finite test is weaker.
    codes=[local_two_code(g) for g in gens]
    rows=[]
    for q in range(1,8):
        if all((q&v).bit_count()%2==0 for v in allowed):
            row=sum(1<<i for i,v in enumerate(codes) if (q&v).bit_count()%2)
            if row:
                rows.append(["two_adic",q,row])
    return rows,allowed


def linear_descent_side(a,b,primes,other_primes,mods,cap,dual=False):
    gens=[-1]+sorted(primes)
    rows=[]
    if dual:
        rows.append(["real",0,1])
    # At a bad odd prime outside the support of d, the quartic is
    # d*(i^2 + a/(2*d)*j^2)^2 modulo p. If -a/2 is a square,
    # a nonsquare d allows neither a nonzero square nor a primitive zero.
    for p in sorted(other_primes-primes):
        if p==2:
            continue
        if (a*a-4*b)%p or a%p==0:
            raise ArithmeticError("unexpected singular prime")
        if pow((-a*pow(2,-1,p))%p,(p-1)//2,p)==1:
            row=legendre_row(gens,p)
            if row:
                rows.append(["legendre",p,row])
    # When p|b, v_p(b) is positive and even. If a is a nonsquare
    # unit and p|d, the quartic terms with d and b/d have odd
    # valuations; the mixed term, when uniquely minimal, has a
    # nonsquare unit. For primitive i,j the minimum is unique
    # (or the mixed term is strictly minimal if i,j both units).
    # Hence no square can occur, and the p exponent of d must vanish.
    for i,p in enumerate(gens):
        if p<=2:
            continue
        if b%p or a%p==0:
            raise ArithmeticError("unexpected squareclass support")
        if pow(a%p,(p-1)//2,p)==p-1:
            rows.append(["valuation",p,1<<i])
    two,allowed_two=two_adic_rows(a,b,gens)
    rows+=two
    kernel=binary_kernel([r[2] for r in rows],len(gens))
    dimension=len(kernel)
    result=dict(initial=1<<len(gens),linear_candidates=1<<dimension,
                linear_dimension=dimension,linear_constraints=rows,
                kernel_basis=kernel,basis_generators=gens,
                two_adic_allowed_codes=allowed_two,
                dimension_upper=dimension,survivors=None,rejected=[],
                enumeration_complete=False)
    if (1<<dimension)>cap:
        # The kernel dimension is itself a certified upper bound, even
        # when the subsequent finite enumeration is too large.
        return result
    masks=[0]
    for v in kernel:
        masks += [m^v for m in masks]
    keep=[]
    rejected=[]
    for d in sorted(mask_class(m,gens) for m in masks):
        for mod in mods:
            if not locally_possible(d,a,b,mod):
                rejected.append([d,mod])
                break
        else:
            keep.append(d)
    if 1 not in keep:
        raise ArithmeticError("identity lost in linear descent")
    result.update(survivors=keep,rejected=rejected,enumeration_complete=True,
                  dimension_upper=len(keep).bit_length()-1)
    return result


def check_linear_realized(d,side):
    mask=class_mask(d,side["basis_generators"])
    if mask_class(mask,side["basis_generators"])!=d:
        raise ArithmeticError("realized class outside basis")
    if any((row&mask).bit_count()%2 for _,_,row in side["linear_constraints"]):
        raise ArithmeticError("realized class excluded by linear constraints")
    if side["survivors"] is not None and d not in side["survivors"]:
        raise ArithmeticError("realized class excluded by finite local test")


def squareclass(x,primes):
    if not x:
        return 1
    n=abs(x.numerator)*x.denominator
    d=-1 if x<0 else 1
    for p in sorted(primes):
        e=0
        while n%p==0:
            n//=p
            e^=1
        if e:
            d*=p
    if not sq(n):
        raise ArithmeticError("squareclass outside descent support")
    return d


def class_span(ds,primes):
    basis={}
    ps=sorted(primes)
    for d in ds:
        mask=int(d<0)
        for i,p in enumerate(ps,1):
            if d%p==0:
                mask|=1<<i
        while mask:
            k=mask.bit_length()
            if k not in basis:
                basis[k]=mask
                break
            mask^=basis[k]
    return len(basis)


def height_constants(A,B,fac):
    # x=s/t is a positive square in lowest terms, so s,t are coprime
    # squares. On doubling, F=(s^2-(AB)^2*t^2)^2,
    # G=4*s*t*(s+A^2*t)*(s+B^2*t).
    # gcd(F,G) has support only in 2*AB*(B^2-A^2).
    ab=Counter(fac["A"])+Counter(fac["B"])
    diff=Counter(fac["sum"])+Counter(fac["difference"])
    e=ab.get(2,0)
    D=2**(4*e+2 if e else 4)
    for p,e in ab.items():
        if p!=2:
            D*=p**(4*e)
    for p,e in diff.items():
        if p%4==1:
            D*=p**(2*e)
    # For s>=2AB*t, F>=s^4/2. For t<=s<2AB*t,
    # G>=4*s^3*t>s^4/(2AB). For s<t, F>=t^4.
    # Thus max(F,G)>=max(s,t)^4/(2AB).
    L=(2*A*B*D).bit_length()
    C=max((1+(A*B)**2)**2,4*(1+A*A)*(1+B*B)).bit_length()
    return L,C,D


def height_interval(P,A,B,L,C,steps,bit_cap):
    # hhat(P)=lim log2 H(x(2^j P))/(2*4^j).
    # h-L/3 <= 2*hhat <= h+C/3; use integer log bounds.
    c2,c4=coeff(A,B)
    lens=[]
    Q=P
    for j in range(steps+1):
        if Q is None or Q[0]==0:
            return F(0),F(0),lens
        x=Q[0]
        if x<0 or not sq(x.numerator) or not sq(x.denominator):
            raise ArithmeticError("height point not in square-coordinate locus")
        h=max(x.numerator,x.denominator).bit_length()
        lens.append(h)
        if j==steps or bits(Q)>bit_cap//4:
            break
        Q=add(Q,Q,c2,c4)
    k=len(lens)-1
    scale=2*4**k
    lo=max(F(0),(F(lens[-1]-1)-F(L,3))/scale)
    hi=(F(lens[-1])+F(C,3))/scale
    return lo,hi,lens


def point_order_count(A,B,p):
    S=Counter(i*i%p for i in range(p))
    return 1+sum(S[x*(x+A*A)*(x+B*B)%p] for x in range(p))


def torsion_certificate(A,B,primes):
    # Good reduction at odd primes is injective on rational torsion
    # (Q_p unramified, p>2). Known order 4 point + independent 2-torsion.
    c2,c4=coeff(A,B)
    T=(F(0),F(0))
    K=(F(A*B),F(A*B*(A+B)))
    assert on_curve(K,c2,c4) and add(K,K,c2,c4)==T
    counts=[]
    bound=0
    for p in good_primes(A,B,primes):
        if p<5:
            continue
        n=point_order_count(A,B,p)
        counts.append([p,n])
        bound=gcd(bound,n)
        if bound==8 and len(counts)>=2:
            break
    if bound and bound%8:
        raise ArithmeticError("torsion count inconsistent")
    return dict(upper=bound or None,exact_order=8 if bound==8 and len(counts)>=2 else None,
                reductions=counts)


def encode_point(P):
    return None if P is None else [str(z) for z in P]


def finite_saturation(P,A,B,L,C,hi,lo,steps,bit_cap,cap):
    # If R is k times a free generator modulo torsion, |k|>=2, then
    # Q=2G has hhat(Q)<=hhat(P)/4.  Its positive square x-coordinate
    # implies log2 H(y) <= (2*hhat(P)/4 + L/3)/2.
    # P=2R. Torsion of 2E must have already been certified as {O,T}.
    exp=ceilf(hi/4+F(L,6))
    result=dict(height_exponent=exp,bound=None,examined=0,heights=[],status="cap")
    if exp>cap.bit_length():
        return result
    N=1<<max(0,exp)
    result["bound"]=N
    if N>cap:
        return result
    c2,c4=coeff(A,B)
    T=(F(0),F(0))
    PT=add(P,T,c2,c4)
    known={P[0],PT[0]}
    unresolved=False
    for j in range(1,N+1):
        aj=A*A*j*j
        bj=B*B*j*j
        for i in range(1,N+1):
            result["examined"]+=1
            if gcd(i,j)!=1:
                continue
            z=i*i
            if not sq(z+aj) or not sq(z+bj):
                continue
            y=F(i,j)
            x=y*y
            r=F(isqrt(z+aj),j)
            s=F(isqrt(z+bj),j)
            Q=(x,y*r*s)
            if x in known:
                qlo=lo
                why="known_torsion_translate"
            else:
                qlo,_,_=height_interval(Q,A,B,L,C,steps,bit_cap)
                why="height_bound"
            result["heights"].append(dict(y=str(y),lower=str(qlo),reason=why))
            if qlo<=hi/4:
                unresolved=True
    result["status"]="unresolved_small_point" if unresolved else "saturated"
    return result


def subgroup_gap(A,B,P,lo,L,C,periodic,exact_n_cap,bit_cap):
    # Both complementary heights must lie on the ordinary branch if the
    # inverse branch is excluded. Their reduced denominators coincide.
    ordinary=periodic["branches"]["ordinary"]
    inverse=periodic["branches"]["inverse"]
    K=2*(A+B).bit_length()+F(L+C,3)
    result=dict(status="unresolved",height_difference_bound=str(K))
    if inverse["status"]!="excluded" or lo<=0:
        return result
    N=int(ordinary["min_abs_n_lower_bound"])
    M=int(ordinary["merged_period"])
    D=gcd(M,*ordinary["residues"])
    result.update(min_abs_n=N,divisor=D)
    # If different absolute indices are both multiples of D, the gap
    # is at least D*(2*N+D). Equal indices imply the midpoint.
    if sq((A+B)**2+4*A*A) and sq((A+B)**2+4*B*B):
        result["status"]="solution"
        result["solution"]=exact_check(A,B,F(A+B,2))
        return result
    if 2*lo*D*(2*N+D)>K:
        result["status"]="excluded_gap"
        return result
    max_small=max(0,(K/(2*lo*D)-D)//2)
    result["max_smaller_index"]=max_small
    if max_small>exact_n_cap:
        result["status"]="exact_index_cap"
        return result
    c2,c4=coeff(A,B)
    count=0
    for n in range(max(D,((N+D-1)//D)*D),max_small+1,D):
        # Both signs of n produce the same positive height. Check raw
        # local conditions from the entire CRT run, not just partial classes.
        if any(n%m not in S and (-n)%m not in S
               for p,m,S,_ in periodic["primes"]):
            continue
        try:
            Q=multiply(n,P,c2,c4,bit_cap)
        except WorkLimit:
            result["status"]="exact_bit_cap"
            return result
        if Q is None or Q[0]==0:
            continue
        x=Q[0]
        y=F(isqrt(x.numerator),isqrt(x.denominator))
        if y*y!=x:
            raise ArithmeticError("invalid doubled point")
        count+=1
        hit=exact_check(A,B,y)
        if hit:
            result.update(status="solution",solution=hit,exact_tested=count)
            return result
    result.update(status="excluded_finite_gap",exact_tested=count)
    return result


def classify(A,B,Rs,opts):
    if not (0<A<B and gcd(A,B)==1 and Rs):
        raise ValueError("invalid model")
    c2,c4=coeff(A,B)
    if any(not on_curve(R,c2,c4) for R in Rs):
        raise ValueError("off-curve input")
    result=dict(A=str(A),B=str(B),generators=[encode_point(R) for R in Rs],
                status="unresolved_factorization",rank_lower=0,rank_upper=None)
    fac=factor_inputs(A,B,opts["factor_limit"])
    if fac is None:
        return result
    result["factors"]={k:{str(p):e for p,e in v.items()} for k,v in fac.items()}
    L,C,D=height_constants(A,B,fac)
    result["height_constants"]=dict(L=L,C=C,gcd_bound=str(D))
    Ps=[add(R,R,c2,c4) for R in Rs]
    intervals=[height_interval(P,A,B,L,C,opts["height_steps"],opts["bit_cap"]) for P in Ps]
    result["heights"]=[dict(P=encode_point(P),lower=str(lo),upper=str(hi),bits=lens)
                       for P,(lo,hi,lens) in zip(Ps,intervals)]
    ep=set(fac["A"])|set(fac["B"])
    dp=set(fac["sum"])|set(fac["difference"])
    realized=[-1,squareclass(F(A*B),ep)]+[squareclass(R[0],ep) for R in Rs if R is not None]
    dim=class_span(realized,ep)
    lower=max(0,dim-2,int(any(lo>0 for lo,_,_ in intervals)))
    # A positive determinant of the canonical-height Gram matrix proves
    # independence, even if the two points have identical squareclasses.
    for i in range(min(len(Ps),opts["independence_cap"])):
        for j in range(i):
            a0,a1,_=intervals[i]
            b0,b1,_=intervals[j]
            S=add(Ps[i],Ps[j],c2,c4)
            s0,s1,_=height_interval(S,A,B,L,C,opts["height_steps"],opts["bit_cap"])
            cross=max(abs(s0-a1-b1),abs(s1-a0-b0))/2
            if a0*b0>cross*cross:
                lower=max(lower,2)
                result["independence"]=[i,j,str(a0*b0-cross*cross)]
    result["rank_lower"]=lower
    mods=list(dict.fromkeys([256,27,25,49,121]+primes_up_to(opts["descent_prime_limit"])))
    if opts.get("bad_prime_linear",0):
        # All factored bad primes enter the symbolic constraints, even
        # those far above the finite modular enumeration limit.
        d1=linear_descent_side(c2,c4,ep,dp,mods,opts["class_cap"])
        d2=linear_descent_side(-2*c2,(B*B-A*A)**2,dp,ep,mods,opts["class_cap"],dual=True)
        for d in realized:
            check_linear_realized(d,d1)
        check_linear_realized(1,d2)
    else:
        d1=descent_side(c2,c4,ep,mods,opts["class_cap"])
        d2=descent_side(-2*c2,(B*B-A*A)**2,dp,mods,opts["class_cap"],positive=True)
    result["descent"]=dict(E=d1,dual=d2,realized_E=realized,moduli=mods)
    if d1 is not None and d2 is not None:
        if d1["survivors"] is not None and any(d not in d1["survivors"] for d in realized):
            raise ArithmeticError("realized squareclass excluded")
        upper=d1["dimension_upper"]+d2["dimension_upper"]-2
        if upper<lower:
            raise ArithmeticError("inconsistent rank bounds")
        result["rank_upper"]=upper
    else:
        upper=None
    if lower>=2:
        result["status"]="rank_at_least_2"
        return result
    tors=torsion_certificate(A,B,primes_up_to(opts["prime_limit"]))
    result["torsion"]=tors
    if upper==0 and tors["exact_order"]==8:
        # Then 2E(Q)={O,T}; neither yields a positive finite height.
        result["status"]="line_excluded_rank_zero"
        return result
    if not any(lo>0 for lo,_,_ in intervals):
        result["status"]="unresolved_height"
        return result
    # Prefer smaller canonical height among certified non-torsion seeds.
    i=min((i for i,h in enumerate(intervals) if h[0]>0),key=lambda i:intervals[i][1])
    P=Ps[i]
    lo,hi,_=intervals[i]
    result["selected"]=i
    periodic=periodic_analyze(Rs[i],A,B,opts["prime_limit"],opts["node_cap"],opts["crt_class_cap"])
    result["periodic"]=periodic
    gap=subgroup_gap(A,B,P,lo,L,C,periodic,opts["exact_n_cap"],opts["bit_cap"])
    result["subgroup"]=gap
    if gap["status"]=="solution":
        result["status"]="solution"
        result["solution"]=gap["solution"]
        return result
    if not gap["status"].startswith("excluded_"):
        result["status"]="unresolved_subgroup"
        return result
    result["status"]="subgroup_excluded_rank_unknown"
    if lower!=1 or upper!=1:
        return result
    if tors["exact_order"]!=8:
        result["status"]="subgroup_excluded_torsion_unknown"
        return result
    sat=finite_saturation(P,A,B,L,C,hi,lo,opts["height_steps"],opts["bit_cap"],opts["saturation_cap"])
    result["saturation"]=sat
    result["status"]="line_excluded_rank_one" if sat["status"]=="saturated" else "subgroup_excluded_saturation_unknown"
    return result


DEFAULTS=dict(factor_limit=100000,descent_prime_limit=97,class_cap=4096,
    height_steps=4,bit_cap=32768,prime_limit=251,node_cap=10000,
    crt_class_cap=1024,exact_n_cap=100,saturation_cap=1024,independence_cap=4,bad_prime_linear=1)


def load_models(paths,max_models):
    groups=defaultdict(set)
    rows=0
    stopped=False
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row=json.loads(line)
                A,B=int(row["A"]),int(row["B"])
                R=tuple(F(v) for v in row["R"])
                if (A,B) not in groups and len(groups)>=max_models:
                    stopped=True
                    continue
                rows+=1
                groups[A,B].add(R)
    return groups,rows,stopped


def run(paths,output,records,max_models,opts):
    out,rec=Path(output),Path(records)
    if out.resolve()==rec.resolve() or {out.resolve(),rec.resolve()}&{Path(p).resolve() for p in paths}:
        raise ValueError("output conflicts")
    for p in (out,rec):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            raise FileExistsError(f"refusing overwrite: {p}")
    tmp=out.with_name(out.name+".partial")
    part=rec.with_name(rec.name+".partial")
    groups,rows,stopped=load_models(paths,max_models)
    stats=Counter(input_rows=rows,curves=0,line_excluded=0,subgroup_excluded=0,
                  rank_one=0,rank_at_least_2=0,solutions=0,stopped_at_curve_cap=int(stopped))
    rank_hist=Counter()
    linear_hist=Counter()
    with gzip.open(part,"wt",encoding="utf-8") as f:
        for (A,B),Rs in sorted(groups.items()):
            r=classify(A,B,sorted(Rs),opts)
            status=r["status"]
            rank_hist[(r["rank_lower"],r["rank_upper"])]+=1
            for side in ("E","dual"):
                d=r.get("descent",{}).get(side)
                if d is not None and "linear_dimension" in d:
                    linear_hist[(side,len(d["basis_generators"])-d["linear_dimension"])]+=1
                    stats["linear_constraints_independent_"+side]+=len(d["basis_generators"])-d["linear_dimension"]
                    stats["linear_enumeration_cap_"+side]+=int(not d["enumeration_complete"])
            stats["curves"]+=1
            stats["status_"+status]+=1
            stats["rank_one"]+=int(r["rank_lower"]==r["rank_upper"]==1)
            stats["rank_at_least_2"]+=int(r["rank_lower"]>=2)
            stats["line_excluded"]+=int(status.startswith("line_excluded_"))
            stats["subgroup_excluded"]+=int(status.startswith("subgroup_excluded_") or status=="line_excluded_rank_one")
            if status=="solution":
                stats["solutions"]+=1
                print("POTENTIAL FOUR-DISTANCE SOLUTION!",json.dumps(r["solution"]),flush=True)
            f.write(json.dumps(r,separators=(",",":"))+"\n")
            if stats["curves"]%25==0:
                print(f"curves={stats['curves']} rank1={stats['rank_one']} rank2plus={stats['rank_at_least_2']} "
                      f"subgroup_excluded={stats['subgroup_excluded']} line_excluded={stats['line_excluded']}",flush=True)
    summary=dict(method="rank-height-saturation-v2" if opts.get("bad_prime_linear") else "rank-height-saturation-v1",
                 parameters=opts,inputs=paths,stats=dict(stats),
                 rank_bounds=[[lo,hi,count] for (lo,hi),count in rank_hist.most_common()],
                 linear_dimension_reductions=[[side,k,count] for (side,k),count in linear_hist.most_common()])
    tmp.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    part.replace(rec)
    tmp.replace(out)
    print(json.dumps(dict(stats)),flush=True)
    return summary


def verify(summary_path,records):
    with open(summary_path,encoding="utf-8") as f:
        summary=json.load(f)
    opts=summary["parameters"]
    n=0
    with gzip.open(records,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            Rs=[tuple(F(v) for v in R) for R in r["generators"]]
            actual=classify(int(r["A"]),int(r["B"]),Rs,opts)
            if actual!=r:
                raise ArithmeticError("record mismatch at curve "+r["A"]+"/"+r["B"])
            n+=1
    if n!=summary["stats"]["curves"]:
        raise ArithmeticError("record count mismatch")
    print(f"verified {n} curve records by exact recomputation")


def self_test():
    from tempfile import TemporaryDirectory
    assert prime64(18446744073709551557)
    assert not prime64(341550071728321)

    # Exhaustively compare the binary kernel against all masks.
    import random
    rng=random.Random(20261125)
    for n in range(1,10):
        for _ in range(20):
            rows=[rng.randrange(1<<n) for j in range(n+2)]
            basis=binary_kernel(rows,n)
            masks=[0]
            for v in basis:
                masks += [m^v for m in masks]
            brute={m for m in range(1<<n)
                   if all((m&r).bit_count()%2==0 for r in rows)}
            assert set(masks)==brute and len(masks)==len(brute)
    # Independently test local conditions at all odd bad primes of
    # small models, and compare 2-adic code filtering to direct mod256.
    for a in range(1,15):
        for b in range(a+1,31):
            if gcd(a,b)!=1:
                continue
            fac=factor_inputs(a,b,100)
            ep=set(fac["A"])|set(fac["B"])
            dp=set(fac["sum"])|set(fac["difference"])
            aa,bb=coeff(a,b)
            for ca,cb,ps,qs,dual in ((aa,bb,ep,dp,False),
                    (-2*aa,(b*b-a*a)**2,dp,ep,True)):
                d=linear_descent_side(ca,cb,ps,qs,[256,27,25],4096,dual)
                gens=d["basis_generators"]
                for sc in classes(ps,4096):
                    mask=class_mask(sc,gens)
                    for kind,p,row in d["linear_constraints"]:
                        if not (mask&row).bit_count()%2:
                            continue
                        if kind=="real":
                            assert sc<0
                        elif kind=="legendre":
                            assert not locally_possible(sc,ca,cb,p)
                        elif kind=="two_adic":
                            assert not locally_possible(sc,ca,cb,256)
                        elif kind=="valuation":
                            # Test deeply enough to see the odd valuation
                            # obstruction; restrict work to small moduli.
                            q=p
                            while cb%q==0 and q<=256:
                                q*=p
                            if q<=256:
                                assert not locally_possible(sc,ca,cb,q)
                assert d["survivors"] is not None
                old=descent_side(ca,cb,ps,[256,27,25],4096,dual)
                assert set(d["survivors"])<=set(old["survivors"])
    # A large bad prime supplies a constraint without scanning p residues.
    # A=1, B=102 has B^2-A^2=101*103. p=101 is beyond the old limit 97.
    a,b=1,102
    ca,cb=coeff(a,b)
    x=linear_descent_side(ca,cb,{2,3,17},{101,103},[256],4096)
    assert any(kind=="legendre" and p==101 for kind,p,_ in x["linear_constraints"])
    # Even a cap of one retains a rigorous dimension bound.
    y=linear_descent_side(ca,cb,{2,3,17},{101,103},[256],1)
    assert y["dimension_upper"]>=x["dimension_upper"]
    assert y["survivors"] is None or y["survivors"]==[1]
    print("linear descent tests passed: kernels, bad primes, 2-adic classes, cap bounds")

    A,B=7,45
    R=(F(27),F(-2052))
    opts=DEFAULTS.copy()
    fac=factor_inputs(A,B,opts["factor_limit"])
    L,C,D=height_constants(A,B,fac)
    assert L==54 and C==34,(L,C,D)
    r=classify(A,B,[R],opts)
    print("pilot",json.dumps({k:v for k,v in r.items() if k!="periodic"}))
    assert r["rank_lower"]==r["rank_upper"]==1,r["descent"]
    assert r["torsion"]["exact_order"]==8
    assert r["saturation"]["status"]=="saturated"
    assert {x["y"] for x in r["saturation"]["heights"]}=={"24","105/8"}
    assert r["status"]=="line_excluded_rank_one"
    small=opts|{"saturation_cap":16}
    assert classify(A,B,[R],small)["status"]=="subgroup_excluded_saturation_unknown"
    # Check the duplication gcd estimate directly on small coprime squares.
    for a,b in ((1,2),(2,3),(7,45),(3,11),(1,7)):
        fac=factor_inputs(a,b,100)
        l,c,d=height_constants(a,b,fac)
        for i in range(1,35):
            for j in range(1,35):
                if gcd(i,j)!=1:
                    continue
                s,t=i*i,j*j
                num=(s*s-(a*b)**2*t*t)**2
                den=4*s*t*(s+a*a*t)*(s+b*b*t)
                assert d%gcd(num,den)==0,(a,b,i,j,d,gcd(num,den))
                H=max(s,t)
                assert max(num,den)*2*a*b>=H**4
    # Recompute serialized proof data, including the intentional cap path.
    with TemporaryDirectory() as root:
        f=Path(root)
        src=f/"models.gz"
        with gzip.open(src,"wt") as out:
            out.write(json.dumps(dict(A=str(A),B=str(B),R=encode_point(R)))+"\n")
        run([str(src)],f/"summary.json",f/"records.gz",1,small)
        verify(f/"summary.json",f/"records.gz")
    print("self-test passed: rank, heights, saturation, line exclusion and replay")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--prior-models",nargs="+",default=[])
    ap.add_argument("--output",default="local-classify/summary.json")
    ap.add_argument("--records",default="local-classify/proofs.jsonl.gz")
    ap.add_argument("--max-models",type=int,default=2000)
    ap.add_argument("--verify",action="store_true")
    for k,v in DEFAULTS.items():
        ap.add_argument("--"+k.replace("_","-"),type=int,default=v)
    a=ap.parse_args()
    if a.self_test:
        self_test()
        raise SystemExit(0)
    if a.verify:
        verify(a.output,a.records)
        raise SystemExit(0)
    if not a.prior_models:
        ap.error("provide --prior-models")
    opts={k:getattr(a,k) for k in DEFAULTS}
    limits=dict(factor_limit=(10,1000000),descent_prime_limit=(3,251),
        class_cap=(16,65536),height_steps=(0,8),bit_cap=(1024,1000000),
        prime_limit=(5,1000),node_cap=(1,100000),crt_class_cap=(1,10000),
        exact_n_cap=(0,10000),saturation_cap=(1,4096),independence_cap=(1,16),bad_prime_linear=(0,1))
    if not 1<=a.max_models<=100000 or any(not limits[k][0]<=v<=limits[k][1] for k,v in opts.items()):
        ap.error("invalid resource limit")
    run(a.prior_models,a.output,a.records,a.max_models,opts)
