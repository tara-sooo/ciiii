#!/usr/bin/env python3
"""Exact rational 3-distance generation on genus-one quartics.

For each primitive slope a:b and 3-distance seed P=(a*t,b*t), the
conic parameterization reduces the other two distances to a quartic
w^2=f(s) with square leading coefficient. Tangent quadratics through
rational points produce further rational points. This is a bounded
experimental rational-point walk, not a proof of completeness.
"""
import argparse
from collections import deque
from fractions import Fraction as F
from hashlib import sha256
from math import gcd, isqrt, log10
import gzip
import json
import sys


def qpoly_coeff(a, b, h2):
    # f(s)=(h2-s^2)^2-4*b*(s+a)*(h2-s^2)+4*h2*(s+a)^2
    return [1, 4*b, -2*h2+4*a*b+4*h2,
            -4*b*h2+8*h2*a, h2*h2-4*a*b*h2+4*h2*a*a]


def quartic(s, cf):
    return (((s+cf[1])*s+cf[2])*s+cf[3])*s+cf[4]


def tangent(s, w, cf, lead):
    if w == 0:
        return None
    deriv = 4*s*s*s + 3*cf[1]*s*s + 2*cf[2]*s + cf[3]
    d = deriv/(2*w)
    beta = d-2*lead*s
    gamma = w+lead*s*s-d*s
    g3 = cf[1]-2*lead*beta
    if not g3:
        return None
    g2 = cf[2]-(beta*beta+2*lead*gamma)
    new_s = -g2/g3 - 2*s
    new_w = lead*new_s*new_s + beta*new_s + gamma
    if new_w*new_w != quartic(new_s,cf):
        raise AssertionError('quartic tangent did not preserve curve')
    return new_s,new_w


def nearest_square_root(n):
    r=isqrt(n)
    if n-r*r > (r+1)*(r+1)-n:
        r+=1
    return r,abs(n-r*r)


def log_int(n):
    # Avoid float conversion of huge integers.
    if n <= 0: return float('-inf')
    bits=n.bit_length()
    shift=max(0,bits-53)
    return log10(n>>shift)+shift*log10(2)


def candidate(s,w,a,b,h2,cf,bit_cap):
    D=h2-s*s
    if not D: return None
    t=2*(s+a)/D
    if t == 0: return None
    q=t.denominator
    if q.bit_length()>bit_cap: return 'over'
    X=a*t.numerator
    Y=b*t.numerator
    ns=[X*X+Y*Y,(q-X)**2+Y*Y,X*X+(q-Y)**2]
    ds=list(map(isqrt,ns))
    if any(z*z!=n for z,n in zip(ds,ns)):
        raise AssertionError('a three-distance identity failed')
    N=(q-X)**2+(q-Y)**2
    R,delta=nearest_square_root(N)
    inside=0<X<q and 0<Y<q
    return dict(x=X,y=Y,q=q,d=ds,fourth_squared=N,nearest=R,gap=delta,
                gap_log10=log_int(delta)-log_int(max(1,2*R)),q_bits=q.bit_length(),inside=inside)


def seeds_to_initial_point(p):
    x,y,q=(int(p[k]) for k in ('x','y','q'))
    A,B,C=[int(v) for v in p['d']]
    if not (0<x<q and 0<y<q and gcd(gcd(x,y),q)==1):
        raise AssertionError('seed is not a reduced interior point')
    if (x*x+y*y!=A*A or (q-x)**2+y*y!=B*B or x*x+(q-y)**2!=C*C):
        raise AssertionError('seed distances inconsistent')
    g=gcd(x,y)
    a,b=x//g,y//g
    h2=a*a+b*b
    h=isqrt(h2)
    if h*h!=h2 or gcd(a,b)!=1:
        raise AssertionError('seed slope is not primitive Pythagorean')
    t=F(g,q)
    s=F(B-q,g)
    D=h2-s*s
    w=F(C,q)*D
    cf=qpoly_coeff(a,b,h2)
    if w*w!=quartic(s,cf) or a*t!=F(x,q) or b*t!=F(y,q):
        raise AssertionError('seed failed quartic embedding')
    return a,b,h2,cf,s,w


def public_record(c):
    return {k:(str(v) if isinstance(v,int) and k not in ('q_bits','inside') else v)
            for k,v in c.items()}


def walk(seed_file, output, records, depth, bit_cap, state_cap):
    seeds=json.load(open(seed_file))
    visited=set()
    unique_points=set()
    frontier=deque()
    orbit_seeds=0
    for p in seeds['seeds']:
        a,b,h2,cf,s,w=seeds_to_initial_point(p)
        for signed_w in (w,-w):
            key=(a,b,s,signed_w)
            if key not in visited:
                visited.add(key)
                frontier.append((a,b,h2,cf,s,signed_w,0))
        orbit_seeds+=1
    stats=dict(input_seeds=orbit_seeds,initial_states=len(visited),visited=0,duplicate_points_skipped=0,three_distance_points=0,
               inside=0,outside=0,over_50_billion=0,max_q_bits=0,bit_cap_skipped=0,
               degenerate=0,found_four=0,stopped_at_state_cap=False)
    best=[]; solutions=[]
    with gzip.open(records,'wt',encoding='utf-8') as dest:
        while frontier and stats['visited']<state_cap:
            a,b,h2,cf,s,w,d=frontier.popleft()
            c=candidate(s,w,a,b,h2,cf,bit_cap)
            stats['visited']+=1
            if c=='over':
                stats['bit_cap_skipped']+=1
                continue
            if c is not None:
                point_key=(c['x'],c['y'],c['q'])
                if point_key in unique_points:
                    stats['duplicate_points_skipped']+=1
                    c=None
                else:
                    unique_points.add(point_key)
            if c is not None:
                stats['three_distance_points']+=1
                stats['max_q_bits']=max(stats['max_q_bits'],c['q_bits'])
                stats['inside' if c['inside'] else 'outside']+=1
                stats['over_50_billion']+=c['q']>50_000_000_000
                dest.write(json.dumps(public_record(c),separators=(',',':'))+'\n')
                if c['gap']==0:
                    stats['found_four']+=1
                    solutions.append(public_record(c))
                else:
                    # Rank near misses by how close the *integer* 4th root is to an integer.
                    best.append((c['gap_log10'],c))
                    best.sort(key=lambda p:p[0])
                    if len(best)>10:best.pop()
            if d>=depth: continue
            for lead in (1,-1):
                nxt=tangent(s,w,cf,lead)
                if nxt is None:
                    stats['degenerate']+=1;continue
                ns,nw=nxt
                if max(ns.numerator.bit_length(),ns.denominator.bit_length())>bit_cap*2:
                    stats['bit_cap_skipped']+=1;continue
                key=(a,b,ns,nw)
                if key in visited:continue
                visited.add(key)
                frontier.append((a,b,h2,cf,ns,nw,d+1))
    stats['stopped_at_state_cap']=bool(frontier)
    output_data={
        'method':'exact-quartic-tangent-walk-v1',
        'scope':'3 rational distances certified for every recorded point; candidate fourth distance tested exactly; not exhaustive in Q',
        'seed_metadata':{k:v for k,v in seeds.items() if k!='seeds'},
        'max_depth':depth,'q_bit_cap':bit_cap,'state_cap':state_cap,
        'stats':stats,
        'best_integer_root_near_misses':[public_record(c) for _,c in best],
        'solutions':solutions,
    }
    with open(output,'w') as f:json.dump(output_data,f,indent=2)
    print('seeds={input_seeds} states={visited} 3-distance={three_distance_points} outside={outside} q>50B={over_50_billion} max_q_bits={max_q_bits} FOUR={found_four}'.format(**stats))
    return output_data


def self_test():
    p={'x':'45','y':'24','q':'52','d':['51','25','53']}
    a,b,h2,cf,s,w=seeds_to_initial_point(p)
    assert (a,b,h2,s,w)==(15,8,289,F(-9),F(212))
    c=candidate(s,w,a,b,h2,cf,4096)
    assert c['fourth_squared']==833 and c['gap']!=0
    for lead in (-1,1):
        ns,nw=tangent(s,w,cf,lead)
        d=candidate(ns,nw,a,b,h2,cf,4096)
        assert d and d!='over' and d['gap']!=0
    print('self-test passed: quartic map and integer distances')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--seeds',default='seeds.json')
    ap.add_argument('--output',default='orbits-summary.json')
    ap.add_argument('--records',default='orbits.jsonl.gz')
    ap.add_argument('--depth',type=int,default=5)
    ap.add_argument('--max-q-bits',type=int,default=4096)
    ap.add_argument('--max-states',type=int,default=50000)
    args=ap.parse_args()
    if args.self_test:
        self_test();sys.exit(0)
    if not(0<=args.depth<=9 and 128<=args.max_q_bits<=8192 and 1<=args.max_states<=100000):
        ap.error('resource limits exceeded')
    walk(args.seeds,args.output,args.records,args.depth,args.max_q_bits,args.max_states)
