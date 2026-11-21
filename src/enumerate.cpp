// Exhaustive enumeration of ordered pairs of primitive Pythagorean triples up to Euclid parameter m.
// This is exhaustive in the PARAMETER PAIRS only, not in the rational plane or denominator Q.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using u64=std::uint64_t;
using u128=unsigned __int128;
struct Triple {u64 h,v,hyp;};
struct Point {u64 x,y,q,d0,d1,d2;};
static u64 root(u128 v){
    u64 r=static_cast<u64>(sqrtl(static_cast<long double>(v)));
    while(static_cast<u128>(r)*r>v)--r;
    while(r<std::numeric_limits<u64>::max() && static_cast<u128>(r+1)*(r+1)<=v)++r;
    return r;
}
static std::vector<Triple> triples(u64 max_m){
    std::vector<Triple> ts;
    for(u64 m=2;m<=max_m;m++)for(u64 n=1;n<m;n++){
        if(((m^n)&1)==0 || std::gcd(m,n)!=1)continue;
        u64 a=m*m-n*n,b=2*m*n,c=m*m+n*n;
        ts.push_back({a,b,c});ts.push_back({b,a,c});
    }
    return ts;
}
static std::array<bool,64> build_qr64(){std::array<bool,64> a{};for(int i=0;i<64;i++)a[(i*i)&63]=true;return a;}
static std::array<bool,64> qr64=build_qr64();
template <int N>static bool isqrmod(u64 x,u64 y){
    u64 a=x%N,b=y%N,r=(a*a+b*b)%N;
    for(int i=0;i<N;i++)if((i*i)%N==r)return true;
    return false;
}
static u64 arg(const std::string&s){size_t at=0;u64 v=std::stoull(s,&at);if(at!=s.size())throw std::runtime_error("bad parameter: "+s);return v;}
int main(int argc,char**argv)try{
    u64 max_m=200,shard=0,shards=1,batch=0,batches=1;
    std::string out="seeds.json";
    for(int i=1;i<argc;i++){
        std::string s=argv[i];if(s=="--self-test"){
            auto ts=triples(4);
            bool found15=false,found7=false;
            for(auto t:ts){if(t.h==15&&t.v==8&&t.hyp==17)found15=true;if(t.h==7&&t.v==24&&t.hyp==25)found7=true;}
            if(!found15||!found7)throw std::runtime_error("missing reference triples");
            if(root(833*64)!=230) { /* floor(sqrt(53312))=230 */ throw std::runtime_error("sqrt test"); }
            std::cout<<"self-test passed\n";return 0;
        }
        auto p=s.find('=');if(p==std::string::npos)throw std::runtime_error("use --key=value");
        auto k=s.substr(0,p),v=s.substr(p+1);
        if(k=="--max-m")max_m=arg(v);
        else if(k=="--shard")shard=arg(v);
        else if(k=="--shards")shards=arg(v);
        else if(k=="--batch")batch=arg(v);
        else if(k=="--batches")batches=arg(v);
        else if(k=="--output")out=v;
        else throw std::runtime_error("unknown flag "+k);
    }
    if(max_m<4||max_m>1200||shards==0||shards>64||batches==0||batches>512||shard>=shards||batch>=batches)throw std::runtime_error("unsafe/invalid parameters");
    const auto ts=triples(max_m);
    auto n=ts.size();
    u64 stride=shards*batches,which=batch*shards+shard;
    u64 selected=0;
    for(u64 i=which;i<n;i+=stride)selected++;
    if(selected*n>4000000000ULL) throw std::runtime_error("work budget exceeded: adjust batches/shards/max_m");
    u64 tested=0,within=0,sieve=0,third=0,full=0;
    std::set<std::tuple<u64,u64,u64>> unique;
    std::vector<Point> points;
    for(u64 i=which;i<n;i+=stride){
        auto t1=ts[i];
        for(const auto&t2:ts){
            tested++;
            u64 X=t1.h*t2.v,Z=t2.h*t1.v,Y=t1.v*t2.v,Q=X+Z;
            if(Y>=Q)continue;
            within++;
            u64 V=Q-Y;
            if(!qr64[((X&63)*(X&63)+(V&63)*(V&63))&63])continue;
            if(!isqrmod<5>(X,V)||!isqrmod<7>(X,V)||!isqrmod<11>(X,V)||!isqrmod<13>(X,V))continue;
            sieve++;
            u128 c2=static_cast<u128>(X)*X+static_cast<u128>(V)*V;
            u64 C=root(c2);if(static_cast<u128>(C)*C!=c2)continue;
            third++;
            u64 g=std::gcd(std::gcd(X,Y),Q);
            u64 x=X/g,y=Y/g,q=Q/g;
            if(!unique.insert({x,y,q}).second)continue;
            auto d=static_cast<u128>(Z)*Z+static_cast<u128>(V)*V;
            auto D=root(d);if(static_cast<u128>(D)*D==d)full++;
            points.push_back({x,y,q,t1.hyp*t2.v/g,t2.hyp*t1.v/g,C/g});
        }
    }
    std::ofstream o(out);if(!o)throw std::runtime_error("cannot open "+out);
    o<<"{\n \"method\":\"exhaustive-primitive-triple-pairs\",\n \"max_m\":"<<max_m<<",\"shard\":"<<shard<<",\"shards\":"<<shards<<",\"batch\":"<<batch<<",\"batches\":"<<batches<<",\n"
     <<" \"triple_orientations\":"<<n<<",\"tested_pairs\":"<<tested<<",\"inside\":"<<within<<",\"sieve_passes\":"<<sieve<<",\"three_before_dedupe\":"<<third<<",\"unique_three\":"<<points.size()<<",\"four_distance\":"<<full<<",\n \"seeds\":[\n";
    for(size_t i=0;i<points.size();i++){
        auto p=points[i];
        o<<"  {\"x\":\""<<p.x<<"\",\"y\":\""<<p.y<<"\",\"q\":\""<<p.q<<"\",\"d\":[\""<<p.d0<<"\",\""<<p.d1<<"\",\""<<p.d2<<"\"]}"<<(i+1==points.size()?"":",")<<"\n";
    }
    o<<" ]\n}\n";
    std::cout<<"N="<<n<<" tested="<<tested<<" 3-distance="<<third<<" unique="<<points.size()<<" four-distance="<<full<<"\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
