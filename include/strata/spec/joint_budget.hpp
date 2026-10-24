#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>

namespace strata::spec {
// Choose an MTP prefix and total candidate count using completed joint rounds.
// Both stages contribute on normal windows; a short context/output tail can
// leave room only for MTP. No current draft confidence affects its own budget.
class JointDraftBudget {
public:
    struct Pick { int prefix=0, candidates=0; };
    JointDraftBudget(int prefix, bool adapt_prefix, bool adapt_block)
        : prefix_(prefix), adapt_prefix_(adapt_prefix), adapt_block_(adapt_block) {
        if(prefix<1||prefix>3)throw std::invalid_argument("joint prefix must be 1..3");
    }
    Pick choose(int cap) const {
        cap=std::clamp(cap,0,7);
        if(cap<2)return {cap,cap};
        Pick best{std::min(prefix_,cap-1),cap};double best_score=-1;
        unsigned oldest=0;
        for(int p=adapt_prefix_?1:best.prefix;p<=std::min(prefix_,cap-1);++p)
            for(int c=adapt_block_?p+1:cap;c<=cap;++c) {
                const auto& a=arms_[p][c];
                if(a.n<2)return {p,c};
                const double score=a.tokens/a.ms;
                if(score>best_score){best_score=score;best={p,c};}
            }
        // Revisit stale choices on a bounded schedule, including across tasks.
        if(rounds_ && rounds_%32==0)
            for(int p=adapt_prefix_?1:std::min(prefix_,cap-1);p<=std::min(prefix_,cap-1);++p)
                for(int c=adapt_block_?p+1:cap;c<=cap;++c)
                    if(rounds_-arms_[p][c].last>oldest){oldest=rounds_-arms_[p][c].last;best={p,c};}
        return best;
    }
    void observe(int prefix,int candidates,int emitted,double ms) {
        if(prefix<1||prefix>prefix_||candidates<=prefix||candidates>7||emitted<1||
           emitted>candidates+1||!std::isfinite(ms)||ms<=0)
            throw std::invalid_argument("invalid joint draft observation");
        auto& a=arms_[prefix][candidates];
        a.tokens=.8*a.tokens+emitted;a.ms=.8*a.ms+ms;++a.n;a.last=++rounds_;
    }
    std::string report() const {
        std::ostringstream s;s<<"joint budget:";
        for(int p=1;p<=prefix_;++p)for(int c=p+1;c<=7;++c){
            const auto& a=arms_[p][c];
            if(a.n)s<<" P"<<p<<"/C"<<c<<'='<<a.n<<'@'<<a.ms/a.tokens<<"ms/token";
        }
        return s.str();
    }
private:
    struct Arm { double tokens=0,ms=0;unsigned n=0,last=0; };
    std::array<std::array<Arm,8>,4> arms_{};
    int prefix_;bool adapt_prefix_,adapt_block_;unsigned rounds_=0;
};
} // namespace strata::spec
