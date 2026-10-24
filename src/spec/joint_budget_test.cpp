#include "strata/spec/joint_budget.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

static void check(bool x){if(!x)throw std::runtime_error("joint budget invariant");}
int main(){
    using strata::spec::JointDraftBudget;int decisions=0;
    for(int limit=1;limit<=3;++limit)for(bool ap:{false,true})for(bool ab:{false,true})
        for(int cap=0;cap<=7;++cap){
            JointDraftBudget b(limit,ap,ab);
            if(cap<2){auto x=b.choose(cap);check(x.prefix==cap&&x.candidates==cap);continue;}
            int best=0;
            for(int step=0;step<500;++step){
                auto x=b.choose(cap),again=b.choose(cap);++decisions;
                check(x.prefix==again.prefix&&x.candidates==again.candidates);
                check(x.prefix>=1&&x.prefix<=std::min(limit,cap-1)&&x.candidates>x.prefix&&x.candidates<=cap);
                if(!ap)check(x.prefix==std::min(limit,cap-1));
                if(!ab)check(x.candidates==cap);
                const bool optimum=x.prefix==std::min(limit,cap-1)&&x.candidates==cap;
                if(step>=400&&optimum)++best;
                // Synthetic controller fixture, not an inference speed result.
                b.observe(x.prefix,x.candidates,1,optimum?1.0:100.0);
            }
            check(best>=90);
        }
    JointDraftBudget b(2,true,true);int refused=0;
    for(int kind=0;kind<5;++kind)try{
        b.observe(kind==0?0:1,kind==1?1:3,kind==2?5:1,
                  kind==3?0:kind==4?std::numeric_limits<double>::quiet_NaN():1);
    }catch(const std::invalid_argument&){++refused;}
    check(refused==5);std::printf("PASS JOINT_BUDGET decisions=%d invalid=%d\n",decisions,refused);
}
