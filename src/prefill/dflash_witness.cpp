#include "strata/core/dflash_native.hpp"
#include "strata/core/native_head.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc,char** argv){
    if(argc!=7){std::fprintf(stderr,"usage: dflash_native_witness PACK TARGET_GGUF SCRIPT CONTEXT eager|graph OUT\n");return 2;}
    std::string err;strata::core::NativeHead head;strata::core::NativeEmbed embed;
    strata::core::DflashNative draft;
    if(!head.load({argv[2]},2560,248320,err)||!embed.load({argv[2]},2560,248320,err)||
       !draft.load(argv[1],head,embed,std::stoi(argv[4]),std::string(argv[5])=="graph",false,err)){
        std::fprintf(stderr,"%s\n",err.c_str());return 1;}
    std::ifstream script(argv[3]);std::string line;int cases=0;
    if(!script)return 2;
    while(std::getline(script,line)){
        std::istringstream in(line);std::string op;in>>op;bool ok=false;
        if(op=="reset")ok=draft.reset(err);
        else if(op=="append"){
            std::string path;int rows;int64_t pos;in>>path>>rows>>pos;
            if(rows<1||rows>262144)return 2;
            std::vector<float> f(size_t(rows)*12800);std::ifstream input(path,std::ios::binary);
            if(!input.read((char*)f.data(),f.size()*4))return 2;
            ok=draft.append_host(f.data(),rows,pos,err);
        }else if(op=="draft"){
            int64_t pos;int anchor,count,got=0;std::string name;in>>pos>>anchor>>count>>name;
            int32_t ids[7];ok=draft.draft(pos,anchor,count,ids,got,err);
            if(ok){std::vector<float> hidden(size_t(got)*2560);ok=draft.hidden(hidden.data(),got,err);
                std::ofstream f(std::string(argv[6])+"/"+name+".f32",std::ios::binary);f.write((char*)hidden.data(),hidden.size()*4);
                std::ofstream tokens(std::string(argv[6])+"/"+name+".ids");
                for(int i=0;i<got;++i)tokens<<ids[i]<<' ';tokens<<'\n';++cases;
            }
        }else return 2;
        if(!ok){std::fprintf(stderr,"%s: %s\n",line.c_str(),err.c_str());return 1;}
    }
    std::printf("WITNESS DFLASH2_NATIVE cases=%d weight_bytes=%llu shared_head_bytes=%llu\n",cases,
                (unsigned long long)draft.weight_bytes(),(unsigned long long)head.weight_bytes());
    return cases?0:1;
}
