#include "CrossValWrapper.h"
#include <string>

using namespace std;

int main(int argc, char** argv) {
    string classifierPath = "/home/jkipen/raid_storage/ProtInfGPU/data/5_Prot/binary/rf_n_est_10_depth_10";
    string outPath= "/home/jkipen/ProtInfGPU/results/5Prot";
    CrossValWrapper CVW;
    CVW.init(outPath,classifierPath);
    
    CVW.computeEMCrossVal();
    
    for(unsigned int i=0;i<10;i++)
        cout << to_string(CVW.ErrMean[i]) << " , ";
    
    return 0;
}
