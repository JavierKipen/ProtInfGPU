#include "CrossValWrapper.h"
#include <string>

using namespace std;

int main() {
    string classifierPath = "/home/jkipen/raid_storage/ProtInfGPU/data/5_Prot/binary/rf_n_est_10_depth_10";
    string outPath= "ASD";
    CrossValWrapper CVW;
    CVW.init(outPath,classifierPath);
    
    CVW.computeEMCrossVal();
    
    return 0;
}
