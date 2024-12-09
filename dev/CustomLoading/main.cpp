#include "CrossValWrapper.h"
#include <string>

using namespace std;

int main() {
    string classifierPath = "/home/jkipen/raid_storage/ProtInfGPU/data/5_Prot/binary/rf_n_est_10_depth_10";
    string outPath= "ASD";
    FileSystemInterface FSI(classifierPath);
    CrossValWrapper CVW;
    CVW.init(outPath,&FSI);
    
    CVW.computeEMCrossVal();
    
    return 0;
}
