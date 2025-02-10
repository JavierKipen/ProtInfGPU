#include "CrossValWrapper.h"
#include <string>
#include "InputParser.h"

using namespace std;

void setConfigurations(InputParser& IP, CrossValWrapper& CVw);

int main(int argc, char** argv) {
    InputParser IP;
    if(IP.parse(argc, argv))
    {
        CrossValWrapper CVW;
        setConfigurations(IP, CVW);
        CVW.computeEMCrossVal();
    }
    else
        IP.displayInfoMsg();
    
    return 0;
}
void setConfigurations(InputParser& IP, CrossValWrapper& CVW)
{
    CVW.nSparsityRed=IP.nSparsity;
    CVW.nEpochs=IP.nEpochs;
    CVW.inputFolderPath=IP.inputDir;
    CVW.outFolderPath=IP.outputDir;
    CVW.gW.gCM.NThreadsPerBlock=IP.nTreadsPerBlock;
    CVW.FSI.nCrossVal=IP.nCrossValDs;
    CVW.FSI.limitRAMGb= IP.limitRAMGb;
    CVW.limitMemGPUGb=IP.limitMemGPUGb;
    CVW.oracle=IP.useOracle;
    CVW.deviceN=IP.deviceN;
    CVW.oraclePErr=IP.oraclePErr;
    if(IP.nSubsetCV>0)
    {
        CVW.FSI.useSubsetCV=true;
        CVW.FSI.nSubsetCV=IP.nSubsetCV;
    }
    CVW.init();
}