#include "FileSystemInterface.h"
#include <string>

using namespace std;

int main() {
    string classifierPath = "/home/jkipen/raid_storage/ProtInfGPU/data/5_Prot/binary/rf_n_est_10_depth_10";
    FileSystemInterface FSI(classifierPath);
    
    while(!FSI.finishedReading)
    {
        FSI.readPartialScores();
        for(unsigned int i=0;i<10;i++)
            cout << FSI.TopNScoresPartialFlattened[i] << ",";
        cout << FSI.TopNScoresPartialFlattened[10] << endl;
    }
    FSI.restartReading();
    while(!FSI.finishedReading)
    {
        FSI.readPartialScores();
        for(unsigned int i=0;i<10;i++)
            cout << FSI.TopNScoresPartialFlattened[i] << ",";
        cout << FSI.TopNScoresPartialFlattened[10] << endl;
    }
    return 0;
}
