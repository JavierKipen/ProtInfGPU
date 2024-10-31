#include "DataIO.h"

using namespace std;

int main()
{
    InputDataPXgICalc inputData;
    string folder_path="/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/"; //Folder where input data is stored.
    //string folder_path = "C:/Users/JK-WORK/source/repos/TestDataIO/RepVars/"; //Folder where input data is stored.
    loadExampleInput(inputData,folder_path);
    saveVectorToCSV(&(inputData.PFexpForI), "/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/out.csv");
    //saveVectorToCSV(&(inputData.PFexpForI), "C:/Users/JK-WORK/source/repos/TestDataIO/RepVars/out.csv");
    return 0;
}