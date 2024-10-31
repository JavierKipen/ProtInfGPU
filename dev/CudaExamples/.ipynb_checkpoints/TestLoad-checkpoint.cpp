#include "DataIO.h"


int main()
{
    InputDataPXgICalc inputData;
    String folder_path="/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/"; //Folder where input data is stored.
    loadExampleInput(InputDataPXgICalc &inputData, String folder_path);
    saveVectorToCSV(&(inputData.PFexpForI), "/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/out.csv")
    
    
}