#include "DataManager.h"
#include "CalcManager.h"

using namespace std;



int main()
{
    DeviceDataPXgICalc pdevData, *d_pdevData; //Data for calculations. pdevData in host, the other is initialized in device!
    DataManager DM;
    DM.loadDataFromCSV("/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/"); //Loads the test data
    /*
    for(unsigned int i=0;i<20;i++)
        cout << DM.InputData.PFexpForI[i] << ",";
    */
    CalcManager CM;
    DM.dataToGPU(&pdevData,&d_pdevData); //Copies input data to GPU and allocates for internal variables 
    CM.setData(&pdevData,d_pdevData);
    CM.calcPRem();
    CM.calcPXgIRel();
    DM.freeData(&pdevData,d_pdevData);
    return 0;
}