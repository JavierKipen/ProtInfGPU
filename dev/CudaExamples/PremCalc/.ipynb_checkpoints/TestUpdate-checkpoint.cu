#include "DataManager.h"
#include "CalcManager.h"
#include <chrono>

using namespace std;


void comp2dArrays(float * pArrayA,float * pArrayB,unsigned int n_row,unsigned int n_col);
void print2Darray(float * pArray,unsigned int n_row,unsigned int n_col);

int main()
{
    DeviceDataPXgICalc pdevData, *d_pdevData; //Data for calculations. pdevData in host, the other is initialized in device!
    DataManager DM;
    DM.loadDataFromCSV("/home/jkipen/ProtInfGPU/dev/P_X_giv_I_rel_w_GPU/RepVars/"); //Loads the test data
    vector<float> updateVals(DM.n_prot, 0);
    CalcManager CM;
    DM.dataToGPU(&pdevData,&d_pdevData); //Copies input data to GPU and allocates for internal variables 
    CM.setData(&pdevData,d_pdevData);
    
    CM.processReads(updateVals.data());
    
    for(unsigned int i=0;i<DM.n_prot;i++)
        cout << to_string(updateVals[i]) << ",";
    
    DM.freeData(&pdevData,d_pdevData);
    return 0;
}


void comp2dArrays(float * pArrayA,float * pArrayB,unsigned int n_row,unsigned int n_col)
{
    float mae=0;
    float maxDiff=0;
    unsigned int maxDiffIdx=0;
    
    for(unsigned int i=0;i<n_row*n_col;i++)
    {
        float diff = abs(pArrayA[i]-pArrayB[i]);
        if(diff>maxDiff)
        {
            maxDiff=diff;
            maxDiffIdx=i;
        }
        mae += diff;
    }
    mae /= (float) (n_row*n_col);
    cout << "The obtained MAE was " << to_string(mae) << ". And the max diff was "  << to_string(maxDiff) << " , where i = " << to_string(maxDiffIdx) << " and MatA[i]=" << to_string(pArrayA[maxDiffIdx]) << " and MatB[i] " << to_string(pArrayB[maxDiffIdx]);
}

void print2Darray(float * pArray,unsigned int n_row,unsigned int n_col)
{
    cout <<"[[" << to_string(pArray[0]) << ","<< to_string(pArray[1]) << ","<< to_string(pArray[2]) << "..." <<  to_string(pArray[n_col-1]) << "]" <<endl;
    cout <<"[" << to_string(pArray[n_col+0]) << ","<< to_string(pArray[n_col+1]) << ","<< to_string(pArray[n_col+2]) << "..." <<  to_string(pArray[2*n_col-1]) << "]" <<endl;
    cout << "..." << endl;
    cout <<"[" << to_string(pArray[(n_row-1)*n_col]) << ","<< to_string(pArray[(n_row-1)*n_col+1]) << ","<< to_string(pArray[(n_row-1)*n_col+2]) << "..." <<  to_string(pArray[n_row*n_col-1]) << "]]" <<endl;
}