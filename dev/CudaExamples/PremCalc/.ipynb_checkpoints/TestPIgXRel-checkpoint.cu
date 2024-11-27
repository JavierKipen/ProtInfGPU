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
    CalcManager CM;
    DM.dataToGPU(&pdevData,&d_pdevData); //Copies input data to GPU and allocates for internal variables 
    CM.setData(&pdevData,d_pdevData);
    auto t1 = chrono::high_resolution_clock::now();
    for(unsigned int i=0;i<10;i++)
    {
        CM.calcPRem();
        CM.calcPXgIRel();
    }
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> ms_double = t2 - t1;
    cout << "The time to run 10 PXGIREL was " << ms_double.count()<< "ms " << endl;
    vector<float> calcPXgIRel(DM.n_reads*DM.n_prot, 0);
    cudaMemcpy(calcPXgIRel.data(), pdevData.d_MatAux, sizeof(float)*DM.n_reads*DM.n_prot, cudaMemcpyDeviceToHost);
    cout<< "Calculated in GPU: " << endl;
    print2Darray(calcPXgIRel.data(),DM.n_reads,DM.n_prot);
    cout<< "True: " << endl;
    print2Darray(DM.InputData.TruePXgivIrel.data(),DM.n_reads,DM.n_prot);
    comp2dArrays(calcPXgIRel.data(),DM.InputData.TruePXgivIrel.data(),DM.n_reads,DM.n_prot);
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