#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include "DataIO.h"

using namespace std;

typedef struct {
    //Input variables
    unsigned int * d_FexpIdForI, * d_NFexpForI, * d_TopNFluExpId;
	float * d_PFexpForI,* d_TopNFluExpScores;
    //Variables used for calculation
    float * d_pRem; //Stores the p_Rem of the sparse matrixes.
} DeviceDataPXgICalc; //Inputs that are needed calculate P(X/I) but on the GPU (device)


class DataManager {  //Class to handle the data movement (later would be also binding and data transfer to python).
    
    public:
        DataManager();
        bool loadDataFromCSV(string folder);
        bool dataToGPU(DeviceDataPXgICalc *d_pDataPXgI);
        void free(DeviceDataPXgICalc *d_pDataPXgI);
        
            
        InputDataPXgICalc InputData;
        unsigned int n_prot,n_reads,n_sparsity,n_flu;
    private:
        void cpuDataToGPU(DeviceDataPXgICalc *d_pData);
        bool malloc_iData(DeviceDataPXgICalc *d_pData);
        
    
};


#endif