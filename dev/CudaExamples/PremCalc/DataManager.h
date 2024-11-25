#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include "DataIO.h"
#include "DeviceData.h"
using namespace std;




class DataManager {  //Class to handle the data movement (later would be also binding and data transfer to python).
    
    public:
        DataManager();
        ~DataManager();
        bool loadDataFromCSV(string folder);
        bool dataToGPU(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc **d_ppdevData);
        void freeData(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc *d_pdevData);
        
            
        InputDataPXgICalc InputData; //Data that will be loaded from the experiments.
        unsigned int n_prot,n_reads,n_sparsity,n_flu_exp,n_reads_max,onesVecLen;
    private:
        void createOnesVec(DeviceDataPXgICalc *pdevData);
        void cpuDataToGPU(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc *d_pdevData);
        bool malloc_iData(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc **d_ppdevData);
        
    
};


#endif