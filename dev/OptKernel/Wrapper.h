#ifndef WRAPPER_H
#define WRAPPER_H

#include "DeviceData.h"
#include "GPUKernelManager.h"
#include "GPUDataManager.h"
#include "IOManager.h"

#define MAX_GB_GPU 30

class Wrapper {  
    
    public:
        Wrapper(); //
        ~Wrapper();
    
        void init();
        void timeBaseKernel();
        void checkFewProtFewReadPerBlock();
    
        void checkRowNonZero(vector<float> &mat,unsigned long nProt,unsigned long nReads);
        void checkNansMat(vector<float> &mat);
    
        GPUDataManager gDM;
        GPUKernelManager gKM;
        IOManager IOM;
        vector<float> PXgIrel; //Obtained result
    private:
        bool allocatedData;
        PNewData genPNewData();
        DeviceData devData, *d_pdevData; //p to dev data both on device and host, save on the call to pass within the functions easier.
};



#endif