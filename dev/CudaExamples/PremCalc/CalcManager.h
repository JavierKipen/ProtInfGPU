#ifndef CALCMANAGER_H
#define CALCMANAGER_H

#include "DeviceData.h"

#include <cublas_v2.h>

#define DEFAULT_N_THREADS_PER_BLOCK 16

class CalcManager {  //Class to handle the data movement (later would be also binding and data transfer to python).
    
    public:
        CalcManager(); //
        ~CalcManager();
        void processReads();
        void setData(DeviceDataPXgICalc *devData,DeviceDataPXgICalc *d_devData); //Updates the pointers to the data
        void sumAlphas();
        void calcAlphas();
        void calcPXgI();
        void calcPXgIRel();
        void calcPRem();
    
        DeviceDataPXgICalc *devData; //Array on host with pointers to device memory
        DeviceDataPXgICalc *d_devData; //Array on device with pointers to all data to use in kernels.
    private:
        cublasHandle_t cuBlasHandle;
        cublasStatus_t cuBlasStatus; //Cublas status for debugging/error printing
        unsigned int NThreadsPerBlock;
        
    
};



#endif