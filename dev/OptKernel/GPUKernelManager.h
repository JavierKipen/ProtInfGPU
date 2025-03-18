#ifndef GPUKERNELMANAGER_H
#define GPUKERNELMANAGER_H

#include "DeviceData.h"
#include <cublas_v2.h>


#define GPU_DEVICE 3
#define DEFAULT_N_THREADS_PER_BLOCK 512

class GPUKernelManager {  //Class to handle the data movement (later would be also binding and data transfer to python).
    
    public:
        GPUKernelManager(); //
        ~GPUKernelManager();
    
        void init();
        void runBaseKernel(DeviceData *pdevData, DeviceData *d_pdevData);
        void calcPRem(DeviceData *pdevData, DeviceData *d_pdevData);
        void runFewProtFewReadPerBlockNonOracle(DeviceData *pdevData, DeviceData *d_pdevData, unsigned int nProtPerBlock, unsigned int nReadPerBlock);
        void setPRemContribution(DeviceData *pdevData, DeviceData *d_pdevData);
        void initCublas();
        unsigned int NThreadsPerBlock;
    private:
        cublasHandle_t cuBlasHandle;
        //DeviceData *pdevData, *d_pdevData; //p to dev data both on device and host, save on the call to pass within the functions easier 
};



#endif