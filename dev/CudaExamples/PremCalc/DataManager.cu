#include "DataManager.h"


DataManager::DataManager()
{
}

bool DataManager::loadDataFromCSV(string folder_path)
{
    loadExampleInput(InputData, folder_path);
    n_prot=InputData.NFexpForI.size();
    n_sparsity=30; //Hardcoded examples are N=30
    n_reads=InputData.TopNFluExpId.size()/n_sparsity;
    n_flu=124; //Can be obtained from the data.
    return true; //TBD check correct loading
}
bool DataManager::dataToGPU(DeviceDataPXgICalc *d_pDataPXgI)
{
    bool retVal=false;
    if(malloc_iData(d_pDataPXgI))
    {
        cpuDataToGPU(d_pDataPXgI);
        retVal=true;
    }
    else
        cout << "Could not allocate data for input transferral" << endl;
    return retVal;
}
        
bool DataManager::malloc_iData(DeviceDataPXgICalc *d_pData)
{
    bool retVal=false;
    if(!cudaMalloc(&(d_pData->d_NFexpForI), sizeof(unsigned int)*n_prot))
        if(!cudaMalloc(&(d_pData->d_TopNFluExpId), sizeof(unsigned int)*n_reads*n_sparsity))
            if(!cudaMalloc(&(d_pData->d_FexpIdForI), sizeof(unsigned int)*InputData.FexpIdForI.size()))
                if(!cudaMalloc(&(d_pData->d_PFexpForI), sizeof(float)*InputData.FexpIdForI.size()))
                    if(!cudaMalloc(&(d_pData->d_TopNFluExpScores), sizeof(float)*n_reads*n_sparsity))
                        if(!cudaMalloc(&(d_pData->d_pRem), sizeof(float)*n_reads))
                            retVal=true;
    return retVal;
}
void DataManager::cpuDataToGPU(DeviceDataPXgICalc *d_pData)
{
    cudaMemcpy(d_pData->d_NFexpForI, InputData.NFexpForI.data(), sizeof(unsigned int)*n_prot, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pData->d_TopNFluExpId, InputData.TopNFluExpId.data(), sizeof(unsigned int)*n_reads*n_sparsity, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pData->d_FexpIdForI, InputData.FexpIdForI.data(), sizeof(unsigned int)*InputData.FexpIdForI.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pData->d_PFexpForI, InputData.PFexpForI.data(), sizeof(unsigned int)*n_prot, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pData->d_TopNFluExpScores, InputData.TopNFluExpScores.data(), sizeof(unsigned int)*InputData.TopNFluExpScores.size(), cudaMemcpyHostToDevice);
}
void DataManager::free(DeviceDataPXgICalc *d_pData)
{
    cudaFree( d_pData->d_NFexpForI );
    cudaFree( d_pData->d_TopNFluExpId );
    cudaFree( d_pData->d_FexpIdForI );
    cudaFree( d_pData->d_PFexpForI );
    cudaFree( d_pData->d_TopNFluExpScores );
    cudaFree( d_pData->d_pRem );
}
