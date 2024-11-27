#include "DataManager.h"
#include <vector>
#include <algorithm>    // std::max

DataManager::DataManager()
{
}

bool DataManager::loadDataFromCSV(string folder_path)
{
    loadExampleInput(inputData, folder_path);
    n_prot=inputData.NFexpForI.size();
    n_sparsity=30; //Hardcoded examples are N=30
    n_reads=inputData.TopNFluExpId.size()/n_sparsity;
    n_flu_exp=124; //Can be obtained from the data.
    n_reads_max=1000; 
    onesVecLen=max({n_sparsity,n_reads_max,n_prot});
    return true; //TBD check correct loading
}
bool DataManager::dataToGPU(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc **d_ppdevData)
{
    bool retVal=false;
    if(malloc_iData(pdevData,d_ppdevData))
    {
        cpuDataToGPU(pdevData,*d_ppdevData);
        retVal=true;
    }
    else
        cout << "Could not allocate data for input transferral" << endl;
    return retVal;
}
        
bool DataManager::malloc_iData(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc **d_ppdevData)
{
    bool retVal=false;
    if(!cudaMalloc(&(pdevData->d_NFexpForI), sizeof(unsigned int)*n_prot))
        if(!cudaMalloc(&(pdevData->d_TopNFluExpId), sizeof(unsigned int)*n_reads*n_sparsity))
            if(!cudaMalloc(&(pdevData->d_FexpIdForI), sizeof(unsigned int)*inputData.FexpIdForI.size()))
                if(!cudaMalloc(&(pdevData->d_PFexpForI), sizeof(float)*inputData.FexpIdForI.size()))
                    if(!cudaMalloc(&(pdevData->d_TopNFluExpScores), sizeof(float)*n_reads*n_sparsity))
                        if(!cudaMalloc(&(pdevData->d_pRem), sizeof(float)*n_reads))
                            if(!cudaMalloc(&(pdevData->d_MatAux), sizeof(float)*n_reads*n_prot))
                                if(!cudaMalloc(&(pdevData->d_ones), sizeof(float)*onesVecLen))
                                    if(!cudaMalloc(d_ppdevData, sizeof(*pdevData)))
                                        if(!cudaMalloc(&(pdevData->d_PIEst), sizeof(float)*n_prot))
                                            if(!cudaMalloc(&(pdevData->d_VecAux), sizeof(float)*onesVecLen))
                                                retVal=true;
    return retVal;
}
void DataManager::cpuDataToGPU(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc *d_pdevData)
{
    //Sets in host pointers of dev data and other values needed for computation
    cudaMemcpy(pdevData->d_NFexpForI, inputData.NFexpForI.data(), sizeof(unsigned int)*n_prot, cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_TopNFluExpId, inputData.TopNFluExpId.data(), sizeof(unsigned int)*n_reads*n_sparsity, cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_FexpIdForI, inputData.FexpIdForI.data(), sizeof(unsigned int)*inputData.FexpIdForI.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_PFexpForI, inputData.PFexpForI.data(), sizeof(float)*inputData.FexpIdForI.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_TopNFluExpScores, inputData.TopNFluExpScores.data(), sizeof(unsigned int)*inputData.TopNFluExpScores.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_PIEst, inputData.PIEst.data(), sizeof(float)*n_prot, cudaMemcpyHostToDevice);
    createOnesVec(pdevData);
    pdevData->n_flu_exp=n_flu_exp;pdevData->n_sparsity=n_sparsity;pdevData->n_reads=n_reads;pdevData->n_prot=n_prot;
    //Copies devData information to device!
    unsigned int size_data=sizeof(*pdevData);
    cudaMemcpy(d_pdevData, pdevData, size_data, cudaMemcpyHostToDevice);
}
void DataManager::createOnesVec(DeviceDataPXgICalc *pdevData)
{
    vector<float> ones(onesVecLen, 1);
    cudaMemcpy(pdevData->d_ones, ones.data(), sizeof(float)*onesVecLen, cudaMemcpyHostToDevice);
}
void DataManager::freeData(DeviceDataPXgICalc *pdevData, DeviceDataPXgICalc *d_pdevData)
{
    cudaFree( pdevData->d_NFexpForI );
    cudaFree( pdevData->d_TopNFluExpId );
    cudaFree( pdevData->d_FexpIdForI );
    cudaFree( pdevData->d_PFexpForI );
    cudaFree( pdevData->d_TopNFluExpScores );
    cudaFree( pdevData->d_pRem );
    cudaFree( pdevData->d_MatAux );
    cudaFree( pdevData->d_ones );
    cudaFree( pdevData->d_PIEst );
    cudaFree( pdevData->d_VecAux );
    cudaFree( d_pdevData );
}

DataManager::~DataManager()
{
}
