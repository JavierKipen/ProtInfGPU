#include "Wrapper.h"

#include <cassert>
#include <chrono>
#include <math.h>


Wrapper::Wrapper()
{
    allocatedData=false;
}
Wrapper::~Wrapper()
{
    if(allocatedData)
        gDM.freeData(&devData, d_pdevData);
}
void Wrapper::init()
{
    
    IOM.init();
    gKM.init();
    unsigned long nBytesLim= MAX_GB_GPU *pow(2,30) ;
    IOM.setReadsMax(gDM.maxReadsToCompute(&(IOM.datasetMetadata),nBytesLim)); //Sets the number of reads to process!
    if(!gDM.allocateForNumberOfReads(&(IOM.datasetMetadata), &devData, &d_pdevData)) //Allocates data!
        assert(0 && "Could not allocate the space");
    IOM.loadScores();
    gDM.metadataToGPU(&(IOM.datasetMetadata),&devData, d_pdevData);
    gDM.loadNewDataToGPU(genPNewData(),&devData,d_pdevData);
    gKM.calcPRem(&devData,d_pdevData);
}

void Wrapper::timeBaseKernel()
{
    auto t1 = chrono::high_resolution_clock::now();
    gKM.runBaseKernel(&devData,d_pdevData);
    auto t2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_total = t2 - t1;
    cout << "Total time on normal kernel: " << ms_total.count() << " ms \n";
    unsigned long nElemMat=((unsigned long)IOM.datasetMetadata.nProt)*((unsigned long)IOM.datasetMetadata.nReadsTotal); //Number of elements of the result matrix
    PXgIrel.resize(nElemMat,0);
    gDM.retrieveOutput(PXgIrel.data(),&devData);
    
}

PNewData Wrapper::genPNewData()
{
    PNewData retVal;
    retVal.nReads=IOM.datasetMetadata.nReadsTotal;
    retVal.pTopNFluExpScores=IOM.topNFluExpScores.data();
    retVal.pPIEst=IOM.pIEsts.data();
    retVal.pTopNFluExpIds=IOM.topNFluExpScoresIds.data();
    return retVal;
}