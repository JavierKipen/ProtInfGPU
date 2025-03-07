#include "Wrapper.h"

#include <cassert>
#include <chrono>
#include <math.h>
#include <string>


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
    //unsigned long nElemMat=((unsigned long)IOM.datasetMetadata.nProt)*((unsigned long)IOM.datasetMetadata.nReadsTotal); //Number of elements of the result matrix
    //PXgIrel.resize(nElemMat,0);
    //gDM.retrieveOutput(PXgIrel.data(),&devData);
    //IOM.saveTruePXgIrel(PXgIrel);
}

void Wrapper::checkFewProtFewReadPerBlock()
{
    //gKM.initCublas();
    auto t1 = chrono::high_resolution_clock::now();
    gKM.setPRemContribution(&devData,d_pdevData);
    gKM.runFewProtFewReadPerBlockOracle(&devData,d_pdevData, 170, 50000);
    auto t2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_total = t2 - t1;
    cout << "Total time on normal kernel: " << ms_total.count() << " ms \n";
    
    
    unsigned long nRows=1000;
    unsigned long nElemMat=((unsigned long)IOM.datasetMetadata.nProt)*nRows; //Number of elements of the result matrix
    PXgIrel.resize(nElemMat,0);
    gDM.retrieveOutput(PXgIrel.data(),&devData,nRows);
    IOM.saveTruePXgIrel(PXgIrel,"PXgIrelToCheckMemTransf.bin");
}

void Wrapper::checkNansMat(vector<float> &mat)
{
    for(unsigned int i=0;i<mat.size();i++)
    {
        if(isnan(mat[i]) || isinf(mat[i]))
        {
            cout << "Element " + to_string(i) + " is not a number." << endl;
        }
    }
}
void Wrapper::checkRowNonZero(vector<float> &mat,unsigned long nProt,unsigned long nReads)
{
    bool retVal=false;
    for(unsigned long i=0;i<nReads;i++)
    {
        bool allReadZeros=true;
        for(unsigned long j=0;j<nProt;j++)
            if(mat[i*nProt+j]!=0)
                allReadZeros=false;
        if(allReadZeros)
        {
            cout << "Row number: " << to_string(i) << " had all probs eq to zero."<< endl;
            break;
        }     
    }
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