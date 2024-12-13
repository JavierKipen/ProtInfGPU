#include "CrossValWrapper.h"
#include <math.h>
#include <numeric> //std::accumulate
#include <algorithm> //std::for_each 
#include <chrono>

CrossValWrapper::CrossValWrapper()
{
    
}
CrossValWrapper::~CrossValWrapper()
{
    
}
void CrossValWrapper::initDefault(string outFolder,string inFolder)
{
    outFolderPath=outFolder;
    FSI.init(inFolder);
    experimentFolder=FSI.createExperimentFolder(outFolder);
    idToCvScoreIdsEnd.resize(FSI.nCrossVal,0);idToCvScoreIdsStart.resize(FSI.nCrossVal,0);
    //CM.setMetadata(); //Data To configure the processing
    nSparsityRed=FSI.datasetMetadata.nSparsity; //This line would change if we want to set a lower sparsity than the dataset (useful to compare performances).
    
    gW.init(&(FSI.datasetMetadata));
    setGPUMemLimit(GB_GPU_USAGE_DEFAULT);
    nEpochs=10;
    ErrMean.resize(nEpochs,0);ErrStd.resize(nEpochs,0);
     //CM.memAlloc(maxReadsToProcessInGpu); //Allocates the memory to process this amount of reads.
    topNFluExpScores.resize(maxReadsToProcessInGpu*nSparsityRed,0);topNFluExpScoresIds.resize(maxReadsToProcessInGpu*nSparsityRed,0);
    
    float norm=0;
    for(unsigned int j=0;j<FSI.datasetMetadata.nProt;j++)
        norm+=FSI.datasetMetadata.expNFluExpGenByI[j];
    vector<float> auxPI(FSI.datasetMetadata.nProt,0);
    for(unsigned int j=0;j<FSI.datasetMetadata.nProt;j++)
        auxPI[j]=FSI.datasetMetadata.expNFluExpGenByI[j]/norm;
    for(unsigned int i=0;i<FSI.nCrossVal;i++)
    {   
        vector<float> aux(FSI.datasetMetadata.nProt,0);
        updates.push_back(aux);
        pIEsts.push_back(auxPI); //equally likely proteins assumption!
    }
    
}

void CrossValWrapper::init() //This init assumes that some variables have already been set (input and output path, nsparsity,nCrossVal, nepochs
{
    FSI.init(inputFolderPath);
    experimentFolder=FSI.createExperimentFolder(outFolderPath);
    idToCvScoreIdsEnd.resize(FSI.nCrossVal,0);idToCvScoreIdsStart.resize(FSI.nCrossVal,0);
    //CM.setMetadata(); //Data To configure the processing
    if(nSparsityRed>FSI.datasetMetadata.nSparsity) 
        cout << "N Sparsity is bigger than dataset sparsity!!" << endl; //This shouldnt happen, might break the code.
    
    gW.init(&(FSI.datasetMetadata));
    setGPUMemLimit(limitMemGPUGb);
    ErrMean.resize(nEpochs,0);ErrStd.resize(nEpochs,0);
     //CM.memAlloc(maxReadsToProcessInGpu); //Allocates the memory to process this amount of reads.
    topNFluExpScores.resize(maxReadsToProcessInGpu*nSparsityRed,0);topNFluExpScoresIds.resize(maxReadsToProcessInGpu*nSparsityRed,0);
    
    float norm=0;
    for(unsigned int j=0;j<FSI.datasetMetadata.nProt;j++)
        norm+=FSI.datasetMetadata.expNFluExpGenByI[j];
    vector<float> auxPI(FSI.datasetMetadata.nProt,0);
    for(unsigned int j=0;j<FSI.datasetMetadata.nProt;j++)
        auxPI[j]=FSI.datasetMetadata.expNFluExpGenByI[j]/norm;
    for(unsigned int i=0;i<FSI.nCrossVal;i++)
    {   
        vector<float> aux(FSI.datasetMetadata.nProt,0);
        updates.push_back(aux);
        pIEsts.push_back(auxPI); //equally likely proteins assumption!
    }
    
}
void CrossValWrapper::setGPUMemLimit(float nGb)
{
    nBytesToUseGPU=nGb*pow(2,30);
    maxReadsToProcessInGpu = gW.maxReadsToCompute(nBytesToUseGPU); //Given the bytes to use in the GPU, returns how many reads could be computed at the time
    unsigned int cvReads = FSI.cvScoreIds[0].size();
    maxReadsToProcessInGpu = (maxReadsToProcessInGpu>cvReads)? cvReads:maxReadsToProcessInGpu; //If we can fit all data, then we dont allocate more.
    gW.allocateWorkingMemory(maxReadsToProcessInGpu); //Allocates memory and sends the metadata that wont change between reads
}

void CrossValWrapper::computeEMCrossVal()
{
    auto t1 = chrono::high_resolution_clock::now();
    for(unsigned int i=0;i<nEpochs;i++)
    {
        computeEMCrossValEpoch(); //Gets the updates values looping through one read of the whole dataset with all crossval picks
        updatePIs(); //Uses the update weights to obtain new P(I) estimates
        calcError(i); //After having the new PIEsts, the error is calculated and stored
    }
    auto t2 = chrono::high_resolution_clock::now();
    auto int_s = std::chrono::duration_cast<chrono::seconds>(t2 - t1);
    timeProcessing = int_s.count();
    exportResults();
}
void CrossValWrapper::computeEMCrossValEpoch()
{
    vector<vector<unsigned int>> emptyVecOfVec; //To reset our variable
    nReadsOffset=0; //Keeps track of the number of reads got before
    while(!FSI.finishedReading) //The dataset may not fit in RAM, so we load batches!
    {
        
        FSI.readPartialScores(); //Loads batch of scores from disk
        getValidCVScoresIds(); //Selects the end IDs of the scores so we use scores that are in RAM
        for(unsigned int i=0;i<FSI.nCrossVal;i++) //For every cross validation dataset
        {
            for(auto& vec : cvScoreIdsVecOfVec)
                vec.clear();
            cvScoreIdsVecOfVec.clear(); //We clean the vect of vectors before using it!
            partitionDataset(i);//The cvScoreIds are partitioned in cvScoreIdsVecOfVec so each compute can be done easily
            for(unsigned int j=0;j<cvScoreIdsVecOfVec.size();j++)
                computeUpdateSubSet(i,j);
        }
        idToCvScoreIdsStart=idToCvScoreIdsEnd; // for the next data, we know that will start after what we had
        for(auto& itr:idToCvScoreIdsStart)
            itr=itr+1; //But we add one so we dont take the same sample twice! Not sure of this.
        nReadsOffset += FSI.nReadsInMemory; //Saves the amount of reads that have already been visited.
    }
    FSI.restartReading();
    for(auto& itr:idToCvScoreIdsStart)
        itr=0; //Restart idxs of start
}

void CrossValWrapper::getValidCVScoresIds()
{
    for(unsigned int i=0;i<FSI.nCrossVal;i++) //For every cross validation dataset
    {
        unsigned int j=0;
        bool found=false;
        for(j=idToCvScoreIdsStart[i];j<FSI.cvScoreIds[i].size();j++)
        {
            if((FSI.cvScoreIds[i][j])>=nReadsOffset+FSI.nReadsInMemory) //The reads in ram are [nReadsOffset,nReadsOffset+FSI->nReadsInMemory)
            {
                found=true;
                break;
            }
        }
            
        idToCvScoreIdsEnd[i]=(found)?(j-1):j; //when using break it still advances j. 
    }
}

void CrossValWrapper::partitionDataset(unsigned int cvIndex)
{
    unsigned int nScores=idToCvScoreIdsEnd[cvIndex]-idToCvScoreIdsStart[cvIndex];//This is the total amount of scores we are going to process of this cross validation run
    auto itr=FSI.cvScoreIds[cvIndex].begin()+idToCvScoreIdsStart[cvIndex];
    auto itr_end=FSI.cvScoreIds[cvIndex].begin()+idToCvScoreIdsEnd[cvIndex];
    while(itr != itr_end)
    {
        auto beg=itr;
        auto end=itr_end; //by default, it would be the end of the scores we are considering
        if(distance(itr, itr_end)>maxReadsToProcessInGpu) //If the vector is bigger than the reads we can process at a time in GPU, we partition it!.
            end=itr+maxReadsToProcessInGpu; 
        
        vector<unsigned int> auxVec(beg, end);
        for (auto& it : auxVec) 
            it = it - nReadsOffset; //Conversion from pointer to whole dataset TO pointer in the RAM memory.
        cvScoreIdsVecOfVec.push_back(auxVec);
        itr=end; //the next batch starts where the previous started.
    }
}
    
void CrossValWrapper::computeUpdateSubSet(unsigned int cvIndex,unsigned int subSetIdx)
{
    loadScoresInBuffer(cvScoreIdsVecOfVec[subSetIdx]); //Puts certain scores on the buffer of this class
    gW.accumulateUpdates(updates[cvIndex].data(), genPNewData(cvIndex, cvScoreIdsVecOfVec[subSetIdx].size())); //Calls the update calculation on the GPU, and sums the updates obtained.
}

PNewData CrossValWrapper::genPNewData(unsigned int cvIndex, unsigned int nReadsToCompute)
{
    PNewData retVal;
    retVal.nReads=nReadsToCompute;
    retVal.pTopNFluExpScores=topNFluExpScores.data();
    retVal.pPIEst=pIEsts[cvIndex].data();
    retVal.pTopNFluExpIds=topNFluExpScoresIds.data();
    return retVal;
}

void CrossValWrapper::loadScoresInBuffer(vector<unsigned int> &IdxsCv)
{//The idxs refer to which point in the buffer from pFSI will be selected: NOTE that they need to be normalized if the dataset enters in RAM memory
    unsigned int len=IdxsCv.size(); //Has to be lower or eq than the buffer length!
    for(unsigned int i=0;i<len;i++)
    {
        unsigned int currId=IdxsCv[i]; //Score Id that will be copied
        for(unsigned int j=0;j<nSparsityRed;j++) //We copy nSparsityRed from the 
        {
            unsigned long idxInPFSI=currId*FSI.datasetMetadata.nSparsity+j; //In pFSI the array has metadata.nSparsity columns, while the array to send to gpu has this->nSparsityRed columns
            unsigned long idxInBuffer=i*nSparsityRed+j;
            if (idxInPFSI>FSI.TopNScoresPartialFlattened.size()) //This 4 lines should be commented when it works
                cout << "ERR1" << endl;
            if (idxInBuffer>topNFluExpScores.size())
                cout << "ERR2" << endl;
            topNFluExpScores[idxInBuffer] = FSI.TopNScoresPartialFlattened[idxInPFSI];
            topNFluExpScoresIds[idxInBuffer] = FSI.TopNScoresIdsPartialFlattened[idxInPFSI];
        }
    }
}
void CrossValWrapper::updatePIs()
{
    unsigned int nProt=FSI.datasetMetadata.nProt;
    unsigned int nDatasets=FSI.nCrossVal;
    for(unsigned int i=0;i<nDatasets;i++)
    {
        float normVal=0;
        for(unsigned int j=0;j<nProt;j++)
            normVal+=updates[i][j];
        for(unsigned int j=0;j<nProt;j++)
            pIEsts[i][j]=updates[i][j]/normVal;
        for(unsigned int j=0;j<nProt;j++)
            updates[i][j]=0;
    }
    
}
void CrossValWrapper::exportResults()
{
    vector<string> cols({"Epoch","Error mean", "Error std"});
    vector<string> epochs,errMeanStr,errStdStr;
    for(unsigned int i=0;i<nEpochs;i++)
    {
        epochs.push_back(to_string(i+1));
        errMeanStr.push_back(to_string(ErrMean[i]));
        errStdStr.push_back(to_string(ErrStd[i]));
    }
    vector<vector<string>> outVector;
    outVector.push_back(epochs);outVector.push_back(errMeanStr);outVector.push_back(errStdStr);
    FSI.saveCSV(experimentFolder+"/ErrVsEpochs.csv", cols, outVector);
    FSI.saveTxt(experimentFolder+"/RunConfig.txt",genRunConfigMsg());
}
string CrossValWrapper::genRunConfigMsg()
{
    string out;
    
    out = "Nepochs: " + to_string(nEpochs) + "\n";
    out+= "Runtime: "+ to_string(timeProcessing) + "\n";
    out+= "NSparsity: " + to_string(nSparsityRed) + "\n";
    out+= "NThreadsPerBlock: "+ to_string(gW.gCM.NThreadsPerBlock) + "\n";
    out+= "NBytesUseGPU: "+ to_string(nBytesToUseGPU) + "\n";
    out+= "MaxReadsGPU: "+ to_string(maxReadsToProcessInGpu) + "\n";
    out+= "NcrossVal: "+ to_string(FSI.nCrossVal) + "\n";
    out+= "NtotalReads: "+ to_string(FSI.cvScoreIds[0].size()) + "\n";
    out+= "Nprot: "+ to_string(FSI.datasetMetadata.nProt) + "\n";
    out+= "NReadsInRam: "+ to_string(FSI.nReadsPartialScores) + "\n";    
    
    return out;
}
void CrossValWrapper::calcError(unsigned int epoch)
{
    
    vector<float> auxPY(FSI.datasetMetadata.nProt,0); //Here we will store the estimated PY for each PI
    vector<float> MAE(FSI.nCrossVal,0); //Here we will store the error for each cross val dataset.
    for(unsigned int i=0;i<FSI.nCrossVal;i++)   //For every crossval run!
    {
        float normVal=0;
        float absAccErr=0;
        for(unsigned int j=0;j<FSI.datasetMetadata.nProt;j++)
        {
            auxPY[j]= pIEsts[i][j] / FSI.datasetMetadata.expNFluExpGenByI[j]; //Weights of each protein
            normVal+=auxPY[j];
        }
        for(unsigned int j=0;j<FSI.datasetMetadata.nProt;j++)
            auxPY[j]/=normVal;
        for(unsigned int j=0;j<FSI.datasetMetadata.nProt;j++)
            absAccErr += std::abs(auxPY[j]-FSI.cvTrueProtDist[i][j]);
        MAE[i] = absAccErr / FSI.datasetMetadata.nProt; //The MAE of the run will be the accumulated absolute error, divided by the number of proteins
    }
    
    double sum = std::accumulate(MAE.begin(), MAE.end(), 0.0); //https://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos
    double mean = sum / MAE.size();

    double accum = 0.0;
    std::for_each (std::begin(MAE), std::end(MAE), [&](const double d) {
    accum += ((d - mean) * (d - mean));
    });

    double stdev = sqrt(accum / (MAE.size()-1));
    ErrMean[epoch]=mean;
    ErrStd[epoch]=stdev;
    
}


void CrossValWrapper::setNSparsity(unsigned int nSparsityRed)
{
    if(nSparsityRed<FSI.datasetMetadata.nSparsity)
        this->nSparsityRed=nSparsityRed;
    else
        cout << "Sparsity specified cannot be lower than the datasets one!" << endl;
}
