#include "FileSystemInterface.h"
#include <fstream>
#include <algorithm>
#include <math.h>       /* pow */

string operator/(string const& c1, string const& c2);

bool readSingleUintBinary(unsigned int * pUint,string path) //reads a binary file into a template vector
{
    bool retVal=false;
    ifstream input( path, ios::binary );
    if(input)
    {
        input.seekg (0, input.end);
        streamsize size = input.tellg();
        input.seekg(0, input.beg);
        input.read((char *) pUint, size);
        retVal=true;
    }
    return retVal;
}


template<typename T> bool readWholeArray(vector<T> &vect,string path) //reads a binary file into a template vector
{
    bool retVal=false;
    ifstream input( path, ios::binary );
    if(input)
    {
        input.seekg (0, input.end);
        streamsize size = input.tellg();
        input.seekg(0, input.beg);
        vect.resize(size/sizeof(T));
        input.read((char *) vect.data(), size);
        retVal=true;
    }
    return retVal;
}


FileSystemInterface::FileSystemInterface(string classifierPath)
{
    this->classifierPath=classifierPath;
    this->datasetPath=classifierPath.substr(0,classifierPath.find_last_of("/\\"));
    commonPath= datasetPath / "Common";
    nCrossVal=N_DATASETS_CROSSVALIDATION_DEFAULT;
    finishedReading=false;
    nScoresElementsInMemory;nReadsInMemory=0; //No results in memory when initializing
    for(unsigned int i=0;i<nCrossVal;i++) //Fills vectors of vectors with empty vectors to be filled
    {
        vector<unsigned int> auxUint;vector<float> auxfloat;
        cvScoreIds.push_back(auxUint);
        cvTrueProtDist.push_back(auxfloat);
    }
    if(loadDataset())
    {
        setRemainingMetadataFields();
        setPartialScoresSize(GB_PARTIAL_SCORES_DEFAULT);
    }
    else
        cout << "There was a problem when loading the dataset" << endl;
}
void FileSystemInterface::setPartialScoresSize(float nGygas)
{
    sizeForPartialScores=nGygas*pow(2,30);
    unsigned long sizePerRead = datasetMetadata.nSparsity*(sizeof(float)+sizeof(unsigned int)); //Number of bytes that one score for reads occupies, considering probs and Ids!
    nReadsPartialScores=(unsigned int)(sizeForPartialScores/sizePerRead);
    bufferSize=((unsigned long)(datasetMetadata.nSparsity))*((unsigned long)(nReadsPartialScores)); //How much we elements should reserve on each vector
    TopNScoresPartialFlattened.resize(bufferSize);
    TopNScoresIdsPartialFlattened.resize(bufferSize);
}
void FileSystemInterface::readPartialScores()
{
    streamsize currG = TopNScoresIdsArrayStream.tellg();
    unsigned long nCharsToRead;
    
    if(currG+nReadsToG(nReadsPartialScores)<nReadsToG(nScoresTotal)) //If when reading nReadsPartialScores it wont finish the file:
        nCharsToRead=nReadsToG(nReadsPartialScores);
    else
        nCharsToRead=nReadsToG(nScoresTotal)-currG;
    
    nScoresElementsInMemory=nCharsToRead/sizeof(float);
    nReadsInMemory=nScoresElementsInMemory/datasetMetadata.nSparsity;
    TopNScoresArrayStream.read((char *) TopNScoresPartialFlattened.data(), nCharsToRead);
    TopNScoresIdsArrayStream.read((char *) TopNScoresIdsPartialFlattened.data(), nCharsToRead);
    
    if(TopNScoresIdsArrayStream.tellg()==nReadsToG(nScoresTotal))
        finishedReading=true;
    
}
void FileSystemInterface::restartReading()
{
    TopNScoresIdsArrayStream.seekg(0, TopNScoresIdsArrayStream.beg);
    TopNScoresArrayStream.seekg(0, TopNScoresArrayStream.beg);
    finishedReading=false;
}


bool FileSystemInterface::loadDataset()
{
    bool retVal=true;
    loadDatasetPaths();
    retVal&=readWholeArray(trueIds,trueIdsPath);
    readSingleUintBinary(&(datasetMetadata.nSparsity),nSparsityPath);
    retVal&=readWholeArray(datasetMetadata.fluExpIdForI,fluExpIdForIPath);
    retVal&=readWholeArray(datasetMetadata.nFluExpForI,nFluExpForIPath);
    retVal&=readWholeArray(datasetMetadata.probFluExpForI,probFluExpForIPath);
    for(unsigned int i=0;i<nCrossVal;i++)
    {
        retVal&=readWholeArray(cvScoreIds[i],cvScoreIdsPath[i]);
        retVal&=readWholeArray(cvTrueProtDist[i],cdTrueProtDistPath[i]);
    }
    TopNScoresArrayStream.open(TopNScoresArrayPath, ios::binary); //Opens the scores files, doesnt load them!
    TopNScoresIdsArrayStream.open(TopNScoresIdsArrayPath, ios::binary);
    TopNScoresArrayStream.seekg(0, TopNScoresArrayStream.beg);
    TopNScoresIdsArrayStream.seekg(0, TopNScoresIdsArrayStream.beg);
    return retVal;
}

void FileSystemInterface::loadDatasetPaths()
{
    trueIdsPath = commonPath / "trueIds.bin";
    nSparsityPath = commonPath / "nSparsity.bin";
    fluExpIdForIPath = commonPath / "fluExpIdForI.bin";
    nFluExpForIPath = commonPath / "nFluExpForI.bin";
    probFluExpForIPath = commonPath / "probFluExpForI.bin";
    TopNScoresArrayPath = classifierPath / "Common/TopNScores.bin";
    TopNScoresIdsArrayPath = classifierPath / "Common/TopNScoresId.bin";
    for(unsigned int i=0;i<nCrossVal;i++)
    {
        cvScoreIdsPath.push_back(classifierPath / ("CrossVal/ScoreIds"+to_string(i)+".bin"));
        cdTrueProtDistPath.push_back(classifierPath / ("CrossVal/TrueProtDist"+to_string(i)+".bin"));
    }
}
void FileSystemInterface::setRemainingMetadataFields()
{
    datasetMetadata.nProt=datasetMetadata.nFluExpForI.size();
    datasetMetadata.nReadsTotal=cvScoreIds[0].size();
    auto aux=max_element(datasetMetadata.fluExpIdForI.begin(), datasetMetadata.fluExpIdForI.end());
    datasetMetadata.nFluExp = *aux + 1;
    TopNScoresIdsArrayStream.seekg (0, TopNScoresIdsArrayStream.end);
    streamsize size = TopNScoresIdsArrayStream.tellg();
    TopNScoresIdsArrayStream.seekg(0, TopNScoresIdsArrayStream.beg);
    nScoresTotal = (size/sizeof(unsigned int))/datasetMetadata.nSparsity;
    
}

unsigned int FileSystemInterface::nReadsToG(unsigned int nReads)
{
    return nReads*datasetMetadata.nSparsity*sizeof(float); //Its the same as the uint32, but if we would use different types this would change...
}

FileSystemInterface::~FileSystemInterface()
{
    if(TopNScoresIdsArrayStream)
        TopNScoresIdsArrayStream.close();
    if(TopNScoresArrayStream)
        TopNScoresArrayStream.close();
}

string operator/(string const& c1, string const& c2)
{
    return c1 + "/" + c2;
}