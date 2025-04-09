#include "FileSystemInterface.h"
#include <fstream>
#include <algorithm>
#include <math.h>       /* pow */
#include <ostream>
#include <iomanip>  //For time to create the experiment folder
#include <ctime>
#include <sstream>

#include <iterator> //shuffle
#include <random>



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


FileSystemInterface::FileSystemInterface()
{
    nCrossVal=N_DATASETS_CROSSVALIDATION_DEFAULT;
    limitRAMGb=GB_PARTIAL_SCORES_DEFAULT;
    useSubsetCV=false;
    nSubsetCV=0;
    allScoresFitInMem=false;
}
void FileSystemInterface::init(string classifierPath, bool oracle)
{
    this->oracle=oracle;
    this->classifierPath=classifierPath;
    this->datasetPath=classifierPath.substr(0,classifierPath.find_last_of("/\\"));
    commonPath= datasetPath / "Common";
    
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
        if(useSubsetCV)
            resampleCVIdxs(); //Reduces to a subset the cv indexes 
        setRemainingMetadataFields();
        setPartialScoresSize(limitRAMGb);
    }
    else
        cout << "There was a problem when loading the dataset" << endl;
}
void FileSystemInterface::setPartialScoresSize(float nGygas)
{
    sizeForPartialScores=nGygas*pow(2,30);
    unsigned long sizePerRead = datasetMetadata.nSparsity*(sizeof(float)+sizeof(unsigned int)); //Number of bytes that one score for reads occupies, considering probs and Ids!
    nReadsPartialScores=(unsigned int)(sizeForPartialScores/sizePerRead);
    if(nReadsPartialScores>nScoresTotal)
    {
        nReadsPartialScores=nScoresTotal; //We dont need to store more than 
        allScoresFitInMem=true; //We can fit all the scores of the disk in our working memory, can reduce drastically runtime.
    }
    bufferSize=((unsigned long)(datasetMetadata.nSparsity))*((unsigned long)(nReadsPartialScores)); //How much we elements should reserve on each vector
    if(!oracle) //Opens score files.
    {
        TopNScoresPartialFlattened.resize(bufferSize);
        TopNScoresIdsPartialFlattened.resize(bufferSize);
    }
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

void FileSystemInterface::resampleCVIdxs()   
{
    vector<unsigned int> idxsToKeep,finalCvScoreIds; 
    finalCvScoreIds.reserve(cvScoreIds[0].size());idxsToKeep.reserve(cvScoreIds[0].size());//Reserves size
    for (unsigned int i = 0; i < cvScoreIds[0].size(); i++) 
        idxsToKeep.push_back(i);  //np.arange(cvScoreIds[i].size())
    
    random_device rd; //For shuffling
    mt19937 g(rd());
    
    
    
    for(int i = 0; i < nCrossVal; i++)
    {
        finalCvScoreIds.clear(); //Clears the final vector
        shuffle(idxsToKeep.begin(), idxsToKeep.end(), g); //Shuffles the range
        auto itBeg=idxsToKeep.begin();
        auto itEnd=idxsToKeep.begin()+nSubsetCV;
        vector<unsigned int> idxsToKeepSubset(itBeg,itEnd); //keeps nSubsetCV first items
        sort(idxsToKeepSubset.begin(), idxsToKeepSubset.end()); //Puts them in order, so the order remains!
        
        for(unsigned int j = 0; j < nSubsetCV; j++)
            finalCvScoreIds.push_back(cvScoreIds[i][idxsToKeepSubset[j]]); //gets those indexes of the bigger cvScoresId
            
        cvScoreIds[i]=finalCvScoreIds;
    }
    
}

void FileSystemInterface::saveCSV(string path, vector<string> &cols, vector<vector<string>> &content)
{
    ofstream outFile(path);
    
    //Set columns
    for(int i = 0; i < cols.size(); i++)
        outFile << cols[i] << ((i==cols.size()-1) ? "\n":",");
    // Send data to the stream
    for(int i = 0; i < content[0].size(); i++) //For each row
    {
        for(int j = 0; j < content.size(); j++) //for each column
            outFile << content[j][i] << ((j==content.size()-1) ? "\n":",");
    }
    
    // Close the file
    outFile.close();
    
}
void FileSystemInterface::saveTxt(string path,string msg)
{
    ofstream outFile(path);
    outFile << msg;
    outFile.close();
    
}

string FileSystemInterface::createExperimentFolder(string pathToResult)
{
    string expFolder;
    auto t = time(nullptr); //https://stackoverflow.com/questions/16357999/current-date-and-time-as-string
    auto tm = *localtime(&t);
    ostringstream oss;
    oss << put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    auto strTime = oss.str();
    expFolder=pathToResult+"/"+strTime; //Adds time to the folder
    //system("echo Going to create a folder!");
    string command="mkdir -p " + expFolder;
    system(command.c_str());
    //filesystem::create_directories(expFolder);
    return expFolder;
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
    retVal&=readWholeArray(datasetMetadata.expNFluExpGenByI,expNFluExpGenByIPath);
    
    for(unsigned int i=0;i<nCrossVal;i++)
    {
        retVal&=readWholeArray(cvScoreIds[i],cvScoreIdsPath[i]);
        retVal&=readWholeArray(cvTrueProtDist[i],cdTrueProtDistPath[i]);
    }
    
    if(!oracle) //Opens score files.
    {
        TopNScoresArrayStream.open(TopNScoresArrayPath, ios::binary); //Opens the scores files, doesnt load them!
        TopNScoresIdsArrayStream.open(TopNScoresIdsArrayPath, ios::binary);
        TopNScoresArrayStream.seekg(0, TopNScoresArrayStream.beg);
        TopNScoresIdsArrayStream.seekg(0, TopNScoresIdsArrayStream.beg);
    }

    for(unsigned int i=0;i<datasetMetadata.probFluExpForI.size();i++)
    {
        if(datasetMetadata.probFluExpForI[i]==0)
        {
            cout << "Zero found in probFluExpForI, this should not happen!" << endl;
        }
        /*if(datasetMetadata.fluExpIdForI[i]==2468)
        {
            cout << "Found fluexp 2468, with prob " << to_string(datasetMetadata.probFluExpForI[i]) << " In pos i= " << to_string(i) << endl;
        }*/
    }
        
    
    return retVal;
}

void FileSystemInterface::loadDatasetPaths()
{
    trueIdsPath = classifierPath / "Common/trueIds.bin";
    nSparsityPath = classifierPath / "Common/nSparsity.bin";
    fluExpIdForIPath = commonPath / "fluExpIdForI.bin";
    nFluExpForIPath = commonPath / "nFluExpForI.bin";
    probFluExpForIPath = commonPath / "probFluExpForI.bin";
    expNFluExpGenByIPath = commonPath / "expNFluExpGenByI.bin";
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

unsigned long FileSystemInterface::nReadsToG(unsigned long nReads)
{
    return nReads*((unsigned long)datasetMetadata.nSparsity)*sizeof(float); //Its the same as the uint32, but if we would use different types this would change...
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