#include "IOManager.h"

#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

using namespace std;

string operator/(string const& c1, string const& c2); //To concatenate folders is easier
template<typename T> bool readWholeArray(vector<T> &vect,string path);
bool fileExists(string filename);
template<typename T> void saveToBinFile(string path, vector<T> &cont);

IOManager::IOManager()
{
    dataFolder="/home/jkipen/ProtInfGPU/dev/OptKernel/data";
    outDataFolder="/home/jkipen/raid_storage/ProtInfGPU/data";
    oracle=false;
}

IOManager::~IOManager()
{
}

bool IOManager::init()
{
    bool retVal=true;
    retVal&=readWholeArray(datasetMetadata.fluExpIdForI,dataFolder / "fluExpIdForI.bin");
    retVal&=readWholeArray(datasetMetadata.nFluExpForI,dataFolder / "nFluExpForI.bin");
    retVal&=readWholeArray(datasetMetadata.probFluExpForI,dataFolder / "probFluExpForI.bin");
    
    datasetMetadata.nProt=datasetMetadata.nFluExpForI.size();
    auto max=max_element(datasetMetadata.fluExpIdForI.begin(), datasetMetadata.fluExpIdForI.end());
    datasetMetadata.nFluExp = *max + 1;
    datasetMetadata.nSparsity=1000; //Will change when it is not oracle!
    datasetMetadata.nReadsTotal=10000000; //Has to be a high enough value, will be changed after to generate the datasets!
    vector<float> aux(datasetMetadata.nProt,0);
    datasetMetadata.expNFluExpGenByI=aux;// fills with zeros.
    pIEsts=aux;
    
    return retVal;
}
void IOManager::setReadsMax(unsigned int nReads)
{
    datasetMetadata.nReadsTotal=nReads;
}

void IOManager::loadScores()
{
    if(oracle)
    {
        if(fileExists(dataFolder / "TopNScores.bin") && fileExists(dataFolder / "TopNScoresId.bin"))
        {
            readWholeArray(topNFluExpScores,dataFolder / "TopNScoresOracle.bin");
            readWholeArray(topNFluExpScoresIds,dataFolder / "TopNScoresIdOracle.bin");
        }
        else
            createOracleScores();
    }
    else
        if(fileExists(outDataFolder / "TopNScoresNonOracle.bin") && fileExists(outDataFolder / "TopNScoresIdNonOracle.bin"))
        {
            readWholeArray(topNFluExpScores,outDataFolder / "TopNScoresNonOracle.bin");
            readWholeArray(topNFluExpScoresIds,outDataFolder / "TopNScoresIdNonOracle.bin");
        }
        else
            createNonOracleScores();
}

void IOManager::createNonOracleScores()
{
    unsigned long sizeScores= ((unsigned long) datasetMetadata.nReadsTotal)*((unsigned long) datasetMetadata.nSparsity);
    topNFluExpScores.resize(sizeScores,0);
    srand(0); //Same seed so it always gen same random 
    topNFluExpScoresIds.resize(sizeScores,0);
    vector<unsigned int> aux;
    aux.resize(datasetMetadata.nSparsity,0);
    generate(topNFluExpScoresIds.begin(), topNFluExpScoresIds.end(), rand);
    
    //We want scores that decrease linearly, and sum equal to 1-sparsityleft. Doing the math we get the parameters of the curve y=y0-m x where x is read and y prob
    float Ns=datasetMetadata.nSparsity;//cast to float
    float sparsityLeft=0.01;
    float m = 2*(1-sparsityLeft)/(Ns*(Ns-1));
    float y0 = m * (Ns-1);
    
    unsigned long offset=51731; //Some random offest 
    for(unsigned long currReadIdx=0;currReadIdx<datasetMetadata.nReadsTotal;currReadIdx++)
    {
        for(unsigned long currReadScoreIdx=0;currReadScoreIdx<datasetMetadata.nSparsity;currReadScoreIdx++)
        {
            unsigned long absIdx= currReadIdx * datasetMetadata.nSparsity+currReadScoreIdx;
            topNFluExpScores[absIdx] = y0 - m * (float) currReadScoreIdx;
            topNFluExpScoresIds[absIdx] = (offset+absIdx)%datasetMetadata.nFluExp; //numbers lower than nfluexp, we make sure that there are not equal numbers in nspar.   
        }
        unsigned long idxStart=currReadIdx * datasetMetadata.nSparsity;
        sort(topNFluExpScoresIds.begin()+idxStart, topNFluExpScoresIds.begin()+idxStart+datasetMetadata.nSparsity);   
        
    }
    saveToBinFile(outDataFolder / "TopNScoresNonOracle.bin", topNFluExpScores);
    saveToBinFile(outDataFolder / "TopNScoresIdNonOracle.bin", topNFluExpScoresIds);
}

void IOManager::createOracleScores()
{
    topNFluExpScores.resize(datasetMetadata.nReadsTotal,0);
    srand(0); //Same seed so it always gen same random 
    topNFluExpScoresIds.resize(datasetMetadata.nReadsTotal,0);
    generate(topNFluExpScoresIds.begin(), topNFluExpScoresIds.end(), rand);
    for(unsigned int i=0;i<datasetMetadata.nReadsTotal;i++)
    {
        topNFluExpScores[i] = DEFAULT_SCORE_ORACLE; 
        topNFluExpScoresIds[i] = topNFluExpScoresIds[i]%datasetMetadata.nFluExp; //numbers lower than nfluexp.
    }
    
    saveToBinFile(dataFolder / "TopNScoresOracle.bin", topNFluExpScores);
    saveToBinFile(dataFolder / "TopNScoresIdOracle.bin", topNFluExpScoresIds);
}
void IOManager::saveTruePXgIrel(vector<float> &PXgIrel,string name)
{
    saveToBinFile(outDataFolder / name, PXgIrel);
}
//Auxiliar funcs;
template<typename T> void saveToBinFile(string path, vector<T> &cont)
{
    ofstream fout(path, ios::binary);
    fout.write((char*)&cont[0], cont.size() * sizeof(T));
    fout.close();
}
bool fileExists(string filename) {
    ifstream file(filename);
    return file.good();
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

string operator/(string const& c1, string const& c2)
{
    return c1 + "/" + c2;
}