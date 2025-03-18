#ifndef IOMANAGER_H
#define IOMANAGER_H

#include <iostream>
#include <string>
#include <fstream>
#include "DatasetMetadata.h"

#define DEFAULT_SCORE_ORACLE 0.99

using namespace std;



class IOManager {  //Class to abstract from the used filesystem to load the datasets and its metadata
    
    public:
        IOManager(); 
        ~IOManager();
        bool init();
        void setReadsMax(unsigned int nReads);
        void loadScores();
        void createOracleScores();
        void createNonOracleScores();
        void saveTruePXgIrel(vector<float> &PXgIrel,string name);
        DatasetMetadata datasetMetadata;
    
        vector<float> topNFluExpScores,pIEsts; //Sparse vectors representation of the scores to compute
        vector<unsigned int> topNFluExpScoresIds; 
        bool oracle;
    private:
        
        string dataFolder;
        string outDataFolder;
    
};


#endif