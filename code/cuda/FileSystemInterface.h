#ifndef FILESYSTEMINTERFACE_H
#define FILESYSTEMINTERFACE_H

#include <iostream>
#include <string>
#include <fstream>

#include "DatasetMetadata.h"


using namespace std;

//#define GB_PARTIAL_SCORES_DEFAULT 4 //4Gb of the default usage to load scores from disk
#define GB_PARTIAL_SCORES_DEFAULT 16
//#define GB_PARTIAL_SCORES_DEFAULT 0.001 
#define N_DATASETS_CROSSVALIDATION_DEFAULT 10 //Number of cross validation runs by default 


class FileSystemInterface {  //Class to abstract from the used filesystem to load the datasets and its metadata
    
    public:
        FileSystemInterface(); 
        ~FileSystemInterface();
        void init(string classifierPath,bool oracle);
        void setPartialScoresSize(float nGygas); //Sets how many gygas are used to store partially the scores.
        void readPartialScores(); //Loads in the vectors the next part of the scores.
        void restartReading(); //Starts reading the scores from the beginning of the file.
        bool finishedReading; //Flag to indicate that finished reading the dataset. If it has finished, the read has to be reset to start fetching the scores again
    
        void saveCSV(string path, vector<string> &cols, vector<vector<string>> &content);//Saves csv results
        void saveTxt(string path,string msg); //Saves txt of results
        string createExperimentFolder(string pathToResult); //Creates a folder for the results!

        bool useSubsetCV,allScoresFitInMem; //Indicates if less samples are used in cv (way to reduce computing t ime).
        unsigned long sizeForPartialScores,bufferSize; //How much bytes dedicated to load reads from disk to RAM!
        unsigned long nScoresElementsInMemory,nReadsInMemory;
        unsigned int nReadsPartialScores; //How many reads are then stored
        unsigned int nCrossVal,nScoresTotal,nSubsetCV;
    
        

        float limitRAMGb;
        DatasetMetadata datasetMetadata;
        vector<unsigned int> trueIds; 
        vector<vector<unsigned int>> cvScoreIds; 
        vector<vector<float>> cvTrueProtDist; 
        vector<float> TopNScoresPartialFlattened;
        vector<unsigned int> TopNScoresIdsPartialFlattened;
        bool oracle; //When we are running an oracle mod, some things are not needed to be set/loaded
    private:
        unsigned long nReadsToG(unsigned long  nReads);
        bool loadDataset(); //Loads all the info of the dataset except for the scores. Returns 0 if the dataset did not load correctly
        void loadDatasetPaths(); //Loads the paths to the data within the folder.
        void resampleCVIdxs();
        void setRemainingMetadataFields(); //Once data is loaded, it gets the redundant metadata information.
        string datasetPath,classifierPath,commonPath,trueIdsPath,fluExpIdForIPath,nFluExpForIPath,
                    probFluExpForIPath,TopNScoresArrayPath,TopNScoresIdsArrayPath,nSparsityPath,expNFluExpGenByIPath;
        ifstream TopNScoresArrayStream,TopNScoresIdsArrayStream;
        vector<string> cvScoreIdsPath,cdTrueProtDistPath;
        //Configuration of how much can be stored in ram
        
    
        //Variables loaded when the dataset is found:
        

};


#endif