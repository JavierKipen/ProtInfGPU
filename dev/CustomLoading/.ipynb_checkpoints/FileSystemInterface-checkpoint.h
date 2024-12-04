#ifndef FILESYSTEMINTERFACE_H
#define FILESYSTEMINTERFACE_H

#include <iostream>
#include <string>
#include <fstream>

#include "DatasetMetadata.h"


using namespace std;

#define GB_PARTIAL_SCORES_DEFAULT 0.001 //4Gb of the default usage to load scores from disk
#define N_DATASETS_CROSSVALIDATION_DEFAULT 10 //Number of cross validation runs by default 


class FileSystemInterface {  //Class to abstract from the used filesystem to load the datasets and its metadata
    
    public:
        FileSystemInterface(string classifierPath); //the path of the dataset and which classifier scores must be clarified (path to the dataset included in the classifier)
        ~FileSystemInterface();
        void setPartialScoresSize(float nGygas); //Sets how many gygas are used to store partially the scores.
        void readPartialScores(); //Loads in the vectors the next part of the scores.
        void restartReading(); //Starts reading the scores from the beginning of the file.
        bool finishedReading; //Flag to indicate that finished reading the dataset. If it has finished, the read has to be reset to start fetching the scores again

        unsigned long sizeForPartialScores,bufferSize; //How much bytes dedicated to load reads from disk to RAM!
        unsigned long nScoresElementsInMemory,nReadsInMemory;
        unsigned int nReadsPartialScores; //How many reads are then stored
        unsigned int nCrossVal,nScoresTotal;
        DatasetMetadata datasetMetadata;
        vector<unsigned int> trueIds; 
        vector<vector<unsigned int>> cvScoreIds; 
        vector<vector<float>> cvTrueProtDist; 
        vector<float> TopNScoresPartialFlattened;
        vector<unsigned int> TopNScoresIdsPartialFlattened;
    private:
        unsigned int nReadsToG(unsigned int nReads);
        bool loadDataset(); //Loads all the info of the dataset except for the scores. Returns 0 if the dataset did not load correctly
        void loadDatasetPaths(); //Loads the paths to the data within the folder.
        void setRemainingMetadataFields(); //Once data is loaded, it gets the redundant metadata information.
        string datasetPath,classifierPath,commonPath,trueIdsPath,fluExpIdForIPath,nFluExpForIPath,
                    probFluExpForIPath,TopNScoresArrayPath,TopNScoresIdsArrayPath,nSparsityPath;
        ifstream TopNScoresArrayStream,TopNScoresIdsArrayStream;
        vector<string> cvScoreIdsPath,cdTrueProtDistPath;
        //Configuration of how much can be stored in ram
        
    
        //Variables loaded when the dataset is found:
        

};


#endif