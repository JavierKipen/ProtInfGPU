#ifndef DATAIO_H
#define DATAIO_H

#include <iostream>
#include <vector>
#include <string>
using namespace std;

typedef struct {
    vector<unsigned int> NFexpForI,FexpIdForI,TopNFluExpId;
	vector<float> PFexpForI,TopNFluExpScores,TruePXgivIrel,PIEst;
} InputData; //Inputs that are needed to calculate an update step (True values are just for comparison in debug.)

void loadCSVToVector(vector<unsigned int> *vectorOut, string path);

void loadCSVToVector(vector<float> *vectorOut, string path);

void loadExampleInput(InputData &dataInput, string folder_path);

void saveVectorToCSV(vector<float> *vectorOut, string path);


#endif