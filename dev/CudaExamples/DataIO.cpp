
#include "DataIO.h"
#include <fstream>
#include <sstream>

using namespace std;

void loadCSVToVector(vector<unsigned int> *vectorOut, string path)
{
    fstream dataCsv(path, ios::in);
	string auxLine;
    if (!dataCsv.is_open())
		cout << "File" << path << "not found!"<<endl;
	else 
	{
		while (getline(dataCsv, auxLine)) 
		{
			stringstream ss(auxLine);
			vector<string> substrings;
			while (ss.good()) { //Gets the 3 substrings for each line
				string valstr;
				getline(ss, valstr, ',');
                vectorOut->push_back(stoi(valstr));
			}
		}
		dataCsv.close();
	}
}

void loadCSVToVector(vector<float> *vectorOut, string path)
{
    fstream dataCsv(path, ios::in);
	string auxLine;
    if (!dataCsv.is_open())
		cout << "File" << path << "not found!"<<endl;
	else 
	{
		while (getline(dataCsv, auxLine)) 
		{
			stringstream ss(auxLine);
			vector<string> substrings;
			while (ss.good()) { //Gets the 3 substrings for each line
				string valstr;
				getline(ss, valstr, ',');
                vectorOut->push_back(stof(valstr));
			}
		}
		dataCsv.close();
	}
}

void loadExampleInput(InputDataPXgICalc &dataInput, string folder_path)
{
    vector<string> var_names = { "n_fe_per_I.csv", "p_fe_for_I_all.csv", "p_fe_for_I_iz_fe_all.csv","top_n_flu_iz.csv","top_n_flu_scores.csv" };
    loadCSVToVector(&dataInput.NFexpForI, folder_path+var_names[0]);
    loadCSVToVector(&dataInput.PFexpForI, folder_path+var_names[1]);
    loadCSVToVector(&dataInput.FexpIdForI, folder_path+var_names[2]);
    loadCSVToVector(&dataInput.TopNFluExpId, folder_path+var_names[3]);
    loadCSVToVector(&dataInput.TopNFluExpScores, folder_path+var_names[4]);
}

void saveVectorToCSV(vector<float> *vectorOut, string path)
{
    std::ofstream myFile(path);
    
    // Send data to the stream
    for(int i = 0; i < vectorOut->size(); i++)
        myFile << to_string((*vectorOut)[i]) << ",";
    
    // Close the file
    myFile.close();
}
