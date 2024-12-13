#ifndef INPUTPARSER_H
#define INPUTPARSER_H

#include <iostream>
#include <string>

using namespace std;


typedef enum
{
    KEY,
    DESCRIPTION,
    DEFAULT_VALUE,
    N_DESCRIPTORS
}   KeyDescriptor; //Each key will have stored

//Table of configs:

                // Key ,        Description      , Default value
/*    keyDescriptions={ {"-m",   "Memory limit on RAM" ,     "8"       },
                      {"-M",   "Memory limit on GPU" ,     "7"       },
                      {"-n",       "N sparsity"      ,     "30"      },
                      {"-o",       "Use oracle"      ,      "0"      }, //Uses oracle.value is the probability of error
                      {"-c","Cross validation datasets",   "10"      },
                      {"-e",     "Number of epochs"  ,     "60"      },
                      {"-v",         "Verbose"       ,     "No"      }, //No value, using the key will make the code verbose.
                      {"-t","Number of threads per block", "16"      }}*/


//Definition of the device variables for calculation
class InputParser{  //Class to handle the data movement (later would be also binding and data transfer to python).
    
    public:
        InputParser();
        ~InputParser();
        
        void init();
        bool parse(int argc, char** argv);
        unsigned int getKeyIdx(string description);
        void displayInfoTable(); //If there was an error in the parsing, here we plot the message of how it should be called.
    
        string inputDir,outputDir; //Directories.
        unsigned int nEpochs,nCrossValDs,nTreadsPerBlock,nSparsity;
        bool useOracle;
        float oraclePErr,limitRAMGb,limitMemGPUGb;
    
    private:
        bool parseOptWithValue(unsigned int *pKeyIndex,unsigned int argc, char** argv)
        vector<array<string,N_DESCRIPTORS>> keyDescriptions;     
    
};



#endif