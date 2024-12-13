#include "InputParser.h"


using namespace std;

InputParser::InputParser()
{
    init();
}
    
InputParser::~InputParser()
{
    
}

bool InputParser::parse(int argc, char** argv)
{
    bool retVal=false;
    if(argc==1)
        retVal=true; //If no args, we use default folders (easy for debug and dev).
    if(argc>=3) // If args, needs at least the 2 folders.
    {
        inputDir=argv[1];
        outputDir=argv[2];
        retVal=true; //Assumes can be parsed until proben uncorrect
        for(unsigned int j=3;j<argc;j++)
        {
            string argString(argv[j]);
            if(keyExists(argString))
            {  
                if(argString=="-v") //Only command without value
                    verbose=true;
                else
                {    
                    retVal=parseOptWithValue(&j,argc,argv); //We check then the value for it
                    if (retVal==false)
                        break;
                }
            }
            else
                {retVal=false; break;} //Wrong key breaks the parsing.
            
        }
    }
    return retVal;
}

bool InputParser::parseOptWithValue(unsigned int *pKeyIndex,unsigned int argc, char** argv)
{
    bool retVal=false;
    string currKey(argv[*pKeyIndex]);
    if((*pKeyIndex+1<argc) && argv[*pKeyIndex+1][0]!='-') //If there was no value for this, or there is a key after a key, it breaks
    {
        string valueStr(argv[*pKeyIndex+1]);
        if(currKey=="-m")
            limitRAMGb=stof(valueStr);
        if(currKey=="-M")
            limitMemGPUGb=stof(valueStr);
        if(currKey=="-n")
            nSparsity=atoi(valueStr.c_str());
        if(currKey=="-o")
            {useOracle=true;oraclePErr=stof(valueStr);}
        if(currKey=="-c")
            nCrossValDs=atoi(valueStr.c_str());
        if(currKey=="-e")
            nEpochs=atoi(valueStr.c_str());
        if(currKey=="-t")
            nTreadsPerBlock=atoi(valueStr.c_str());
    }
}
bool InputParser::keyExists(string str)
{
    bool retVal=false;
    for(unsigned int i=0;i<keyDescriptions.size();i++)
    {
        if(keyDescriptions[i][0]==str)
            retVal=true;
    }
    return retVal;
}
void InputParser::init()
{
                     // Key ,        Description      , Default value
    vector<array<string,N_DESCRIPTORS>> aux({ {"-m",   "Memory limit on RAM" ,     "8"       },
                      {"-M",   "Memory limit on GPU" ,     "7"       },
                      {"-n",       "N sparsity"      ,     "30"      },
                      {"-o",       "Use oracle"      ,      "0"      }, //Uses oracle. value  is the probability of error
                      {"-c","Cross validation datasets",   "10"      },
                      {"-e",     "Number of epochs"  ,     "60"      },
                      {"-v",         "Verbose"       ,     "No"      }, //No value, using the key will make the code verbose.
                      {"-t","Number of threads per block", "16"      }});
    
    keyDescriptions=aux;
    inputDir = DEFAULT_INPUT_PATH;
    outputDir = DEFAULT_OUTPUT_PATH;
    
    //Default values from table
    nEpochs=atoi(keyDescriptions[getKeyIdx("Number of epochs")][DEFAULT_VALUE].c_str());
    nTreadsPerBlock=atoi(keyDescriptions[getKeyIdx("Number of threads per block")][DEFAULT_VALUE].c_str());
    nCrossValDs=atoi(keyDescriptions[getKeyIdx("Cross validation datasets")][DEFAULT_VALUE].c_str());
    nSparsity=atoi(keyDescriptions[getKeyIdx("N sparsity")][DEFAULT_VALUE].c_str());
    limitRAMGb=stof(keyDescriptions[getKeyIdx("Memory limit on RAM")][DEFAULT_VALUE]);
    limitMemGPUGb=stof(keyDescriptions[getKeyIdx("Memory limit on GPU")][DEFAULT_VALUE]);
    useOracle=false;
    verbose=false;
    oraclePErr=0;
}

unsigned int InputParser::getKeyIdx(string definition)
{
    unsigned int retVal=0;
    for(unsigned int i=0;i<keyDescriptions.size();i++)
    {
        if(keyDescriptions[i][1]==definition)
            retVal=i;
    }
    return retVal;
}

void InputParser::displayInfoMsg()
{
    cout << "There was an error in the parsing of the arguments, try again!" <<endl; //Should be a more descriptive msg
}