#include "Wrapper.h"
#include <string>
#include <iostream>



int main(int argc, char** argv) {
    Wrapper wrapper;
    wrapper.init();
    //wrapper.timeBaseKernel();
    wrapper.checkFewProtFewReadPerBlock();
    
    return 0;
}