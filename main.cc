//#include <stdlib.h>
//#include <stdio.h>
#include <tensorflow/c/c_api.h>

#include <iostream>
#include <vector>
#include <cassert>

//#include <stdint.h>
using namespace std;

// Ref 
// 1. https://stackoverflow.com/questions/42807435/what-is-the-equivalent-cpp-function-for-tf-train-import-meta-graph-in-tensorfl
// 2. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api_test.cc
int main() {
    // Load the saved model.
    const TF_SessionOptions* opt = TF_NewSessionOptions();
    const TF_Buffer* run_options = TF_NewBufferFromString("", 0);
    const char* export_dir = "./assets";
    const char* tags[1] = {"serve"};
    const int num_tags = 1;
    TF_Graph* graph = TF_NewGraph();
    TF_Buffer* metagraph = TF_NewBuffer();
    TF_Status* s = TF_NewStatus();
    TF_Session* session = TF_LoadSessionFromSavedModel(opt, 
                                                       run_options,
                                                       export_dir,
                                                       tags,
                                                       num_tags,
                                                       graph,
                                                       metagraph,
                                                       s);

    cout << TF_Message(s) << endl;
    
    TF_DeleteBuffer(metagraph);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);
    return 1;
}
