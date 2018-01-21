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

    cout << TF_GetCode(s) << " != TF_OK(" << TF_OK << ")" << endl;
    cout << TF_Message(s) << endl;
    //cout << "loaded the saved model" << endl;
    return 1;

    // Input Images
    const int kBatchSize = 1;
    const int kImageSize = 200;
    const int kImageChannels = 3;
    const long images_dims[] = {kBatchSize, kImageSize, kImageSize, kImageChannels};
    const int images_num_dims = 4;
    const size_t images_data_size = kBatchSize*kImageSize*kImageSize*kImageChannels*sizeof(float);
    float* images_data = new float[images_data_size];
    TF_Tensor* images_tensor = TF_NewTensor(TF_FLOAT, 
                                            images_dims,
                                            images_num_dims,
                                            images_data,
                                            images_data_size,
                                            NULL, NULL);
    cout << "made input images" << endl;
    return -1;
    
    // Input Labels
    const int kSeqLength = 10;
    const long labels_dims[] = {kBatchSize, kSeqLength};
    const int labels_num_dims = 2;
    const size_t labels_data_size = kBatchSize*kSeqLength*sizeof(long);
    int64_t* labels_data = new int64_t[labels_data_size];
    TF_Tensor* labels_tensor = TF_NewTensor(TF_INT64, 
                                            labels_dims,
                                            labels_num_dims,
                                            labels_data,
                                            labels_data_size,
                                            NULL,
                                            NULL);

  
    cout << "hello world" << endl; 
    return -1;
    // Output
    struct TF_Output * output;
    
    // Session Run
    vector<TF_Output> inputs;
    vector<TF_Tensor*> input_values;
    input_values.push_back(images_tensor); 
    input_values.push_back(labels_tensor); 

    vector<TF_Output> outputs;
    TF_Operation* output_op = TF_GraphOperationByName(graph, "import/predict");
    outputs.push_back({output_op, 0});
    vector<TF_Tensor*> output_values(outputs.size(), NULL);


    TF_Operation* target_opers = NULL;
    int num_targets = 0;
    TF_Buffer* handle = NULL;

    TF_SessionRun(session, run_options,
                  &inputs[0], &input_values[0], inputs.size(),
                  &outputs[0], &output_values[0], outputs.size(),
                  &target_opers, num_targets,
                  handle, s);


    void* output_data = TF_TensorData(output_values[0]);
    assert(TF_GetCode(s) == TF_OK);

    for (int i = 0; i < inputs.size(); ++i) TF_DeleteTensor(input_values[i]);
    for (int i = 0; i < outputs.size(); ++i) TF_DeleteTensor(output_values[i]);

    //TF_CloseSession(session, s); 
    //TF_DeleteSession(session, s);
    TF_DeleteBuffer(metagraph);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);
    return 1;
}
