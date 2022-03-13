#ifndef IMAGEHELPER
#define IMAGEHELPER


#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

#define imageVec std::vector<std::vector<std::vector<float>>>
class ImageHelper {
    private: 
        std::string input_path;
        
        ImageHelper(std::string input_path) {
            this -> input_path = input_path;
        }

        void parseToTeacherData(std::vector<imageVec> &learning_data, std::vector<std::vector<float>> &learning_labels, std::vector<imageVec> &validation_data, std::vector<std::vector<float>> &validation_labels) {
            
        }
};

#endif