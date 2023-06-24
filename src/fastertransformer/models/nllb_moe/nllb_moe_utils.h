#pragma once

#include <iostream>
#include <vector>

#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

namespace xiaohu_dbg {
template<typename T>
void PrintGPUArray(T* arr, int num_elements)
{
    std::vector<T> vec(num_elements);
    cudaD2Hcpy(vec.data(), arr, num_elements);
    std::cout << "{";
    for (int i = 0; i < num_elements; i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << "}" << std::flush;
}
}  // namespace xiaohu_dbg

}