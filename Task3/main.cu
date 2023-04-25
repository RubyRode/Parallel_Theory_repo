#include <iostream>
#include <cmath>
#include <cstring>

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

class parser{
public:
    parser(int argc, char** argv){
        this->_grid_size = 512;
        this->_accur = 1e-6;
        this->_iters = 1000000;
        for (int i=0; i<argc-1; i++){
            std::string arg = argv[i];
            if (arg == "-accur"){
                std::string dump = std::string(argv[i+1]);
                this->_accur = std::stod(dump);
            }else if (arg == "-a"){
                this->_grid_size = std::stoi(argv[i + 1]);
            }else if (arg == "-i"){
                this->_iters = std::stoi(argv[i + 1]);
            }
        }

    };
    __host__ double accuracy() const{
        return this->_accur;
    }
    __host__ int iterations() const{
        return this->_iters;
    }
    __host__ int grid()const{
        return this->_grid_size;
    }
private:
    double _accur;
    int _grid_size;
    int _iters;

};

double corners[4] = {10, 20, 30, 20};

__global__
void cross_calc(double* A_kernel, double* B_kernel, size_t size){
    // get the block and thread indices
    
    size_t j = blockIdx.x;
    size_t i = threadIdx.x;
    // main cross computation. the average of 4 incident cells is taken
    if (i != 0 && j != 0){
       
        B_kernel[j * size + i] = 0.25 * (
            A_kernel[j * size + i - 1] + 
            A_kernel[j * size + i + 1] + 
            A_kernel[(j + 1) * size + i] + 
            A_kernel[(j - 1) * size + i]
        );
    
    }

}

__global__
void get_error_matrix(double* A_kernel, double* B_kernel, double* out){
    // get thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // take the difference between B_kernel and A_kernel
    if (blockIdx.x != 0 && threadIdx.x != 0){
        
        out[idx] = std::abs(B_kernel[idx] - A_kernel[idx]);
    
    }

}


int main(int argc, char ** argv){
    parser input = parser(argc, argv);

    int size = input.grid();
    double min_error = input.accuracy();
    int max_iter = input.iterations();
    int full_size = size * size;
    double step = (corners[1] - corners[0]) / (size - 1);
    // Matrixes initialization
    auto* A_kernel = new double[size * size];
    auto* B_kernel = new double[size * size];

    std::memset(A_kernel, 0, sizeof(double) * size * size);


    A_kernel[0] = corners[0];
    A_kernel[size - 1] = corners[1];
    A_kernel[size * size - 1] = corners[2];
    A_kernel[size * (size - 1)] = corners[3];



    for (int i = 1; i < size - 1; i ++) {
        A_kernel[i] = corners[0] + i * step;
        A_kernel[size * i] = corners[0] + i * step;
        A_kernel[(size-1) + size * i] = corners[1] + i * step;
        A_kernel[size * (size-1) + i] = corners[3] + i * step;
    }

    std::memcpy(B_kernel, A_kernel, sizeof(double) * full_size);
    // matrix output before computations
    // for (int i = 0; i < size; i ++) {
    //     for (int j = 0; j < size; j ++) {
    //         std::cout << A_kernel[j * size + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    
    // Choosing the device
    cudaSetDevice(3);
    
    double* dev_A, *dev_B, *dev_err, *dev_err_mat, *temp_stor = NULL;
    size_t tmp_stor_size = 0;
    // Memory allocation for 2 matrixes and error variable on the device 
    cudaError_t status_A = cudaMalloc(&dev_A, sizeof(double) * full_size);
    cudaError_t status_B = cudaMalloc(&dev_B, sizeof(double) * full_size);
    cudaError_t status = cudaMalloc(&dev_err, sizeof(double));
    // some memory allocation accertions to catch errors
    if (status != cudaSuccess){
        std::cout << "Device error variable allocation error " << status << std::endl;
        return status;
    }
    // memory allocation on device for error matrix
    status = cudaMalloc(&dev_err_mat, sizeof(double) * full_size);
    if (status != cudaSuccess){
        std::cout << "Device error matrix allocation error " << status << std::endl;
        return status;
    }
    if (status_A != cudaSuccess){
        std::cout << "Kernel A allocation error " << status << std::endl;
        return status;
    } else if (status_B != cudaSuccess){
        std::cout << "Kernel B allocation error " << status << std::endl;
        return status;
    }

    status_A = cudaMemcpy(dev_A, A_kernel, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_A != cudaSuccess){
        std::cout << "Kernel A copy to device error " << status << std::endl;
        return status_A;
    }
    status_B = cudaMemcpy(dev_B, B_kernel, sizeof(double) * full_size, cudaMemcpyHostToDevice);
    if (status_B != cudaSuccess){
        std::cout << "kernel B copy to device error " << status << std::endl;
        return status_B;
    }

    status = cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
    if (status != cudaSuccess){
        std::cout << "Max reduction error " << status << std::endl;
        return status;
    }

    status = cudaMalloc(&temp_stor, tmp_stor_size);
    if (status != cudaSuccess){
        std::cout << "Temporary storage allocation error " << status  << std::endl;
        return status;
    }

    int i = 0;
    double error = 1.0;
    // openining the nvtx mark for profiling
    nvtxRangePushA("Main loop");
    // main loop
    while (i < max_iter && error > min_error){
        i++;
        // compute one cross compurtation
        cross_calc<<<size-1, size-1>>>(dev_A, dev_B, size);

        if (i % 100 == 0){
            // get the error matrix. the difference between the matrixes
            // number of threads = (size-1)^2
            get_error_matrix<<<size - 1, size - 1>>>(dev_A, dev_B, dev_err_mat);
            // find the maximum error. result in dev_err
            cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
            // copying the error from device to host memory
            cudaMemcpy(&error, dev_err, sizeof(double), cudaMemcpyDeviceToHost);

        }
        // matrix swapping
        std::swap(dev_A, dev_B);


    }
    // closing the nvtx mark
    nvtxRangePop();
    // matrix output check after the computations
    // cudaMemcpy(A_kernel, dev_A, sizeof(double) * full_size, cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < size; i ++) {
    //     for (int j = 0; j < size; j ++) {
    //         std::cout << A_kernel[j * size + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // print out the results
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << i << std::endl;
    // memory deallocation
    cudaFree(temp_stor);
    cudaFree(dev_err_mat);
    cudaFree(dev_A);
    cudaFree(dev_B);
    delete[] A_kernel;
    delete[] B_kernel;
    return 0;
}
