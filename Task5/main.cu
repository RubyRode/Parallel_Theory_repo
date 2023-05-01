#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <mpi.h>

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
void cross_calc(double* A_kernel, double* B_kernel, size_t size, size_t dev_size){
    // get the block and thread indices
    
    size_t j = blockIdx.x;
    size_t i = threadIdx.x;
    // main computation
    if (i != 0 && j != 0 && j != size && i != dev_size - 1){
       
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
    // get index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // take the maximum error
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

    // MPI initialization 
    int rank, group_size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);
    // Set device number 
    cudaSetDevice(rank);

    size_t proc_area = size / group_size;
    size_t start_idx = proc_area * rank;

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

    // for (int i = 0; i < size; i ++) {
    //     for (int j = 0; j < size; j ++) {
    //         std::cout << A_kernel[j * size + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    double* dev_A, *dev_B, *dev_err, *dev_err_mat, *temp_stor = NULL;
    
    // memory for one process
    if (rank != 0 && rank != group_size -1){
        proc_area += 2;
    }else{
        proc_area++;
    }

    size_t mem_size = size * proc_area;
    
    unsigned int threads_x = (size <= 1024) ? size : 1024;
    unsigned int blocks_x = proc_area;
    unsigned int blocks_y = size / threads_x;

    dim3 blockDim(threads_x, 1);
    dim3 gridDim(blocks_x, blocks_y);

    
    // Memory allocation on the device
    // Kernels A and B
    cudaMalloc(&dev_A, sizeof(double) * full_size);
    cudaMalloc(&dev_B, sizeof(double) * full_size);
    // Device error variable
    cudaMalloc(&dev_err, sizeof(double));

    // Device error matrix allocation
    cudaMalloc(&dev_err_mat, sizeof(double) * full_size);

    // one row offset for the grid
    size_t offset = (rank != 0) ? size : 0;
    // reset matrix
    cudaMemset(dev_A, 0, sizeof(double) * mem_size);
    cudaMemset(dev_B, 0, sizeof(double) * mem_size);
    // copy matrices to device
    cudaMemcpy(dev_A, A_kernel + (start_idx * size) - offset, sizeof(double) * mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B_kernel + (start_idx * size) - offset, sizeof(double) * mem_size, cudaMemcpyHostToDevice);


    // Temporary storage allocation
    size_t tmp_stor_size = 0;
    cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, size * proc_area);
    cudaMalloc(&temp_stor, tmp_stor_size);

    int i = 0;
    double error = 1.0;

    nvtxRangePushA("Main loop");
    // main loop
    while (i < max_iter && error > min_error){
        i++;
        // compute the iteration
        cross_calc<<<size-1, size-1>>>(dev_A, dev_B, size, proc_area);

        if (i % 100 == 0){
            // get the error matrix
            get_error_matrix<<<size - 1, size - 1>>>(dev_A, dev_B, dev_err_mat);
            // find the maximum error
            cub::DeviceReduce::Max(temp_stor, tmp_stor_size, dev_err_mat, dev_err, full_size);
            // copy to host memory
            cudaMemcpy(&error, dev_err, sizeof(double), cudaMemcpyDeviceToHost);

        }
        // Bounds exchange
        // Top Bound
        if (rank != 0){
            MPI_Sendrecv(dev_B + size + 1, size - 2, MPI_DOUBLE, rank - 1, 0, 
            dev_B + 1, size - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Bottom Bound
        if (rank != group_size - 1){
            MPI_Sendrecv(dev_B + (proc_area - 2) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0, 
            dev_B + (proc_area - 1) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // matrix swapping
        std::swap(dev_A, dev_B);


    }

    nvtxRangePop();
    // cudaMemcpy(A_kernel, dev_A, sizeof(double) * full_size, cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < size; i ++) {
    //     for (int j = 0; j < size; j ++) {
    //         std::cout << A_kernel[j * size + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    if (rank == 0){
        std::cout << "Error: " << error << std::endl;
        std::cout << "Iteration: " << i << std::endl;
    }
    
    cudaFree(temp_stor);
    cudaFree(dev_err_mat);
    cudaFree(dev_A);
    cudaFree(dev_B);
    delete[] A_kernel;
    delete[] B_kernel;
    MPI_Finalize();
    return 0;
}
