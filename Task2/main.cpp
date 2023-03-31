#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <driver_types.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

class parser{
public:
    parser(int argc, char** argv){
        this->_grid_size = 512;
        this->_accur = 10e-6;
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
    [[nodiscard]] double accuracy() const{
        return this->_accur;
    }
    [[nodiscard]] int iterations() const{
        return this->_iters;
    }
    [[nodiscard]] int grid()const{
        return this->_grid_size;
    }
private:
    double _accur;
    int _grid_size;
    int _iters;

};

double corners[4] = {10, 20, 30, 20};

int main(int argc, char ** argv){
    parser input = parser(argc, argv);

    int size = input.grid();

    auto* A_kernel = new double[size * size];
    auto* B_kernel = new double[size * size];

    std::memset(A_kernel, 0, sizeof(double) * size * size);


    A_kernel[IDX2C(0, 0, size)] = corners[0];
    A_kernel[IDX2C(0, size-1, size)] = corners[1];
    A_kernel[IDX2C(size-1, size-1, size)] = corners[2];
    A_kernel[IDX2C(size-1, 0, size)] = corners[3];

    int full_size = size * size;
    double step = (corners[1] - corners[0]) / (size - 1);

    for (int i = 0; i < size; i ++) {
        A_kernel[IDX2C(i, 0, size)] = corners[0] + i * step;
        A_kernel[IDX2C(0, i, size)] = corners[0] + i * step;
        A_kernel[IDX2C(i, size-1, size)] = corners[1] + i * step;
        A_kernel[IDX2C(size-1, i, size)] = corners[1] + i * step;
    }

//    for (int i = 0; i < size; i ++){
//        for (int j = 0; j < size; j ++){
//            std::cout << A_kernel[IDX2C(i, j, size)] << ' ';
//        }
//        std::cout << std::endl;
//    }
    std::memcpy(B_kernel, A_kernel, sizeof(double) * full_size);

    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;

    double error = 1.0;
    int iter = 0;
    double min_error = input.accuracy();
    int max_iter = input.iterations();
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        std::cout << "cuBLAS init failed" << std::endl;
        return EXIT_FAILURE;
    }

    nvtxRangePushA("Main loop");
#pragma acc enter data copyin(B_kernel[0:full_size], A_kernel[0:full_size])
    {
        int index = 0;
        double alpha = -1.0;

        while (error > min_error && iter < max_iter) {
            iter++;

#pragma acc data present(A_kernel, B_kernel)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) async
            for (int i = 1; i < size - 1; i++)
            {
                for (int j = 1; j < size-1; j++)
                {
                    B_kernel[IDX2C(i, j, size)] = 0.25 *
                            (A_kernel[IDX2C(i-1, j, size)] +
                            A_kernel[IDX2C(i, j - 1, size)] +
                            A_kernel[IDX2C(i, j + 1, size)] +
                            A_kernel[IDX2C(i + 1, j, size)]);
                }
            }
            if(iter % 100 == 0){
#pragma acc data present (A_kernel, B_kernel) wait
#pragma acc host_data use_device(A_kernel, B_kernel)
                {
                    // находим разницу матриц
                    status = cublasDaxpy(handle, full_size, &alpha, B_kernel, 1, A_kernel, 1);
                    if (status != CUBLAS_STATUS_SUCCESS){
                        std::cout << "Daxpy failed" << std::endl;
                        cublasDestroy(handle);
                        return EXIT_FAILURE;
                    }
                    // находим индекс наибольшего элемента
                    status = cublasIdamax(handle, full_size, A_kernel, 1, &index);
                    if (status != CUBLAS_STATUS_SUCCESS){
                        std::cout << "Idamax failed: " << status <<std::endl;
                        cublasDestroy(handle);
                        return EXIT_FAILURE;
                    }
                }
// обновляем значение ошибки на ЦПУ
#pragma acc update host(A_kernel[index-1]);
            error = std::abs(A_kernel[index-1]);
// возвращаем значения матрицы А
#pragma acc host_data use_device(A_kernel, B_kernel)
            status = cublasDcopy(handle, full_size, B_kernel, 1, A_kernel, 1);
            }
            double* temp = A_kernel;
            A_kernel = B_kernel;
            B_kernel = temp;

        }
    }
    cublasDestroy(handle);
    nvtxRangePop();

    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl;
    free(A_kernel);
    free(B_kernel);
    return 0;
}