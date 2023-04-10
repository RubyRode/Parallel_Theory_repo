#include <iostream>
#include <cmath>
#include <cstring>

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    double min_error = input.accuracy();
    int max_iter = input.iterations();
    int full_size = size * size;
    double step = (corners[1] - corners[0]) / (size - 1);

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


    cublasStatus_t status;
    cublasHandle_t handle;

    double error = 1.0;
    int iter = 0;
    status = cublasCreate(&handle);


nvtxRangePushA("Main loop");
#pragma acc enter data copyin(A_kernel[0:full_size], B_kernel[0:full_size]) 
	{
		int idx = 0;
		double alpha = -1.0;

		while (error > min_error && iter < max_iter)
		{
			iter++;

#pragma acc data present(A_kernel, B_kernel)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(1024) async
			for (int i = 1; i < size - 1; i++)
			{
				for (int j = 1; j < size - 1; j++)
				{
					B_kernel[i * size + j] = 0.25 *
						(A_kernel[i * size + j - 1] +
							A_kernel[(i - 1) * size + j] +
							A_kernel[(i + 1) * size + j] +
							A_kernel[i * size + j + 1]);
				}
			}

			if (iter % 100 == 0)
			{

#pragma acc data present (A_kernel, B_kernel) wait
#pragma acc host_data use_device(A_kernel, B_kernel)
				{
			status = cublasDaxpy(handle,full_size, &alpha, B_kernel, 1, A_kernel, 1);
			status = cublasIdamax(handle, full_size, A_kernel, 1, &idx);
				}


#pragma acc update host(A_kernel[idx - 1]) 
			error = std::abs(A_kernel[idx - 1]);
		
#pragma acc host_data use_device(A_kernel, B_kernel)
			status = cublasDcopy(handle, full_size, B_kernel, 1, A_kernel, 1);
			}

			double* temp = A_kernel;
			A_kernel = B_kernel;
			B_kernel = temp;
		}


	}
    nvtxRangePop();

    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl;
    cublasDestroy(handle);
    delete[] A_kernel;
    delete[] B_kernel;
    return 0;
}
