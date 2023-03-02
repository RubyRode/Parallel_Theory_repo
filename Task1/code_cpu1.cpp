#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

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

class Grid{
private:

    std::vector<std::vector<double>> a;
    std::vector<std::vector<double>> a_new;
    int size;
    double corners[4] = {10, 20, 30, 20};
    double _accuracy;
    int _iters;

public:
    explicit Grid(parser input){
        this->size = input.grid();
        this->a = std::vector<std::vector<double>>(this->size, std::vector<double>(this->size, 0));
        this->a_new = std::vector<std::vector<double>>(this->size, std::vector<double>(this->size, 0));
        this->a_new[0][0] = this->a[0][0] = this->corners[0];
        this->a_new[0][this->size - 1] = this->a[0][this->size - 1] = this->corners[1];
        this->a_new[this->size - 1][0] = this->a[this->size - 1][0] = this->corners[3];
        this->a_new[this->size - 1][this->size - 1] = this->a[this->size - 1][this->size - 1] = this->corners[2];
        this->_accuracy = input.accuracy();
        this->_iters = input.iterations();
    }

    void fill(){

        double step = (this->corners[1] - this->corners[0]) / (this->size - 1);
        #pragma acc data copy(this->a[0:this->size]), copy(this->a_new[0:this->size])
        {
            #pragma acc parallel loop seq gang num_gangs(256) vector vector_length(256)
            for (int i=1; i<this->size-1; i++){
                this->a_new[0][i] = this->a[0][i] = this->a[0][i - 1] + step;
                this->a_new[this->size - 1][i] = this->a[this->size - 1][i] = this->a[this->size - 1][i - 1] + step;
                this->a_new[i][0] = this->a[i][0] = this->a[i - 1][0] + step;
                this->a_new[i][this->size-1] = this->a[i][this->size - 1] = this->a[i - 1][this->size - 1] + step;
            }
        }


    }

    double update(double error){
        error = 0.0;

        #pragma acc parallel loop seq vector vector_length(this->size) gang num_gangs(256) reduction(max:error) \
            present(a[0:this->size], a_new[0:this->size])
        for (int i=1; i<this->size-1; i++){
            for (int j=1; j<this->size-1; j++){
                this->a[i][j] = (this->a[i - 1][j] + this->a[i][j - 1] + this->a[i][j + 1] + this->a[i + 1][j]) / 4;
                error = fmax(error, this->a[i][j] - this->a_new[i][j]);
            }
        }
        return error;
    }

    void sw_ap(){
        #pragma acc parallel loop seq vector vector vector_length(this->size) gang num_gangs(256) \
        present(this->a[0:this->size], this->a_new[0:this->size])
        for (int i=1; i<this->size-1; i++){
            for (int j=1; j<this->size-1; j++){
                this->a_new[i][j] = this->a[i][j];
            }
        }
    }

    [[nodiscard]] double grid_val(int i, int j) const{
        return this->a[i][j];
    }
    [[nodiscard]] int iters() const{
        return this->_iters;
    }
    [[nodiscard]] double accuracy() const{
        return this->_accuracy;
    }

};


int main(int argc, char ** argv){
    parser input = parser(argc, argv);
    Grid gr(input);

    clock_t start = clock();

    gr.fill();

    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    std::cout << "Initialization time: " << elapsed_secs << std::endl;

    double error = 1;
    int g;
    start = clock();

    for (int i = 0; i < gr.iters() && error > gr.accuracy(); i++) {
         error = gr.update(error);
         gr.sw_ap();
         g = i;
    }

    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    std::cout << "Computations time: " << elapsed_secs << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << g+1 << std::endl;
    return 0;
}
