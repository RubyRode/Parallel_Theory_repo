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
            }else if (arg == "-grid"){
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

    std::vector<std::vector<double>> grid;
    int size;
    double corners[4] = {10, 20, 30, 20};
    double accuracy;
    int _iters;

public:
    explicit Grid(parser input){
        this->size = input.grid();
        this->grid = std::vector<std::vector<double>>(this->size, std::vector<double>(this->size, 0));
        this->grid[0][0] = this->corners[0];
        this->grid[0][this->size-1] = this->corners[1];
        this->grid[this->size-1][0] = this->corners[2];
        this->grid[this->size-1][this->size-1] = this->corners[3];
        this->accuracy = input.accuracy();
        this->_iters = input.iterations();
    }

    void init(){
        for (int i=1; i<this->size-1; i++){
            this->grid[0][i] = this->grid[0][i-1] + ((this->corners[1]-this->corners[0])/(this->size-1));
            this->grid[this->size-1][i] = this->grid[this->size-1][i-1] + ((this->corners[3]-this->corners[2])/(this->size-1));
            this->grid[i][0] = this->grid[i-1][0] + ((this->corners[2]-this->corners[0])/(this->size-1));
            this->grid[i][this->size-1] = this->grid[i-1][this->size-1] + ((this->corners[3]-this->corners[1])/(this->size-1));
        }

    }

    void fill(){
        for (int i=1; i<this->size-1; i++){
            for (int j=1; j<this->size-1; j++){
                this->grid[i][j] = (this->grid[i-1][j] + this->grid[i][j-1] + this->grid[i][j+1] + this->grid[i+1][j])/4;
            }
        }
    }

    [[nodiscard]] double grid_val(int i, int j) const{
        return this->grid[i][j];
    }
    [[nodiscard]] int iters() const{
        return this->_iters;
    }

};


int main(int argc, char ** argv){
    parser input = parser(argc, argv);
    Grid gr(input);


    gr.init();
    for (int i=0; i<gr.iters(); i++) {
        gr.fill();
    }
    for (int i=0; i<input.grid(); i++){
        for (int j=0; j<input.grid(); j++){
            fprintf(stdout, "%0.2f  " , gr.grid_val(i, j));
        }
        std::cout << std::endl;
    }

    

    return 0;
}