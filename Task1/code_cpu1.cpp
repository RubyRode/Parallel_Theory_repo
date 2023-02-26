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
            }else if (arg == "grid"){
                this->_grid_size = std::stoi(argv[i + 1]);
            }else if (arg == "i"){
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
    int iters;

public:
    explicit Grid(parser input){
        this->size = input.grid();
        this->grid = std::vector<std::vector<double>>(this->size, std::vector<double>(this->size, 0));
        this->grid[0][0] = this->corners[0];
        this->grid[0][this->size-1] = this->corners[1];
        this->grid[this->size-1][0] = this->corners[2];
        this->grid[this->size-1][this->size-1] = this->corners[3];
        this->accuracy = input.accuracy();
        this->iters = input.iterations();
    }

    [[nodiscard]] double grid_val(int i, int j) const{
        return this->grid[i][j];
    }

};


int main(int argc, char ** argv){
    parser input = parser(argc, argv);
    Grid gr(input);

    return 0;
}