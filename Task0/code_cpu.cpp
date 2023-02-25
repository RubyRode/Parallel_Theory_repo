#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define arr_size 10000001

//define
#ifdef FLOAT_TYPE
#define TYPE float
#define FORMAT "%.23f"
#else
#define TYPE double
#define FORMAT "%.23lf"
#endif

using namespace std;



void Arr_fill(TYPE * ds, int len){
    TYPE step = 2.0 * M_PI / len;
    int vector_len = 256;
    #pragma acc parallel loop gang vector vector_length(vector_len) present(ds)
    for (int i = 0; i < len; i++){
        ds[i] = sin(step * i);
    }
}

TYPE Arr_sum(const TYPE * ds, int len){
    TYPE sum = 0.0;
    int vector_len = 256;
    #pragma acc data copy(sum)
    #pragma acc parallel loop gang num_gangs(39063) vector vector_length(vector_len) reduction(+:sum) present(ds)
    for (int i = 0; i < len; i++){
        sum += ds[i];
    }
    return sum;
}

int main (int argc, char** argv){

    double time = 0.0;
    auto * dbs = (TYPE*) malloc(sizeof(TYPE) * arr_size);

    if (dbs == nullptr){
        fprintf(stderr, "Failed to allocate host arrays.\n");
        exit(EXIT_FAILURE);
    }
    #pragma acc data create(dbs[0:arr_size])
    {
        clock_t begin = clock();

        Arr_fill(dbs, arr_size);

        cout << "sum = ";
        printf(FORMAT, Arr_sum(dbs, arr_size));

        clock_t end = clock();
        time += (double)(end - begin)/CLOCKS_PER_SEC;
        cout << "\nTime = " << time << " secs\n" << endl;

    }
    free(dbs);
    return 0;
}
