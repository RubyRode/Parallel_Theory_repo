How to use:

  1) Run `sh start.sh`. It will compile and run `code_cpu.cpp` with g++ and pgc++. Also it creates profiling files.
  2) `all.txt` - the results of computing (time and sum) for both types (double/float) and both parallelized and unparallelized code.
  3) `gpu_d_prof` and `gpu_f_prof` - results of `nvprof` for openacc programm.
