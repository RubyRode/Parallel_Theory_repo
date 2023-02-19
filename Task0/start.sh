pgc++ code_cpu.cpp -o code_cpu_1_float -D FLOAT_TYPE -acc=host -Minfo=accel
pgc++ code_cpu.cpp -o code_cpu_1_double -acc=host -Minfo=accel
pgc++ code_cpu.cpp -o code_multicore_float -D FLOAT_TYPE -acc=multicore -Minfo=accel
pgc++ code_cpu.cpp -o code_multicore_double -acc=multicore -Minfo=accel
pgc++ code_cpu.cpp -o code_gpu_float -D FLOAT_TYPE -acc=gpu -Minfo=accel
pgc++ code_cpu.cpp -o code_gpu_double -acc=gpu -Minfo=accel
echo "cpu-1 double result:" > results.txt
./code_cpu_1_double >> results.txt
echo "cpu-1 float result" >> results.txt
./code_cpu_1_float >> results.txt
echo "multicore double result" >> results.txt
./code_multicore_double >> results.txt
echo "multicore float result" >> results.txt
./code_multicore_float >> results.txt 
echo "gpu double result" >> results.txt
PGI_ACC_TIME=1 nvprof --log-file gpu_d_prof.txt ./code_gpu_double >> results.txt
echo "gpu float result" >> results.txt
PGI_ACC_TIME=1 nvprof --log-file gpu_f_prof.txt ./code_gpu_float >> results.txt
echo "Program: start.sh\nCompiler: pgc++\n\nResults of computations(sum,time):" > summary.txt
cat results.txt >> summary.txt
echo "Code: code_cpu.cpp\n" >> summary.txt 
echo "Multicore computations on host(cpu) are the longest. It takes a lot of time to parallelize the code between cpu cores. Cpu imperative computations on one core take lesser time. But both of them cannot compare with gpu parallelized computations. It takes almost x10 lesser time than cpu computations on one core. But running gpu code is more time consumptioning because of data movements." >> summary.txt
