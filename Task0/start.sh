echo "g++ float compiling" > all.txt
g++ code_cpu.cpp -o code_cpu_float -D FLOAT_TYPE >> all.txt
echo "g++ double compiling" >> all.txt
g++ code_cpu.cpp -o code_cpu_double >> all.txt
echo "pgc++ float compiling" >> all.txt
pgc++ code_cpu.cpp -o code_gpu_float -D FLOAT_TYPE -acc -Minfo=accel >> all.txt
echo "pgc++ double compiling" >> all.txt
pgc++ code_cpu.cpp -o code_gpu_double -acc -Minfo=accel >> all.txt
echo "cpu double result:" >> all.txt
./code_cpu_double >> all.txt
echo "cpu float result" >> all.txt
./code_cpu_float >> all.txt
echo "gpu double result" >> all.txt
PGI_ACC_TIME=1 nvprof --log-file gpu_d_prof.txt ./code_gpu_double >> all.txt
echo "gpu float result" >> all.txt
PGI_ACC_TIME=1 nvprof --log-file gpu_f_prof.txt ./code_gpu_float >> all.txt
