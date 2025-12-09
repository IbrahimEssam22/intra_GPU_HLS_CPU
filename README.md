nvcc intra_predect_GPU.cu gpu_testbench.cu -o gpu_test_perf_v2 && ./gpu_test_perf_v2

gcc cpu_intra_predict.c cpu_testbench.c -o cpu_test_perf -O3 && ./cpu_test_perf
