// nvcc intra_predect_GPU.cu gpu_testbench.cu -o gpu_test_separated && ./gpu_test_separated
// nvcc intra_predect_GPU.cu gpu_testbench.cu -o gpu_test_perf_v2 && ./gpu_test_perf_v2
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "intra_predect_GPU.h"
#include "allMats.c"

#define REF_SIZE 17

void filter_references(
    const int* left_ref,
    const int* top_ref,
    int* left_filtered,
    int* top_filtered
) {
    // Filter corner pixel
    left_filtered[0] = (left_ref[1] + (left_ref[0] << 1) + top_ref[1] + 2) >> 2;
    top_filtered[0] = left_filtered[0];
    
    // Filter left edge
    for (int i = 1; i < REF_SIZE - 1; i++) {
        left_filtered[i] = (left_ref[i-1] + (left_ref[i] << 1) + left_ref[i+1] + 2) >> 2;
    }
    left_filtered[REF_SIZE-1] = left_ref[REF_SIZE-1];
    
    // Filter top edge
    for (int i = 1; i < REF_SIZE - 1; i++) {
        top_filtered[i] = (top_ref[i-1] + (top_ref[i] << 1) + top_ref[i+1] + 2) >> 2;
    }
    top_filtered[REF_SIZE-1] = top_ref[REF_SIZE-1];
}

int main() {
    printf("=== GPU Intra Prediction Testbench ===\n");

    // 1. Prepare Input Data
    int h_l_raw[REF_SIZE];
    int h_t_raw[REF_SIZE];
    int h_l_filt[REF_SIZE];
    int h_t_filt[REF_SIZE];

    // Copy from allMats.c (unsigned char) to int arrays
    for (int i = 0; i < REF_SIZE; i++) {
        h_l_raw[i] = left_ref[i];
        h_t_raw[i] = top_ref[i];
    }

    // Filter references
    filter_references(h_l_raw, h_t_raw, h_l_filt, h_t_filt);

    // 2. Allocate Output Buffer
    // 35 modes, 8x8 block per mode
    unsigned char* h_output = (unsigned char*)malloc(35 * 64 * sizeof(unsigned char));

    // 3. Run GPU Kernel
    printf("Running GPU kernel...\n");
    launch_mode_major_intra(1, h_l_raw, h_t_raw, h_l_filt, h_t_filt, h_output);

    // 4. Verify Results
    printf("Verifying results...\n");
    int total_errors = 0;
    
    // Calculate expected costs (SAD) and compare with GPU output SAD
    // Note: The GPU outputs pixels, so we calculate SAD of GPU output vs Original
    // and compare it with Expected SAD (calculated from Reference Predictions vs Original)
    
    for (int mode = 0; mode < 35; mode++) {
        int gpu_sad = 0;
        int ref_sad = 0;
        int pixel_errors = 0;

        printf("Mode %2d: ", mode);

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                unsigned char gpu_val = h_output[(mode * 64) + (y * 8) + x];
                unsigned char ref_val = reference_predictions[mode][y][x];
                unsigned char orig_val = test_block[y][x];

                // Check pixel match
                if (gpu_val != ref_val) {
                    pixel_errors++;
                }

                // Calculate SADs
                gpu_sad += abs((int)gpu_val - (int)orig_val);
                ref_sad += abs((int)ref_val - (int)orig_val);
            }
        }

        if (pixel_errors == 0) {
            printf("PASS (SAD: %d)\n", gpu_sad);
        } else {
            printf("FAIL (Pixels mismatch: %d, GPU SAD: %d, Ref SAD: %d)\n", pixel_errors, gpu_sad, ref_sad);
            total_errors++;
        }
    }

    if (total_errors == 0) {
        printf("\nALL MODES PASSED!\n");
    } else {
        printf("\n%d MODES FAILED!\n", total_errors);
    }

    // 5. Performance Measurement
    printf("\n=== Performance Measurement ===\n");
    const int NUM_ITERATIONS = 1000;
    printf("Running %d iterations to calculate average execution time...\n", NUM_ITERATIONS);
    printf("(Includes: cudaMalloc, HostToDevice, Kernel Execution, DeviceToHost, cudaFree)\n");
    
    // Warmup call
    launch_mode_major_intra(1, h_l_raw, h_t_raw, h_l_filt, h_t_filt, h_output);
    
    double total_accumulated_time_us = 0.0;
    
    for(int i=0; i<NUM_ITERATIONS; i++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        launch_mode_major_intra(1, h_l_raw, h_t_raw, h_l_filt, h_t_filt, h_output);
        
        gettimeofday(&end, NULL);
        double iteration_time = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
        total_accumulated_time_us += iteration_time;
    }
    
    double avg_time_us = total_accumulated_time_us / NUM_ITERATIONS;
    
    printf("Total accumulated time for %d iterations: %.2f ms\n", NUM_ITERATIONS, total_accumulated_time_us / 1000.0);
    printf("Average time per call: %.2f us\n", avg_time_us);

    free(h_output);
    return 0;
}
