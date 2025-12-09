#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "cpu_intra_predict.h"
#include "allMats.c"

int main() {
    printf("=== CPU Intra Prediction Testbench ===\n");

    // 1. Prepare Input Data
    struct intra_input_t input;
    struct intra_output_t output;

    // Copy from allMats.c to input structure
    for (int y = 0; y < BLOCK_SIZE; y++) {
        for (int x = 0; x < BLOCK_SIZE; x++) {
            input.orig_block[y][x] = test_block[y][x];
        }
    }
    for (int i = 0; i < REF_SIZE; i++) {
        input.left_ref[i] = left_ref[i];
        input.top_ref[i] = top_ref[i];
    }

    // 2. Run CPU Function (Verification Run)
    printf("Running CPU function...\n");
    cpu_intra_predict_top(&input, &output);

    // 3. Verify Results
    printf("Verifying results...\n");
    int total_errors = 0;
    
    for (int mode = 0; mode < NUM_MODES; mode++) {
        int cpu_sad = output.costs[mode];
        int ref_sad = 0;
        int pixel_errors = 0;

        // Calculate expected SAD from reference predictions
        for (int y = 0; y < BLOCK_SIZE; y++) {
            for (int x = 0; x < BLOCK_SIZE; x++) {
                pixel_t cpu_val = output.predictions[mode][y][x];
                pixel_t ref_val = reference_predictions[mode][y][x];
                pixel_t orig_val = test_block[y][x];

                if (cpu_val != ref_val) {
                    pixel_errors++;
                }
                ref_sad += abs((int)ref_val - (int)orig_val);
            }
        }

        if (pixel_errors == 0) {
            printf("Mode %2d: PASS (SAD: %d)\n", mode, cpu_sad);
        } else {
            printf("Mode %2d: FAIL (Pixels mismatch: %d, CPU SAD: %d, Ref SAD: %d)\n", 
                   mode, pixel_errors, cpu_sad, ref_sad);
            total_errors++;
        }
    }

    if (total_errors == 0) {
        printf("\nALL MODES PASSED!\n");
    } else {
        printf("\n%d MODES FAILED!\n", total_errors);
    }

    // 4. Performance Measurement
    printf("\n=== Performance Measurement ===\n");
    const int NUM_ITERATIONS = 10000; // More iterations for CPU as it might be fast/jittery
    printf("Running %d iterations to calculate average execution time...\n", NUM_ITERATIONS);
    
    // Warmup
    cpu_intra_predict_top(&input, &output);
    
    double total_accumulated_time_us = 0.0;
    
    for(int i=0; i<NUM_ITERATIONS; i++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        cpu_intra_predict_top(&input, &output);
        
        gettimeofday(&end, NULL);
        double iteration_time = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
        total_accumulated_time_us += iteration_time;
    }
    
    double avg_time_us = total_accumulated_time_us / NUM_ITERATIONS;
    
    printf("Total accumulated time for %d iterations: %.2f ms\n", NUM_ITERATIONS, total_accumulated_time_us / 1000.0);
    printf("Average time per call: %.2f us\n", avg_time_us);

    return 0;
}
