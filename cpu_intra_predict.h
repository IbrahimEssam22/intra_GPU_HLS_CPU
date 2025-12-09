#ifndef CPU_INTRA_PREDICT_H
#define CPU_INTRA_PREDICT_H

#include <stdint.h>

// Utility macros
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// CPU-compatible data types (simulating HLS types)
typedef uint8_t pixel_t;        // 8-bit pixel values
typedef int16_t diff_t;         // 16-bit for differences  
typedef uint16_t cost_t;        // 16-bit for SAD costs
typedef int16_t coeff_t;        // 16-bit for interpolation coefficients

// Fixed block size for hardware implementation
#define BLOCK_SIZE 8
#define NUM_MODES 35
#define REF_SIZE 17  // 2*BLOCK_SIZE + 1

// Input/Output structures for testing
struct intra_input_t {
    pixel_t orig_block[BLOCK_SIZE][BLOCK_SIZE];
    pixel_t left_ref[REF_SIZE];   // Reference pixels: left[0..16]
    pixel_t top_ref[REF_SIZE];    // Reference pixels: top[0..16]
};

struct intra_output_t {
    cost_t costs[NUM_MODES];      // SAD costs for all 35 modes
    pixel_t predictions[NUM_MODES][BLOCK_SIZE][BLOCK_SIZE]; // All prediction matrices
};

// Function prototypes
void cpu_intra_predict_top(
    struct intra_input_t *input,
    struct intra_output_t *output
);

void filter_references_cpu(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    pixel_t left_filtered[REF_SIZE],
    pixel_t top_filtered[REF_SIZE]
);

void planar_predict_cpu(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
);

void dc_predict_cpu(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
);

void kvz_angular_predict_cpu(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    int mode,
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
);

cost_t calculate_sad_cpu(
    const pixel_t orig[BLOCK_SIZE][BLOCK_SIZE],
    const pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
);

#endif // CPU_INTRA_PREDICT_H
