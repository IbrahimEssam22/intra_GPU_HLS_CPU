#include "cpu_intra_predict.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h> // Added for printf debugging

// Kvazaar's exact lookup tables for angular prediction
static const int8_t modedisp2sampledisp[9] = { 0, 2, 5, 9, 13, 17, 21, 26, 32 };
static const int16_t modedisp2invsampledisp[9] = { 0, 4096, 1638, 910, 630, 482, 390, 315, 256 };

// Top-level function (CPU version simulating HLS)
void cpu_intra_predict_top(
    struct intra_input_t *input,
    struct intra_output_t *output
) {
    // Local arrays for filtered references
    pixel_t left_filtered[REF_SIZE];
    pixel_t top_filtered[REF_SIZE];
    
    // Temporary prediction arrays for all modes
    pixel_t pred_arrays[NUM_MODES][BLOCK_SIZE][BLOCK_SIZE];
    
    // Filter references once
    filter_references_cpu(input->left_ref, input->top_ref, left_filtered, top_filtered);
    
    // Generate all predictions (simulating parallel execution)
    for (int mode = 0; mode < NUM_MODES; mode++) {
        if (mode == 0) {
            // Mode 0: Planar (always use filtered references)
            planar_predict_cpu(left_filtered, top_filtered, pred_arrays[mode]);
        }
        else if (mode == 1) {
            // Mode 1: DC (use unfiltered references)
            dc_predict_cpu(input->left_ref, input->top_ref, pred_arrays[mode]);
        }
        else {
            // Modes 2-34: Angular - Use exact Kvazaar algorithm
            kvz_angular_predict_cpu(input->left_ref, input->top_ref, mode, pred_arrays[mode]);
        }
    }
    
    // Calculate SAD costs for all modes
    for (int mode = 0; mode < NUM_MODES; mode++) {
        output->costs[mode] = calculate_sad_cpu(input->orig_block, pred_arrays[mode]);
    }
    
    // Copy prediction matrices to output
    for (int mode = 0; mode < NUM_MODES; mode++) {
        for (int y = 0; y < BLOCK_SIZE; y++) {
            for (int x = 0; x < BLOCK_SIZE; x++) {
                output->predictions[mode][y][x] = pred_arrays[mode][y][x];
            }
        }
    }
}

// Reference filtering with 3-tap filter [1 2 1]/4
void filter_references_cpu(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    pixel_t left_filtered[REF_SIZE],
    pixel_t top_filtered[REF_SIZE]
) {
    // Filter corner pixel
    left_filtered[0] = (left_ref[1] + 2 * left_ref[0] + top_ref[1] + 2) >> 2;
    top_filtered[0] = left_filtered[0];
    
    // Filter left edge (except endpoints)
    for (int i = 1; i < REF_SIZE - 1; i++) {
        left_filtered[i] = (left_ref[i-1] + 2 * left_ref[i] + left_ref[i+1] + 2) >> 2;
    }
    left_filtered[REF_SIZE-1] = left_ref[REF_SIZE-1];
    
    // Filter top edge (except endpoints)  
    for (int i = 1; i < REF_SIZE - 1; i++) {
        top_filtered[i] = (top_ref[i-1] + 2 * top_ref[i] + top_ref[i+1] + 2) >> 2;
    }
    top_filtered[REF_SIZE-1] = top_ref[REF_SIZE-1];
}

// Planar prediction for Mode 0
void planar_predict_cpu(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
    // Fixed values for 8x8 block
    const pixel_t top_right = top_ref[BLOCK_SIZE + 1];    // top_ref[9]
    const pixel_t bottom_left = left_ref[BLOCK_SIZE + 1]; // left_ref[9]
    
    for (int y = 0; y < BLOCK_SIZE; y++) {
        for (int x = 0; x < BLOCK_SIZE; x++) {
            // Horizontal interpolation: (7-x) * left + (x+1) * top_right
            uint32_t hor = (BLOCK_SIZE - 1 - x) * left_ref[y + 1] + (x + 1) * top_right;
            
            // Vertical interpolation: (7-y) * top + (y+1) * bottom_left  
            uint32_t ver = (BLOCK_SIZE - 1 - y) * top_ref[x + 1] + (y + 1) * bottom_left;
            
            // Average and round: (hor + ver + 8) >> 4
            pred[y][x] = (hor + ver + BLOCK_SIZE) >> 4;
        }
    }
}

// DC prediction for Mode 1
void dc_predict_cpu(
    const pixel_t left_ref[REF_SIZE], 
    const pixel_t top_ref[REF_SIZE],
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
    // Calculate DC value: sum of left and top references
    uint32_t sum = 0;
    for (int i = 1; i <= BLOCK_SIZE; i++) {
        sum += left_ref[i] + top_ref[i];
    }
    
    pixel_t dc_val = (sum + BLOCK_SIZE) >> 4;  // (sum + 8) >> 4
    
    // For 8x8 luma blocks, use filtered DC with edge enhancement
    for (int y = 0; y < BLOCK_SIZE; y++) {
        for (int x = 0; x < BLOCK_SIZE; x++) {
            if (y == 0 && x == 0) {
                // Filter corner: (left[1] + 2*dc + top[1] + 2) / 4
                pred[y][x] = (left_ref[1] + 2 * dc_val + top_ref[1] + 2) >> 2;
            }
            else if (y == 0) {
                // Filter top edge: (top[x+1] + 3*dc + 2) / 4
                pred[y][x] = (top_ref[x + 1] + 3 * dc_val + 2) >> 2;
            }
            else if (x == 0) {
                // Filter left edge: (left[y+1] + 3*dc + 2) / 4
                pred[y][x] = (left_ref[y + 1] + 3 * dc_val + 2) >> 2;
            }
            else {
                // Interior pixels: plain DC
                pred[y][x] = dc_val;
            }
        }
    }
}

// Exact Kvazaar angular prediction implementation
void kvz_angular_predict_cpu(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE], 
    int mode,
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
    pixel_t tmp_ref[2 * 32];
    const int width = BLOCK_SIZE; // 8x8 fixed
    
    // Whether to swap references to always project on the left reference row.
    const bool vertical_mode = mode >= 18;
    // Modes distance to horizontal or vertical mode.
    const int mode_disp = vertical_mode ? mode - 26 : 10 - mode;
    // Sample displacement per column in fractions of 32.
    const int sample_disp = (mode_disp < 0 ? -1 : 1) * modedisp2sampledisp[abs(mode_disp)];

    // Pointer for the reference we are interpolating from.
    const pixel_t *ref_main;
    // Pointer for the other reference.
    const pixel_t *ref_side;
    
    // Determine which reference to use based on exact Kvazaar logic
    const pixel_t *left_to_use, *top_to_use;
    static const int kvz_intra_hor_ver_dist_thres[5] = { 0, 7, 1, 0, 0 };
    int filter_threshold = kvz_intra_hor_ver_dist_thres[3 - 2]; // log2_width=3 for 8x8
    int dist_from_vert_or_hor = MIN(abs(mode - 26), abs(mode - 10));
    int use_filtered = (dist_from_vert_or_hor > filter_threshold);
    
    // Buffers for filtered references (must be in scope for the whole function)
    pixel_t left_filtered[REF_SIZE], top_filtered[REF_SIZE];
    
    if (use_filtered) {
        // Use filtered references - need to filter them first
        filter_references_cpu(left_ref, top_ref, left_filtered, top_filtered);
        left_to_use = left_filtered;
        top_to_use = top_filtered;
    } else {
        left_to_use = left_ref;
        top_to_use = top_ref;
    }

    // Set ref_main and ref_side such that, when indexed with 0, they point to
    // index 0 in block coordinates.
    if (sample_disp < 0) {
        // Negative sample_disp means, we need to use both references.
        ref_side = (vertical_mode ? left_to_use : top_to_use) + 1;
        ref_main = (vertical_mode ? top_to_use : left_to_use) + 1;

        // Move the reference pixels to start from the middle to the later half of
        // the tmp_ref, so there is room for negative indices.
        for (int x = -1; x < width; ++x) {
            tmp_ref[x + width] = ref_main[x];
        }
        // Get a pointer to block index 0 in tmp_ref.
        ref_main = &tmp_ref[width];

        // Extend the side reference to the negative indices of main reference.
        int col_sample_disp = 128; // rounding for the ">> 8"
        int inv_abs_sample_disp = modedisp2invsampledisp[abs(mode_disp)];
        int most_negative_index = (width * sample_disp) >> 5;
        for (int x = -2; x >= most_negative_index; --x) {
            col_sample_disp += inv_abs_sample_disp;
            int side_index = col_sample_disp >> 8;
            tmp_ref[x + width] = ref_side[side_index - 1];
        }
    }
    else {
        // sample_disp >= 0 means we don't need to refer to negative indices,
        // which means we can just use the references as is.
        ref_main = (vertical_mode ? top_to_use : left_to_use) + 1;
        ref_side = (vertical_mode ? left_to_use : top_to_use) + 1;
    }

    if (sample_disp != 0) {
        // The mode is not horizontal or vertical, we have to do interpolation.
        int delta_pos = 0;
        for (int y = 0; y < width; ++y) {
            delta_pos += sample_disp;
            int delta_int = delta_pos >> 5;
            int delta_fract = delta_pos & (32 - 1);

            if (delta_fract) {
                // Do linear filtering
                for (int x = 0; x < width; ++x) {
                    pixel_t ref1 = ref_main[x + delta_int];
                    pixel_t ref2 = ref_main[x + delta_int + 1];
                    pred[y][x] = ((32 - delta_fract) * ref1 + delta_fract * ref2 + 16) >> 5;
                }
            }
            else {
                // Just copy the integer samples
                for (int x = 0; x < width; x++) {
                    pred[y][x] = ref_main[x + delta_int];
                }
            }
        }
    }
    else {
        // Mode is horizontal or vertical, just copy the pixels.
        for (int y = 0; y < width; ++y) {
            for (int x = 0; x < width; ++x) {
                pred[y][x] = ref_main[x];
            }
        }
    }

    // Flip the block if this is was a horizontal mode.
    if (!vertical_mode) {
        for (int y = 0; y < width - 1; ++y) {
            for (int x = y + 1; x < width; ++x) {
                pixel_t temp = pred[y][x];
                pred[y][x] = pred[x][y];
                pred[x][y] = temp;
            }
        }
    }
}

// Calculate SAD cost between original and prediction
cost_t calculate_sad_cpu(
    const pixel_t orig[BLOCK_SIZE][BLOCK_SIZE],
    const pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
    uint32_t sad = 0;
    
    for (int y = 0; y < BLOCK_SIZE; y++) {
        for (int x = 0; x < BLOCK_SIZE; x++) {
            int32_t diff = orig[y][x] - pred[y][x];
            sad += (diff < 0) ? -diff : diff;  // Absolute difference
        }
    }
    
    return (cost_t)sad;
}
