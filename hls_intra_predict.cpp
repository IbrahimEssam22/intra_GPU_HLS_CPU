#include "hls_intra_predict.h"
#include <ap_int.h>

// Pre-computed lookup tables for angular prediction (from Kvazaar)
static const ap_int<8> modedisp2sampledisp[9] = { 0, 2, 5, 9, 13, 17, 21, 26, 32 };
static const ap_uint<16> modedisp2invsampledisp[9] = { 0, 4096, 1638, 910, 630, 482, 390, 315, 256 };

// Kvazaar filtering threshold table
static const ap_int<8> kvz_intra_hor_ver_dist_thres[5] = { 0, 7, 1, 0, 0 };

// Reference filtering with 3-tap filter [1 2 1]/4
void filter_references_hls(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    pixel_t left_filtered[REF_SIZE],
    pixel_t top_filtered[REF_SIZE]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=left_ref complete
#pragma HLS ARRAY_PARTITION variable=top_ref complete
#pragma HLS ARRAY_PARTITION variable=left_filtered complete
#pragma HLS ARRAY_PARTITION variable=top_filtered complete

    // Filter corner pixel
    left_filtered[0] = (left_ref[1] + (left_ref[0] << 1) + top_ref[1] + 2) >> 2;
    top_filtered[0] = left_filtered[0];
    
    // Filter left edge (unroll for performance)
    FILTER_LEFT: for (int i = 1; i < REF_SIZE - 1; i++) {
#pragma HLS UNROLL
        left_filtered[i] = (left_ref[i-1] + (left_ref[i] << 1) + left_ref[i+1] + 2) >> 2;
    }
    left_filtered[REF_SIZE-1] = left_ref[REF_SIZE-1];
    
    // Filter top edge (unroll for performance)
    FILTER_TOP: for (int i = 1; i < REF_SIZE - 1; i++) {
#pragma HLS UNROLL
        top_filtered[i] = (top_ref[i-1] + (top_ref[i] << 1) + top_ref[i+1] + 2) >> 2;
    }
    top_filtered[REF_SIZE-1] = top_ref[REF_SIZE-1];
}

// Optimized planar prediction (Mode 0)
void planar_predict_hls(
    const pixel_t left_filtered[REF_SIZE],
    const pixel_t top_filtered[REF_SIZE],
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=pred complete dim=0

    const pixel_t top_right = top_filtered[BLOCK_SIZE + 1];    // top_filtered[9]
    const pixel_t bottom_left = left_filtered[BLOCK_SIZE + 1]; // left_filtered[9]
    
    PLANAR_ROW: for (int y = 0; y < BLOCK_SIZE; y++) {
#pragma HLS PIPELINE II=1
        PLANAR_COL: for (int x = 0; x < BLOCK_SIZE; x++) {
#pragma HLS UNROLL
            // Pre-compute weights
            ap_uint<4> weight_x_inv = BLOCK_SIZE - 1 - x;  // (7-x)
            ap_uint<4> weight_x = x + 1;                   // (x+1)
            ap_uint<4> weight_y_inv = BLOCK_SIZE - 1 - y;  // (7-y)
            ap_uint<4> weight_y = y + 1;                   // (y+1)
            
            // Horizontal interpolation
            accum_t hor = weight_x_inv * left_filtered[y + 1] + weight_x * top_right;
            
            // Vertical interpolation
            accum_t ver = weight_y_inv * top_filtered[x + 1] + weight_y * bottom_left;
            
            // Average and round
            pred[y][x] = (hor + ver + BLOCK_SIZE) >> 4;
        }
    }
}

// Optimized DC prediction (Mode 1)
void dc_predict_hls(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=pred complete dim=0

    // Calculate DC value using tree reduction
    accum_t sum = 0;
    DC_SUM: for (int i = 1; i <= BLOCK_SIZE; i++) {
#pragma HLS UNROLL
        sum += left_ref[i] + top_ref[i];
    }
    
    pixel_t dc_val = (sum + BLOCK_SIZE) >> 4;  // (sum + 8) >> 4
    
    // Generate DC prediction with edge enhancement
    DC_ROW: for (int y = 0; y < BLOCK_SIZE; y++) {
#pragma HLS PIPELINE II=1
        DC_COL: for (int x = 0; x < BLOCK_SIZE; x++) {
#pragma HLS UNROLL
            if (y == 0 && x == 0) {
                // Filter corner
                pred[y][x] = (left_ref[1] + (dc_val << 1) + top_ref[1] + 2) >> 2;
            }
            else if (y == 0) {
                // Filter top edge
                pred[y][x] = (top_ref[x + 1] + (dc_val * 3) + 2) >> 2;
            }
            else if (x == 0) {
                // Filter left edge
                pred[y][x] = (left_ref[y + 1] + (dc_val * 3) + 2) >> 2;
            }
            else {
                // Interior pixels
                pred[y][x] = dc_val;
            }
        }
    }
}

// Exact Kvazaar angular prediction implementation (HLS version)
void angular_predict_hls(
    const pixel_t left_ref[REF_SIZE],
    const pixel_t top_ref[REF_SIZE],
    intra_mode_t mode,
    pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=pred complete dim=0

    pixel_t tmp_ref[2 * 32];
#pragma HLS ARRAY_PARTITION variable=tmp_ref complete
    
    const ap_int<8> width = BLOCK_SIZE; // 8x8 fixed
    
    // Whether to swap references to always project on the left reference row.
    const bool vertical_mode = mode >= 18;
    
    // Modes distance to horizontal or vertical mode.
    const ap_int<8> mode_disp = vertical_mode ? (ap_int<8>)(mode - 26) : (ap_int<8>)(10 - mode);
    
    // Sample displacement per column in fractions of 32.
    ap_int<8> abs_mode_disp = (mode_disp < 0) ? (ap_int<8>)(-mode_disp) : mode_disp;
    const ap_int<8> sample_disp = (mode_disp < 0 ? (ap_int<8>)(-1) : (ap_int<8>)(1)) * modedisp2sampledisp[abs_mode_disp];
    
    // Pointer for the reference we are interpolating from.
    const pixel_t *ref_main;
    // Pointer for the other reference.
    const pixel_t *ref_side;
    
    // Set ref_main and ref_side such that, when indexed with 0, they point to
    // index 0 in block coordinates.
    if (sample_disp < 0) {
        // Negative sample_disp means, we need to use both references.
        ref_side = (vertical_mode ? left_ref : top_ref) + 1;
        ref_main = (vertical_mode ? top_ref : left_ref) + 1;
        
        // Move the reference pixels to start from the middle to the later half of
        // the tmp_ref, so there is room for negative indices.
        COPY_REF: for (int x = -1; x < width; ++x) {
#pragma HLS UNROLL
            tmp_ref[x + width] = ref_main[x];
        }
        // Get a pointer to block index 0 in tmp_ref.
        ref_main = &tmp_ref[width];
        
        // Extend the side reference to the negative indices of main reference.
        accum_t col_sample_disp = 128; // rounding for the ">> 8"
        ap_uint<16> inv_abs_sample_disp = modedisp2invsampledisp[abs_mode_disp];
        ap_int<8> most_negative_index = (width * sample_disp) >> 5;
        
        EXTEND_NEG: for (int x = -2; x >= most_negative_index; --x) {
#pragma HLS PIPELINE II=1
            col_sample_disp += inv_abs_sample_disp;
            ap_int<8> side_index = col_sample_disp >> 8;
            tmp_ref[x + width] = ref_side[side_index - 1];
        }
    }
    else {
        // sample_disp >= 0 means we don't need to refer to negative indices,
        // which means we can just use the references as is.
        ref_main = (vertical_mode ? top_ref : left_ref) + 1;
        ref_side = (vertical_mode ? left_ref : top_ref) + 1;
    }
    
    if (sample_disp != 0) {
        // The mode is not horizontal or vertical, we have to do interpolation.
        ap_int<16> delta_pos = 0;
        
        ANG_ROW: for (int y = 0; y < width; ++y) {
#pragma HLS PIPELINE II=1
            delta_pos += sample_disp;
            ap_int<8> delta_int = delta_pos >> 5;
            ap_int<8> delta_fract = delta_pos & (32 - 1);
            
            ANG_COL: for (int x = 0; x < width; ++x) {
#pragma HLS UNROLL
                if (delta_fract) {
                    // Do linear filtering
                    pixel_t ref1 = ref_main[x + delta_int];
                    pixel_t ref2 = ref_main[x + delta_int + 1];
                    pred[y][x] = ((32 - delta_fract) * ref1 + delta_fract * ref2 + 16) >> 5;
                }
                else {
                    // Just copy the integer samples
                    pred[y][x] = ref_main[x + delta_int];
                }
            }
        }
    }
    else {
        // Mode is horizontal or vertical, just copy the pixels.
        PERFECT_ROW: for (int y = 0; y < width; ++y) {
#pragma HLS PIPELINE II=1
            PERFECT_COL: for (int x = 0; x < width; ++x) {
#pragma HLS UNROLL
                pred[y][x] = ref_main[x];
            }
        }
    }
    
    // Flip the block if this is was a horizontal mode.
    if (!vertical_mode) {
        TRANSPOSE_I: for (int y = 0; y < width - 1; ++y) {
#pragma HLS PIPELINE II=1
            TRANSPOSE_J: for (int x = y + 1; x < width; ++x) {
#pragma HLS UNROLL factor=4
                pixel_t temp = pred[y][x];
                pred[y][x] = pred[x][y];
                pred[x][y] = temp;
            }
        }
    }
}

// Fast SAD calculation with tree reduction
cost_t calculate_sad_hls(
    const pixel_t orig[BLOCK_SIZE][BLOCK_SIZE],
    const pixel_t pred[BLOCK_SIZE][BLOCK_SIZE]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=orig complete dim=0
#pragma HLS ARRAY_PARTITION variable=pred complete dim=0

    accum_t sad = 0;
    
    SAD_ROW: for (int y = 0; y < BLOCK_SIZE; y++) {
#pragma HLS UNROLL
        SAD_COL: for (int x = 0; x < BLOCK_SIZE; x++) {
#pragma HLS UNROLL
            diff_t diff = (diff_t)orig[y][x] - (diff_t)pred[y][x];
            accum_t abs_diff = (diff < 0) ? (accum_t)(-diff) : (accum_t)diff;
            sad += abs_diff;
        }
    }
    
    return (cost_t)sad; // Raw SAD without scaling - matches CPU reference
}

// Efficient data packing/unpacking functions
void unpack_input_data(
    hls::stream<hls_input_packet_t> &input_stream,
    hls_input_t &unpacked_input
) {
#pragma HLS INLINE
    
    // Input requires 2 packets of 512 bits each (total 1024 bits for 784-bit data)
    // Packet 0: orig_block (512 bits)
    hls_input_packet_t packet0 = input_stream.read();
    axi_data_t data0 = packet0.data;
    
    // Unpack orig_block from first 512 bits
    UNPACK_ORIG: for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma HLS UNROLL
        UNPACK_ORIG_COL: for (int j = 0; j < BLOCK_SIZE; j++) {
#pragma HLS UNROLL
            int bit_pos = (i * BLOCK_SIZE + j) * 8;
            unpacked_input.orig_block[i][j] = data0.range(bit_pos + 7, bit_pos);
        }
    }
    
    // Packet 1: references (272 bits total, padded to 512 bits)
    hls_input_packet_t packet1 = input_stream.read();
    axi_data_t data1 = packet1.data;
    
    // Unpack left_ref (136 bits)
    UNPACK_LEFT: for (int i = 0; i < REF_SIZE; i++) {
#pragma HLS UNROLL
        int bit_pos = i * 8;
        unpacked_input.left_ref[i] = data1.range(bit_pos + 7, bit_pos);
    }
    
    // Unpack top_ref (136 bits, starting at bit 136)
    UNPACK_TOP: for (int i = 0; i < REF_SIZE; i++) {
#pragma HLS UNROLL
        int bit_pos = 136 + i * 8;
        unpacked_input.top_ref[i] = data1.range(bit_pos + 7, bit_pos);
    }
}

void pack_output_data(
    const hls_output_t &unpacked_output,
    hls::stream<hls_output_packet_t> &output_stream
) {
#pragma HLS INLINE
    
    // Output requires 2 packets (560 bits total fits in 2×512 = 1024 bits)
    hls_output_packet_t packet0, packet1;
    
    // Initialize side channels
    packet0.keep = -1; packet0.strb = -1; packet0.user = 0; packet0.id = 0; packet0.dest = 0;
    packet1.keep = -1; packet1.strb = -1; packet1.user = 0; packet1.id = 0; packet1.dest = 0;
    
    // Pack first 32 costs in packet0 (32 × 16 = 512 bits)
    PACK_COSTS_0: for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
        int bit_pos = i * 16;
        packet0.data.range(bit_pos + 15, bit_pos) = unpacked_output.costs[i];
    }
    packet0.last = 0;
    output_stream.write(packet0);
    
    // Pack remaining 3 costs in packet1 (3 × 16 = 48 bits, padded to 512)
    packet1.data = 0;
    PACK_COSTS_1: for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
        int bit_pos = i * 16;
        packet1.data.range(bit_pos + 15, bit_pos) = unpacked_output.costs[32 + i];
    }
    packet1.last = 1;
    output_stream.write(packet1);
}

// Main HLS top-level function with optimized AXI interface
void hls_intra_predict_top(
    hls::stream<hls_input_packet_t> &input_stream,
    hls::stream<hls_output_packet_t> &output_stream
) {
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Dataflow optimization
#pragma HLS DATAFLOW

    // Internal structures
    hls_input_t input;
    hls_output_t output;
    
    // Unpack input data from AXI stream
    unpack_input_data(input_stream, input);
    
    // Local arrays for filtered references
    pixel_t left_filtered[REF_SIZE];
    pixel_t top_filtered[REF_SIZE];
#pragma HLS ARRAY_PARTITION variable=left_filtered complete
#pragma HLS ARRAY_PARTITION variable=top_filtered complete
    
    // Filter references once
    filter_references_hls(input.left_ref, input.top_ref, left_filtered, top_filtered);
    
    // Process all modes sequentially (resource-efficient)
    MODE_LOOP: for (intra_mode_t mode = 0; mode < NUM_MODES; mode++) {
#pragma HLS PIPELINE II=1
        
        pixel_t pred[BLOCK_SIZE][BLOCK_SIZE];
#pragma HLS ARRAY_PARTITION variable=pred complete dim=0
        
        // Reference selection logic based on exact Kvazaar logic
        const pixel_t *left_to_use = input.left_ref;
        const pixel_t *top_to_use = input.top_ref;
        
        if (mode == 0) {
            // Planar uses filtered references
            left_to_use = left_filtered;
            top_to_use = top_filtered;
            planar_predict_hls(left_to_use, top_to_use, pred);
        }
        else if (mode == 1) {
            // DC uses unfiltered references
            dc_predict_hls(left_to_use, top_to_use, pred);
        }
        else {
            // Angular modes: choose filtered vs unfiltered based on distance from hor/ver
            ap_int<8> filter_threshold = kvz_intra_hor_ver_dist_thres[3 - 2]; // log2_width=3 for 8x8
            ap_int<16> temp_26 = (ap_int<16>)mode - 26;
            ap_int<16> temp_10 = (ap_int<16>)mode - 10;
            ap_int<8> dist_from_vert_or_hor_26 = (temp_26 < 0) ? (ap_int<8>)(-temp_26) : (ap_int<8>)(temp_26);
            ap_int<8> dist_from_vert_or_hor_10 = (temp_10 < 0) ? (ap_int<8>)(-temp_10) : (ap_int<8>)(temp_10);
            ap_int<8> dist_from_vert_or_hor = (dist_from_vert_or_hor_26 < dist_from_vert_or_hor_10) ? 
                                              dist_from_vert_or_hor_26 : dist_from_vert_or_hor_10;
            
            if (dist_from_vert_or_hor > filter_threshold) {
                left_to_use = left_filtered;
                top_to_use = top_filtered;
            }
            
            angular_predict_hls(left_to_use, top_to_use, mode, pred);
        }
        
        // Calculate cost
        output.costs[mode] = calculate_sad_hls(input.orig_block, pred);
    }
    
    // Pack and send output
    pack_output_data(output, output_stream);
}