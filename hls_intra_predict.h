#ifndef HLS_INTRA_PREDICT_H
#define HLS_INTRA_PREDICT_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

// HLS-optimized data types
typedef ap_uint<8> pixel_t;        // 8-bit pixel values
typedef ap_uint<16> cost_t;        // 16-bit for SAD costs
typedef ap_int<9> diff_t;          // 9-bit for differences (-255 to +255)
typedef ap_int<16> accum_t;        // 16-bit for accumulation
typedef ap_uint<8> intra_mode_t;   // 8-bit for mode index (0-34)

// Constants
#define BLOCK_SIZE 8
#define NUM_MODES 35
#define REF_SIZE 17

// Standard AXI data width - 512 bits for optimal DMA compatibility
typedef ap_uint<512> axi_data_t;

// Use ap_axiu to include side channels (TLAST, TKEEP, etc.)
// D=512, U=1, TI=1, TD=1
typedef ap_axiu<512, 1, 1, 1> hls_packet_t;

typedef hls_packet_t hls_input_packet_t;
typedef hls_packet_t hls_output_packet_t;

// Internal unpacked structures for processing (unchanged algorithm)
struct hls_input_t {
    pixel_t orig_block[BLOCK_SIZE][BLOCK_SIZE];
    pixel_t left_ref[REF_SIZE];
    pixel_t top_ref[REF_SIZE];
};

// Output structure - costs only to reduce memory usage
struct hls_output_t {
    cost_t costs[NUM_MODES];
};

// Top-level function for HLS synthesis with optimized AXI interface
void hls_intra_predict_top(
    hls::stream<hls_input_packet_t> &input_stream,
    hls::stream<hls_output_packet_t> &output_stream
);

#endif // HLS_INTRA_PREDICT_H