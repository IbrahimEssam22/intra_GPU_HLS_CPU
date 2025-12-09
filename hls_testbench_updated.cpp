#include "hls_intra_predict.h"
#include "allMats.c"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== HLS Intra Prediction Testbench (AXI Stream) ===" << std::endl;
    std::cout << "Testing with new 512-bit AXI Stream interface" << std::endl << std::endl;
    
    // Create HLS streams for new interface
    hls::stream<hls_input_packet_t> input_stream("input_stream");
    hls::stream<hls_output_packet_t> output_stream("output_stream");
    
    // Prepare input data packets (simplified - no .data field)
    hls_input_packet_t packet0, packet1;
    
    // Initialize packet 0
    packet0.data = 0;
    packet0.keep = -1;
    packet0.strb = -1;
    packet0.user = 0;
    packet0.id = 0;
    packet0.dest = 0;
    packet0.last = 0;

    // Initialize packet 1
    packet1.data = 0;
    packet1.keep = -1;
    packet1.strb = -1;
    packet1.user = 0;
    packet1.id = 0;
    packet1.dest = 0;
    packet1.last = 1;
    
    std::cout << "Preparing input data packets..." << std::endl;
    
    // Pack input data
    // Packet 0: Original block (64 pixels × 8 bits = 512 bits)
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int bit_pos = (i * BLOCK_SIZE + j) * 8;
            packet0.data.range(bit_pos + 7, bit_pos) = test_block[i][j];
        }
    }
    
    // Packet 1: References (17+17 pixels × 8 bits = 272 bits, padded to 512)
    // Pack left_ref (136 bits)
    for (int i = 0; i < REF_SIZE; i++) {
        int bit_pos = i * 8;
        packet1.data.range(bit_pos + 7, bit_pos) = left_ref[i];
    }
    
    // Pack top_ref (136 bits, starting at bit 136)
    for (int i = 0; i < REF_SIZE; i++) {
        int bit_pos = 136 + i * 8;
        packet1.data.range(bit_pos + 7, bit_pos) = top_ref[i];
    }
    
    // Write input packets to stream
    input_stream.write(packet0);
    input_stream.write(packet1);
    
    std::cout << "Input data packed into 2 × 512-bit packets" << std::endl;
    std::cout << "Packet 0: Original block (512 bits)" << std::endl;
    std::cout << "Packet 1: References + padding (272 + 240 bits)" << std::endl << std::endl;
    
    // Call HLS function
    std::cout << "Calling HLS function..." << std::endl;
    hls_intra_predict_top(input_stream, output_stream);
    
    // Read output packets
    std::cout << "Reading output data..." << std::endl;
    hls_output_packet_t out_packet0 = output_stream.read();
    hls_output_packet_t out_packet1 = output_stream.read();
    
    // Unpack output data
    cost_t hls_costs[NUM_MODES];
    
    // Unpack first 32 costs from packet 0
    for (int i = 0; i < 32; i++) {
        int bit_pos = i * 16;
        hls_costs[i] = out_packet0.data.range(bit_pos + 15, bit_pos);
    }
    
    // Unpack remaining 3 costs from packet 1
    for (int i = 0; i < 3; i++) {
        int bit_pos = i * 16;
        hls_costs[32 + i] = out_packet1.data.range(bit_pos + 15, bit_pos);
    }
    
    std::cout << "Output data unpacked from 2 × 512-bit packets" << std::endl;
    std::cout << "Packet 0: Costs 0-31 (512 bits)" << std::endl;
    std::cout << "Packet 1: Costs 32-34 + padding (48 + 464 bits)" << std::endl << std::endl;
    
    // Calculate expected costs from reference predictions
    std::cout << "Calculating reference costs from prediction matrices..." << std::endl;
    cost_t expected_costs[NUM_MODES];
    
    for (int mode = 0; mode < NUM_MODES; mode++) {
        int sad = 0;
        for (int y = 0; y < BLOCK_SIZE; y++) {
            for (int x = 0; x < BLOCK_SIZE; x++) {
                int diff = (int)test_block[y][x] - (int)reference_predictions[mode][y][x];
                sad += (diff < 0) ? -diff : diff;
            }
        }
        expected_costs[mode] = (cost_t)sad;
    }
    
    // Compare with expected results
    std::cout << "=== Results Comparison ===" << std::endl;
    std::cout << "Mode | HLS Cost | Expected | Match" << std::endl;
    std::cout << "-----+----------+----------+------" << std::endl;
    
    bool all_match = true;
    for (int mode = 0; mode < NUM_MODES; mode++) {
        bool match = (hls_costs[mode] == expected_costs[mode]);
        all_match &= match;
        
        std::cout << std::setw(4) << mode << " | "
                  << std::setw(8) << hls_costs[mode] << " | "
                  << std::setw(8) << expected_costs[mode] << " | "
                  << (match ? "✓" : "✗") << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    if (all_match) {
        std::cout << "✅ ALL TESTS PASSED!" << std::endl;
        std::cout << "HLS implementation matches reference exactly." << std::endl;
    } else {
        std::cout << "❌ SOME TESTS FAILED!" << std::endl;
        std::cout << "Check the implementation for discrepancies." << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Interface Summary ===" << std::endl;
    std::cout << "Input:  2 × 512-bit AXI Stream packets (1024 bits total)" << std::endl;
    std::cout << "Output: 2 × 512-bit AXI Stream packets (1024 bits total)" << std::endl;
    std::cout << "Data efficiency: 784/1024 = 76.6% (input), 560/1024 = 54.7% (output)" << std::endl;
    std::cout << "Standard AXI width: ✅ (No converters needed in Vivado)" << std::endl;
    
    return all_match ? 0 : 1;
}