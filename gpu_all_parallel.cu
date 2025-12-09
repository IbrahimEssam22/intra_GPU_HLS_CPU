#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8
#define REF_SIZE 17
#define THREADS_PER_BLOCK 256

__constant__ int8_t c_modedisp2sampledisp[9] = { 0, 2, 5, 9, 13, 17, 21, 26, 32 };
__constant__ int16_t c_modedisp2invsampledisp[9] = { 0, 4096, 1638, 910, 630, 482, 390, 315, 256 };

__device__ __forceinline__ int clamp_pixel(int val) {
    return (val < 0) ? 0 : (val > 255 ? 255 : val);
}

__device__ __forceinline__ void store_uchar4(unsigned char* ptr, uchar4 val) {
    *reinterpret_cast<uchar4*>(ptr) = val;
}

// --- ALL MODES PER BLOCK KERNEL ---
// Grid X dimension = num_blocks
// Grid Y dimension = 1
// Threads = 64 (One thread per Mode, with some idle threads)
__global__ void kernel_all_modes_per_block(
    const int* __restrict__ all_left_raw,
    const int* __restrict__ all_top_raw,
    const int* __restrict__ all_left_filt,
    const int* __restrict__ all_top_filt,
    unsigned char* __restrict__ output_buffer,
    const int num_blocks
) {
    // Shared memory for references (Common to all modes in this block)
    __shared__ int s_left_raw[REF_SIZE];
    __shared__ int s_top_raw[REF_SIZE];
    __shared__ int s_left_filt[REF_SIZE];
    __shared__ int s_top_filt[REF_SIZE];

    int block_idx = blockIdx.x;
    int tid = threadIdx.x; // 0..63
    
    if (block_idx >= num_blocks) return;

    // 1. Cooperative Load
    if (tid < REF_SIZE) {
        int input_offset = block_idx * REF_SIZE;
        s_left_raw[tid] = all_left_raw[input_offset + tid];
        s_top_raw[tid] = all_top_raw[input_offset + tid];
        s_left_filt[tid] = all_left_filt[input_offset + tid];
        s_top_filt[tid] = all_top_filt[input_offset + tid];
    }
    __syncthreads();

    // 2. Each thread handles one Mode
    int mode = tid;
    if (mode >= 35) return;

    unsigned char* out_ptr = output_buffer + (block_idx * 35 + mode) * 64;

    // --- MODE 0: PLANAR ---
    if (mode == 0) {
        int tr = s_top_filt[BLOCK_SIZE + 1];
        int bl = s_left_filt[BLOCK_SIZE + 1];

        #pragma unroll
        for (int y = 0; y < 8; y++) {
            uchar4 res1, res2;
            for (int x = 0; x < 4; x++) {
                int hor = (7 - x) * s_left_filt[y + 1] + (x + 1) * tr;
                int ver = (7 - y) * s_top_filt[x + 1] + (y + 1) * bl;
                ((unsigned char*)&res1)[x] = (hor + ver + 8) >> 4;
            }
            for (int x = 4; x < 8; x++) {
                int hor = (7 - x) * s_left_filt[y + 1] + (x + 1) * tr;
                int ver = (7 - y) * s_top_filt[x + 1] + (y + 1) * bl;
                ((unsigned char*)&res2)[x-4] = (hor + ver + 8) >> 4;
            }
            store_uchar4(out_ptr + y * 8, res1);
            store_uchar4(out_ptr + y * 8 + 4, res2);
        }
        return;
    }

    // --- MODE 1: DC ---
    if (mode == 1) {
        int sum = 0;
        for (int i = 1; i <= 8; i++) sum += s_left_raw[i] + s_top_raw[i];
        int dc_val = (sum + 8) >> 4;

        int l1 = s_left_raw[1];
        int t1 = s_top_raw[1];

        #pragma unroll
        for (int y = 0; y < 8; y++) {
            uchar4 res1, res2;
            for (int x = 0; x < 4; x++) {
                int val = dc_val;
                if (y==0 && x==0) val = (l1 + 2*dc_val + t1 + 2) >> 2;
                else if (y==0) val = (s_top_raw[x+1] + 3*dc_val + 2) >> 2;
                else if (x==0) val = (s_left_raw[y+1] + 3*dc_val + 2) >> 2;
                ((unsigned char*)&res1)[x] = val;
            }
            for (int x = 4; x < 8; x++) {
                int val = dc_val;
                if (y==0) val = (s_top_raw[x+1] + 3*dc_val + 2) >> 2;
                ((unsigned char*)&res2)[x-4] = val;
            }
            store_uchar4(out_ptr + y*8, res1);
            store_uchar4(out_ptr + y*8 + 4, res2);
        }
        return;
    }

    // --- ANGULAR ---
    bool vertical_mode = (mode >= 18);
    int mode_disp = vertical_mode ? (mode - 26) : (10 - mode);
    int abs_mode_disp = (mode_disp < 0) ? -mode_disp : mode_disp;
    int sample_disp = c_modedisp2sampledisp[abs_mode_disp];
    if (mode_disp < 0) sample_disp = -sample_disp;

    int dist_ver = (mode - 26); if(dist_ver < 0) dist_ver = -dist_ver;
    int dist_hor = (mode - 10); if(dist_hor < 0) dist_hor = -dist_hor;
    int dist = (dist_ver < dist_hor) ? dist_ver : dist_hor;

    const int* p_left = (dist > 7) ? s_left_filt : s_left_raw;
    const int* p_top = (dist > 7) ? s_top_filt : s_top_raw;
    const int* ref_main = (vertical_mode ? p_top : p_left) + 1;
    const int* ref_side = (vertical_mode ? p_left : p_top) + 1;

    int ref_arr[33];
    int* ref_ptr = &ref_arr[1];
    for(int k=-1; k<17; k++) ref_ptr[k] = ref_main[k];

    if (sample_disp < 0) {
        int inv_sample_disp = c_modedisp2invsampledisp[abs_mode_disp];
        int col_sample_disp = 128;
        int most_neg = (8 * sample_disp) >> 5;
        for (int rx = -2; rx >= most_neg; --rx) {
            col_sample_disp += inv_sample_disp;
            int side_index = col_sample_disp >> 8;
            ref_ptr[rx] = ref_side[side_index - 1];
        }
    }

    if (sample_disp == 0) {
        #pragma unroll
        for (int y = 0; y < 8; y++) {
            uchar4 res1, res2;
            for(int x=0; x<4; x++) ((unsigned char*)&res1)[x] = ref_ptr[x];
            for(int x=4; x<8; x++) ((unsigned char*)&res2)[x-4] = ref_ptr[x];

            if (vertical_mode) {
                store_uchar4(out_ptr + y*8, res1);
                store_uchar4(out_ptr + y*8 + 4, res2);
            } else {
                for(int x=0; x<8; x++) out_ptr[x*8+y] = ref_ptr[x];
            }
        }
    } else {
        #pragma unroll
        for (int y = 0; y < 8; y++) {
            int delta_pos = (y + 1) * sample_disp;
            int d_int = delta_pos >> 5;
            int d_fract = delta_pos & 31;

            uchar4 res1, res2;
            for(int x=0; x<4; x++) {
                int val;
                if(d_fract) val = ((32-d_fract)*ref_ptr[x+d_int] + d_fract*ref_ptr[x+d_int+1] + 16)>>5;
                else val = ref_ptr[x+d_int];
                ((unsigned char*)&res1)[x] = val;
            }
            for(int x=4; x<8; x++) {
                int val;
                if(d_fract) val = ((32-d_fract)*ref_ptr[x+d_int] + d_fract*ref_ptr[x+d_int+1] + 16)>>5;
                else val = ref_ptr[x+d_int];
                ((unsigned char*)&res2)[x-4] = val;
            }

            if (vertical_mode) {
                store_uchar4(out_ptr + y*8, res1);
                store_uchar4(out_ptr + y*8 + 4, res2);
            } else {
                for(int x=0; x<8; x++) {
                    if (x<4) out_ptr[x*8+y] = ((unsigned char*)&res1)[x];
                    else out_ptr[x*8+y] = ((unsigned char*)&res2)[x-4];
                }
            }
        }
    }
}

extern "C" void launch_mode_major_intra(
    int num_blocks,
    const int* h_l_raw, const int* h_t_raw,
    const int* h_l_filt, const int* h_t_filt,
    unsigned char* h_output
) {
    int *d_l_raw, *d_t_raw, *d_l_filt, *d_t_filt;
    unsigned char *d_out;

    size_t input_bytes = num_blocks * REF_SIZE * sizeof(int);
    size_t out_bytes = num_blocks * 35 * 64 * sizeof(unsigned char);

    cudaMalloc(&d_l_raw, input_bytes);
    cudaMalloc(&d_t_raw, input_bytes);
    cudaMalloc(&d_l_filt, input_bytes);
    cudaMalloc(&d_t_filt, input_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_l_raw, h_l_raw, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_raw, h_t_raw, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l_filt, h_l_filt, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_filt, h_t_filt, input_bytes, cudaMemcpyHostToDevice);

    // KEY CHANGE: Grid Dimensions
    // X = Number of Blocks chunks
    // Y = 1 (All modes in one block)
    // Threads = 64 (One thread per Mode, with some idle threads)
    dim3 threads(64);
    dim3 grid(num_blocks, 1);

    kernel_all_modes_per_block<<<grid, threads>>>(
        d_l_raw, d_t_raw, d_l_filt, d_t_filt, d_out, num_blocks
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_l_raw); cudaFree(d_t_raw); cudaFree(d_l_filt); cudaFree(d_t_filt); cudaFree(d_out);
}
