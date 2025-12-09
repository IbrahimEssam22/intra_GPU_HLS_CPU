#ifndef INTRA_PREDICT_GPU_H
#define INTRA_PREDICT_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_mode_major_intra(
    int num_blocks,
    const int* h_l_raw, const int* h_t_raw,
    const int* h_l_filt, const int* h_t_filt,
    unsigned char* h_output
);

#ifdef __cplusplus
}
#endif

#endif // INTRA_PREDICT_GPU_H
