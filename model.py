"""
Matrix Multiplication Algorithms in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_cpu
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: C[m*N + n] = sum over k of A[m*K + k] * B[k*N + n]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;

            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

# Step 2 - max_abs_diff
#include <cmath>

float max_abs_diff(const float* a, const float* b, int n) {
    // TODO: largest |a[i] - b[i]| over n elements, 0 for n == 0
    float max = 0.0f;
    if (n==0) return max;
    float diff = 0.0f;
    for(int i=0; i<n; i++){
        diff = std::abs(a[i]-b[i]);
        if (diff > max) {
            max = diff;
        }
    }
    return max;
}

# Step 3 - matmul_naive_kernel (not yet solved)
# TODO: implement

# Step 4 - matmul_coalesced_kernel (not yet solved)
# TODO: implement

# Step 5 - time_launch_ms (not yet solved)
# TODO: implement

# Step 6 - matmul_tiled_kernel (not yet solved)
# TODO: implement

# Step 7 - matmul_tiled_1d_kernel (not yet solved)
# TODO: implement

# Step 8 - matmul_tiled_2d_kernel (not yet solved)
# TODO: implement

# Step 9 - matmul_vectorized_kernel (not yet solved)
# TODO: implement

# Step 10 - matmul_double_buffered_kernel (not yet solved)
# TODO: implement

# Step 11 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 12 - matmul_batched_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_splitk_kernel (not yet solved)
# TODO: implement

# Step 14 - gemv_kernel (not yet solved)
# TODO: implement

# Step 15 - matmul_bias_relu_kernel (not yet solved)
# TODO: implement

# Step 16 - matrix_addsub_kernel (not yet solved)
# TODO: implement

# Step 17 - strassen_one_level (not yet solved)
# TODO: implement

# Step 18 - csr_spmm_kernel (not yet solved)
# TODO: implement

# Step 19 - matmul_lower_triangular_kernel (not yet solved)
# TODO: implement

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

