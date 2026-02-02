#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <limits>

#include "../tester/utils.h"

/**
 * @brief CUDA kernel to compute partial sums of diagonal elements for trace calculation
 * Uses warp shuffle for efficient reduction within warps
 *
 * @tparam T Data type of matrix elements
 * @param input Pointer to the flattened matrix data
 * @param partial_sums Output array to store partial sums from each thread block
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
template <typename T>
__global__ void traceKernel(const T* input, T* partial_sums, size_t rows, size_t cols) {
    size_t min_dim = min(rows, cols);

    // Shared memory for warp-level results
    __shared__ T warp_sums[8];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    T sum = 0;
    // Each thread processes multiple elements if needed
    for (size_t i = idx; i < min_dim; i += blockDim.x * gridDim.x) {
        sum += input[i * (cols + 1)];  // Diagonal elements are spaced by (cols+1)
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces all warp sums
    if (warp_id == 0) {
        // Load warp sum (only valid for threads < num_warps)
        int num_warps = blockDim.x / 32;
        sum = (lane < num_warps) ? warp_sums[lane] : T(0);

        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Write result of this block to global memory
        if (tid == 0) {
            partial_sums[blockIdx.x] = sum;
        }
    }
}

/**
 * @brief Computes trace of a matrix using CUDA.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in device memory.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param d_input A flattened matrix stored in device memory of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T traceCuda(const T* d_input, size_t rows, size_t cols) {
    size_t min_dim = min(rows, cols);
    if (min_dim == 0) return T(0);

    // Configure kernel launch parameters
    const int BLOCK_SIZE = 256;
    const int MAX_BLOCKS = 32;
    const int NUM_BLOCKS = min(MAX_BLOCKS, (int)((min_dim + BLOCK_SIZE - 1) / BLOCK_SIZE));

    // Allocate memory for partial sums from each block
    T* d_partial_sums;
    RUNTIME_CHECK(cudaMalloc(&d_partial_sums, NUM_BLOCKS * sizeof(T)));

    // Initialize partial sums to zero
    RUNTIME_CHECK(cudaMemset(d_partial_sums, 0, NUM_BLOCKS * sizeof(T)));

    // Launch kernel to compute partial sums
    traceKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        d_input, d_partial_sums, rows, cols);
    RUNTIME_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    std::vector<T> h_partial_sums(NUM_BLOCKS);
    RUNTIME_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums,
                             NUM_BLOCKS * sizeof(T), cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    T trace = T(0);
    for (int i = 0; i < NUM_BLOCKS; i++) {
        trace += h_partial_sums[i];
    }

    // Clean up
    RUNTIME_CHECK(cudaFree(d_partial_sums));

    return trace;
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (h_input.empty() || rows == 0 || cols == 0) return T(0);

    // Allocate device memory
    T* d_input;
    RUNTIME_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(T)));

    // Copy input data to device
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(),
                             h_input.size() * sizeof(T), cudaMemcpyHostToDevice));

    // Compute trace using CUDA
    T result = traceCuda(d_input, rows, cols);

    // Clean up
    RUNTIME_CHECK(cudaFree(d_input));

    return result;
}

// Helper functions for type conversion
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }

__device__ __forceinline__ float from_float(float x, float) { return x; }
__device__ __forceinline__ half from_float(float x, half) { return __float2half(x); }

/**
 * @brief Flash Attention kernel with online softmax algorithm
 * Single-threaded per query position for numerical accuracy
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    float scale, bool is_causal) {

    // Grid: (tgt_seq_len, query_heads, batch_size)
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int tgt_idx = blockIdx.x;

    // Only thread 0 does the work
    if (threadIdx.x != 0) return;

    if (batch_idx >= batch_size || head_idx >= query_heads || tgt_idx >= tgt_seq_len) {
        return;
    }

    // Grouped Query Attention: map query head to kv head
    int kv_head_idx = (head_idx * kv_heads) / query_heads;

    // Load query vector for this position
    int q_offset = batch_idx * tgt_seq_len * query_heads * head_dim +
                   tgt_idx * query_heads * head_dim +
                   head_idx * head_dim;

    // 1: Find max score for numerical stability
    float max_score = -INFINITY;

    for (int src_idx = 0; src_idx < src_seq_len; src_idx++) {
        // Apply causal mask: skip future positions
        if (is_causal && src_idx > tgt_idx) {
            continue;
        }

        int k_offset = batch_idx * src_seq_len * kv_heads * head_dim +
                      src_idx * kv_heads * head_dim +
                      kv_head_idx * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += to_float(Q[q_offset + d]) * to_float(K[k_offset + d]);
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // 2: Compute softmax and weighted sum
    float sum_exp = 0.0f;
    float output[128];
    for (int d = 0; d < head_dim; d++) output[d] = 0.0f;

    for (int src_idx = 0; src_idx < src_seq_len; src_idx++) {
        // Apply causal mask: skip future positions
        if (is_causal && src_idx > tgt_idx) {
            continue;
        }

        int k_offset = batch_idx * src_seq_len * kv_heads * head_dim +
                      src_idx * kv_heads * head_dim +
                      kv_head_idx * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += to_float(Q[q_offset + d]) * to_float(K[k_offset + d]);
        }
        score *= scale;

        int v_offset = batch_idx * src_seq_len * kv_heads * head_dim +
                      src_idx * kv_heads * head_dim +
                      kv_head_idx * head_dim;

        float exp_score = expf(score - max_score);
        sum_exp += exp_score;

        for (int d = 0; d < head_dim; d++) {
            output[d] += exp_score * to_float(V[v_offset + d]);
        }
    }

    // Write final result
    int o_offset = batch_idx * tgt_seq_len * query_heads * head_dim +
                   tgt_idx * query_heads * head_dim +
                   head_idx * head_dim;

    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int d = 0; d < head_dim; d++) {
            O[o_offset + d] = from_float(output[d] * inv_sum, T());
        }
    } else {
        for (int d = 0; d < head_dim; d++) {
            O[o_offset + d] = from_float(0.0f, T());
        }
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    // Allocate device memory
    T *d_q, *d_k, *d_v, *d_o;
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t v_size = batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim * sizeof(T);

    RUNTIME_CHECK(cudaMalloc(&d_q, q_size));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_size));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_size));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size));

    // Copy input data to device
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));

    // Compute scaling factor: 1/sqrt(head_dim)
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Configure kernel launch parameters
    // Grid: (target_seq_len, query_heads, batch_size)
    dim3 grid(target_seq_len, query_heads, batch_size);

    // Block size: single thread per query position for numerical accuracy
    int block_size = 1;

    // Launch kernel (no shared memory needed for single-threaded version)
    flashAttentionKernel<<<grid, block_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        scale, is_causal
    );

    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));

    // Free device memory
    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int batch_size, int target_seq_len, int src_seq_len,
  int query_heads, int kv_heads, int head_dim, bool is_causal);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int batch_size, int target_seq_len, int src_seq_len,
  int query_heads, int kv_heads, int head_dim, bool is_causal);
