#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief CUDA kernel to compute partial sums of diagonal elements for trace calculation
 * 
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
    
    // Use byte-based shared memory and cast to appropriate type
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    T sum = 0;
    // Each thread processes multiple elements if needed
    for (size_t i = idx; i < min_dim; i += blockDim.x * gridDim.x) {
        sum += input[i * (cols + 1)];  // Diagonal elements are spaced by (cols+1)
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result of this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Computes the trace of a matrix using CUDA.
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
    traceKernel<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(T)>>>(
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
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
