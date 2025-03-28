#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

// Copy all elements using threads in a 2D grid
__global__ void copy2d(int num_cols, int num_rows, double* dst, double* src) {
    // NOTE: compute row and col using
    // - threadIdx.x, threadIdx.y
    // - blockIdx.x, blockIdx.y
    // - blockDim.x, blockDim.y
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    // NOTE: Make sure there's no out-of-bounds access
    // row must be < number of rows
    // col must be < number of columns

    // We're computing 1D index from a 2D index and copying from src to dst
    const size_t index = row * num_cols + col;
    //dst[index] = src[index];
    dst[index] = 4.;
}

int main() {
    static constexpr size_t num_cols = 600;
    static constexpr size_t num_rows = 400;
    static constexpr size_t num_values = num_cols * num_rows;
    static constexpr size_t num_bytes = sizeof(double) * num_values;
    std::vector<double> x(num_values);
    std::vector<double> y(num_values, 8.0);

    // Initialise data
    for (size_t i = 0; i < num_values; i++) {
        x[i] = static_cast<double>(i) / 1000.0;
    }

    // NOTE: Allocate + copy initial values to GPU
    void *dx;
    void *dy;
    HIP_ERRCHK(hipMalloc(&dx, num_bytes));
    HIP_ERRCHK(hipMalloc(&dy, num_bytes));

    HIP_ERRCHK(hipMemcpy(dx, x.data(), num_bytes, hipMemcpyDefault));
    HIP_ERRCHK(hipMemcpy(dy, y.data(), num_bytes, hipMemcpyDefault));
    
    // NOTE: Define grid dimensions
    // Use dim3 structure for threads and blocks
    dim3 threads(8, 8, 1);
    dim3 blocks(2, 2, 1);

    // NOTE: launch the device kernel
    copy2d<<<blocks, threads>>>(num_cols, num_rows, (double*)dy, (double*)dx);

    // NOTE: Copy results back to the CPU vector y
    HIP_ERRCHK(hipMemcpy(y.data(), dy, num_bytes, hipMemcpyDefault));

    // NOTE: Free device memory
    HIP_ERRCHK(hipFree(dx));
    HIP_ERRCHK(hipFree(dy));

    // Check result of computation on the GPU
    printf("reference: %f %f %f %f ... %f %f\n", x[0], x[1], x[2], x[3], x[num_values - 2], x[num_values - 1]);
    printf("   result: %f %f %f %f ... %f %f\n", y[0], y[1], y[2], y[3], y[num_values - 2], y[num_values - 1]);

    double error = 0.0;
    for (size_t i = 0; i < num_values; i++) {
        error += abs(x[i] - y[i]);
    }

    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", x[42 * num_rows + 42]);
    printf("     result: %f at (42,42)\n", y[42 * num_rows + 42]);

    return 0;
}
