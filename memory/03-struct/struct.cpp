#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* Example struct to practise copying structs with pointers to device memory */
typedef struct
{
  float *x;
  int *idx;
  int size;
} Example;

/* GPU kernel definition */
__global__ void hipKernel(Example* const d_ex)
{
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread < d_ex->size)
  {
    printf("x[%d]: %.2f, idx[%d]:%d/%d \n", thread, d_ex->x[thread], thread, d_ex->idx[thread], d_ex->size - 1);
  }
}

/* Run on host */
void runHost()
{
  // Allocate host struct
  Example *ex;
  ex = (Example*)malloc(sizeof(Example));
  ex->size = 10;

  // Allocate host struct members
  ex->x = (float*)malloc(ex->size * sizeof(float));
  ex->idx = (int*)malloc(ex->size * sizeof(int));

  // Initialize host struct members
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Print struct values from host
  printf("\nHost:\n");
  for(int i = 0; i < ex->size; i++)
  {
    printf("x[%d]: %.2f, idx[%d]:%d/%d \n", i, ex->x[i], i, ex->idx[i], ex->size - 1);
  }

  // Free host struct
  free(ex->x);
  free(ex->idx);
  free(ex);
}

/* Run on device using Unified Memory */
void runDeviceUnifiedMem()
{
  Example *ex;
  /* #error Allocate struct using Unified Memory */
  HIP_CHECK(hipMallocManaged((void**) &ex, sizeof(Example)));

  ex->size = 10;
  /* #error Allocate struct members using Unified Memory */
  HIP_CHECK(hipMallocManaged((void**) &(ex->x), sizeof(float) * ex->size));
  HIP_CHECK(hipMallocManaged((void**) &(ex->idx), sizeof(int) * ex->size));


  // Initialize struct from host
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  dim3 grid(1,1,1);
  dim3 block(16,16,1);
/* #error Print struct values from device by calling hipKernel() */
  hipKernel<<<grid, block, 0, 0>>>(ex);
  printf("\nDevice (UnifiedMem):\n");

  HIP_CHECK(hipFree(ex->x)); HIP_CHECK(hipFree(ex->idx)); HIP_CHECK(hipFree(ex));
}

/* Create the device struct (needed for explicit memory management) */
Example* createDeviceExample(Example *ex)
{
  //#error Allocate device struct
  Example *d_ex;
  HIP_CHECK(hipMalloc((void**) &d_ex, sizeof(Example)));

  //#error Allocate device struct members
  HIP_CHECK(hipMalloc((void**)&(d_ex->x), sizeof(float) * ex->size));
  HIP_CHECK(hipMalloc((void**)&(d_ex->idx), sizeof(int) * ex->size));

  //#error Copy arrays pointed by the struct members from host to device
  HIP_CHECK(hipMemcpy(d_ex->x, ex->x, sizeof(float) * ex->size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ex->idx, ex->idx, sizeof(int) * ex->size, hipMemcpyHostToDevice));

  //#error Copy struct members from host to device
  HIP_CHECK(hipMemcpy(&(d_ex->size), &(ex->size), sizeof(int), hipMemcpyHostToDevice));

  //#error Return device struct
  return d_ex;
}

/* Free the device struct (needed for explicit memory management) */
void freeDeviceExample(Example *d_ex)
{
  //#error Copy struct members (pointers) from device to host
  float* d_x;
  int* d_idx;
  HIP_CHECK(hipMemcpy(&d_x, &(d_ex->x), sizeof(float*), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&d_idx, &(d_ex->idx), sizeof(int*), hipMemcpyDeviceToHost));

  //#error Free device struct members
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_idx));

  //#error Free device struct
  HIP_CHECK(hipFree(d_ex));
}

/* Run on device using Explicit memory management */
void runDeviceExplicitMem()
{
  //#error Allocate host struct
  Example *ex;
  ex = (Example*)malloc(sizeof(Example));
  ex->size = 10;

  //#error Allocate host struct members
  ex->x = (float*)malloc(ex->size * sizeof(float));
  ex->idx=(int*)malloc(ex->size * sizeof(int));

  // Initialize host struct
  for(int i = 0; i < ex->size; i++)
  {
    ex->x[i] = (float)i;
    ex->idx[i] = i;
  }

  // Allocate device struct and copy values from host to device
  Example *d_ex = createDeviceExample(ex);

  dim3 grid(1,1,1);
  dim3 block(16,16,1);
  //#error Print struct values from device by calling hipKernel()
  hipKernel<<<grid, block, 0, 0>>>(d_ex);
  printf("\nDevice (ExplicitMem):\n");

  // Free device struct
  freeDeviceExample(d_ex);

  //#error Free host struct
  free(ex->x);
  free(ex->idx);
  free(ex);
}

/* The main function */
int main(int argc, char* argv[])
{
  runHost();
  runDeviceUnifiedMem();
  runDeviceExplicitMem();
}
