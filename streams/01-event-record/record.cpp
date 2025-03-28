#include <cstdio>
#include <time.h>
#include <hip/hip_runtime.h>
#include <chrono>

#define get_mus(X) std::chrono::duration_cast<std::chrono::microseconds>(X).count()
#define chrono_clock std::chrono::high_resolution_clock::now()

/* A simple GPU kernel definition */
__global__ void kernel(int *d_a, int n_total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_total)
    d_a[idx] = idx;
}

/* The main function */
int main(){
  
  // Problem size
  // vary ~ 22 - 26
  constexpr int n_total = 1<<22;

  // Device grid sizes
  constexpr int blocksize = 256;
  constexpr int gridsize = (n_total - 1 + blocksize) / blocksize;

  // Allocate host and device memory
  int *a, *d_a;
  const int bytes = n_total * sizeof(int);
  hipHostMalloc((void**)&a, bytes); // host pinned
  hipMalloc((void**)&d_a, bytes);   // device pinned

  // Create events
  //#error create the required timing events here
  hipEvent_t start_kernel_event;
  hipEvent_t d2h_copy_event;
  hipEvent_t stop_event;
  hipEventCreate(&start_kernel_event);
  hipEventCreate(&d2h_copy_event);
  hipEventCreate(&stop_event);

  // Create stream
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Start timed GPU kernel and device-to-host copy
  // #error record the events somewhere across the below lines of code
  // #error such that you can get the timing for the kernel, the
  // #error memory copy, and the total combined time of these
  auto start_kernel_clock = chrono_clock;
  kernel<<<gridsize, blocksize, 0, stream>>>(d_a, n_total);
  hipEventRecord(start_kernel_event, stream);

  auto start_d2h_clock = chrono_clock;
  hipMemcpyAsync(a, d_a, bytes, hipMemcpyDeviceToHost, stream);
  hipEventRecord(d2h_copy_event, stream);

  auto stop_clock = chrono_clock;
  hipStreamSynchronize(stream);
  hipEventRecord(stop_event);

  // Exctract elapsed timings from event recordings
  // #error get the elapsed time from the timing events
  hipEventSynchronize(stop_event);
  float kernel_event_time;
  float d2h_event_time;
  float total_event_time;
  hipEventElapsedTime(&kernel_event_time, start_kernel_event, d2h_copy_event);
  hipEventElapsedTime(&d2h_event_time, d2h_copy_event, stop_event);
  hipEventElapsedTime(&total_event_time, start_kernel_event, stop_event);

  // Check that the results are right
  int error = 0;
  for(int i = 0; i < n_total; ++i){
    if(a[i] != i)
      error = 1;
  }

  // Print results
  if(error)
    printf("Results are incorrect!\n");
  else
    printf("Results are correct!\n");

  // Print event timings
  printf("Event timings:\n");
  // #error print event timings here
  printf(" %.3f ms - kernel\n", kernel_event_time);
  printf(" %.3f ms - device to host copy\n", d2h_event_time);
  printf(" %.3f ms - total time\n", total_event_time);

  // Print clock timings
  printf("clock_t timings:\n");
  printf("  %.3f ms - kernel\n", 1e-3 * (double)get_mus(start_d2h_clock - start_kernel_clock));
  printf("  %.3f ms - device to host copy\n", 1e-3 * (double)get_mus(stop_clock - start_d2h_clock));
  printf("  %.3f ms - total time\n", 1e-3 * (double)get_mus(stop_clock - start_kernel_clock));

  // Destroy Stream
  hipStreamDestroy(stream);

  // Destroy events
  // #error destroy events here
  hipEventDestroy(start_kernel_event);
  hipEventDestroy(d2h_copy_event);
  hipEventDestroy(stop_event);

  // Deallocations
  hipFree(d_a); // Device
  hipHostFree(a); // Host
}
