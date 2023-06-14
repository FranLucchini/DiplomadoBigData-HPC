/** Sumador de vectores **/

/*
nvcc -o vecadd vecadd.cu -I/usr/local/cuda-5.0/samples/common/inc
*/

#include<stdio.h>
#include<cuda_runtime.h>
#include<helper_functions.h>
#include<helper_string.h>
#include<cublas_v2.h>

__global__ void vec_add (float *A, float *B, float *C, int N){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i>=N) return;
  C[i] = A[i] + B[i];
}

__host__ void vec_add_seq (float *A, float *B, float *C, int N){
  
  for(int i=0; i<N; i++) {
    C[i] = A[i] + B[i];
  }
}


int checkResult(float r, float *C, int N) {
  for(int i=0; i<N; i++) {
    if(C[i] != r) {
      return 0;
    }
  }
  return 1;
}
  
int main(int argc, char **argv) {

  int N=400;
  int nIter = 1000;

  if(argc > 1) {
    N = atoi(argv[1]); 
  }
  printf("Executing addition A[%d] + B[%d] = C[%d] ...\n", N, N, N);

  // Memoria host
  printf("Allocating host memory ...\n");
  float *A_h = new float[N];
  float *B_h = new float[N];
  float *C_h = new float[N];
  for(int i=0; i<N; i++) {
    A_h[i] = 1.6f;
    B_h[i] = 3.2f;
  }
  // Memoria device
  printf("Allocating device memory ...\n");
  float *A_d, *B_d, *C_d;
  cudaMalloc( (void**)&A_d, N*sizeof(float));
  cudaMalloc( (void**)&B_d, N*sizeof(float));
  cudaMalloc( (void**)&C_d, N*sizeof(float));
  
  //Eventos para timing
  cudaEvent_t gstart,gstop,mstart,mstop;
  cudaEventCreate(&gstart);
  cudaEventCreate(&gstop);
  cudaEventCreate(&mstart);
  cudaEventCreate(&mstop);

  // Copia de memoria host a device
  printf("Copying host -> device ...\n");
  cudaEventRecord(mstart);
  cudaMemcpy(A_d, A_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, N*sizeof(float), cudaMemcpyHostToDevice);



    
  int threads = 512;
  // Ejecucion de kernel de N/512 blocks de 512 threads
  dim3 nBlocks(N/threads+1);
  dim3 nThreads(threads);
  printf("Launching kernel ... nBlocks=%d, nThreads=%d\n", nBlocks.x, nThreads.x);
  cudaEventRecord(gstart, NULL);
  for(int k=0; k<nIter; k++) {
    vec_add<<<nBlocks,nThreads>>>(A_d, B_d, C_d, N);
  }
  cudaEventRecord(gstop, NULL);

  printf("Copying device -> host ...\n");
  // Recuperacion de resultados de device a host
  cudaMemcpy(C_h, C_d, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(mstop);

  printf("Checking results... ");
  if(checkResult(A_h[0]+B_h[0], C_h, N)) {
    printf("OK!\n");
  }
  else {
    printf("Errors :(\n");
  }
  
  // Compute and print the performance
  float msecTotal = 0.0f, msecWithCopy = 0.0f;
  cudaEventElapsedTime(&msecTotal, gstart, gstop);
  cudaEventElapsedTime(&msecWithCopy, mstart, mstop);
  float msecPerVecAdd = msecTotal / nIter;
  double flopsPerVecAdd = N;
  double gigaFlops = (flopsPerVecAdd * 1.0e-9f) / (msecPerVecAdd / 1000.0f);
  printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, TimeWithCopy= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerVecAdd,
        msecWithCopy,
        flopsPerVecAdd,
        nThreads.x * nThreads.y);


  printf("Computing result using host CPU...");
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);


  printf("Running on host ...\n");
  sdkStartTimer(&timer);
  for(int k=0; k<nIter; k++) {
    vec_add_seq(A_h, B_h, C_h, N);
  }
  sdkStopTimer(&timer);
  printf("Checking results... ");
  if(checkResult(A_h[0]+B_h[0], C_h, N)) {
    printf("OK!\n");
  }
  else {
    printf("Errors :(\n");
  }
  double msecTotalCPU = sdkGetTimerValue(&timer);
  float msecPerVecAddCPU = msecTotalCPU / nIter;
  printf("Time spent by CPU: %.3f msec\n", msecPerVecAddCPU);
  printf("Speedup: %.3f\n", msecPerVecAddCPU / msecPerVecAdd);

    
  // Free
  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}
