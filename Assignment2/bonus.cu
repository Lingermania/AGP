
/*
//Serial version


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEED 921
#define NUM_ITER 1000000000

int main(int argc, char * argv[]) {
  int count = 0;
  double x, y, z, pi;
  srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!    // Calculate PI following a Monte Carlo method 
  for (int iter = 0; iter < NUM_ITER; iter++) { // Generate random (X,Y) points 
    x = (double) random() / (double) RAND_MAX;
    y = (double) random() / (double) RAND_MAX;
    z = sqrt((x * x) + (y * y));
    // Check if point is in unit circle       
    if (z <= 1.0) {
      count++;
    }
  } // Estimate Pi and display the result 
  pi = ((double) count / (double) NUM_ITER) * 4.0;
  printf("The result is %f\n", pi);
  return 0;
}
*/

//SAXPY - Single-Precision A*X Plus Y

#include <stdio.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <curand.h>

#define BLOCK_SIZE 256
#define NUM_ITER 1000
#define ARRAY_SIZE 10000

__global__ void pi_kernel(int *count, curandState *states){

    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < ARRAY_SIZE){
        curand_init(i, i, 0, &states[i]);
        double x, y, z; 
        for(int j = 0; j < NUM_ITER; j++){
            x = curand_uniform(&states[i]);
            y = curand_uniform(&states[i]);
            z = sqrt((x * x) + (y * y));
            if (z <= 1.0){
                count[i]++;
            }
        }
    }
}



int main(){

    
    int *count_d = 0;
    int count_h[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) { count_h[i] = 0; }

    int TB = (ARRAY_SIZE + BLOCK_SIZE -1)/BLOCK_SIZE;

    curandState *dev_random;
    cudaMalloc((void**)&dev_random, BLOCK_SIZE*TB*sizeof(curandState));
    cudaMalloc(&count_d, ARRAY_SIZE*sizeof(int));
    cudaMemcpy(count_d, count_h, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice);
    pi_kernel <<<TB, BLOCK_SIZE>>>(count_d, dev_random);
    cudaDeviceSynchronize();

    cudaMemcpy(count_h, count_d, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    int count = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) { count += count_h[i]; }
    //printf("%d, %d", count, ARRAY_SIZE*NUM_ITER);
    double pi = ((double) count / (double) (ARRAY_SIZE*NUM_ITER)) * 4.0;
    printf("The result is %f\n", pi);
    return 0;


}
