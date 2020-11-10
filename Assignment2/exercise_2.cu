//SAXPY - Single-Precision A*X Plus Y

#include <stdio.h>
#define TPB 256
#define ARRAY_SIZE 10000

/*
__device__ float ax(float x, float a){
    return a*x;
}
*/

__global__ void saxpyKernel(float *x, float *y, float a){
    
    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    y[i] += x[i]*a;
    
}

__host__ void verify(float *x, float *y, float a, int N){
    /*
    Host program to verify device computations
    */

    for(int i = 0; i < N; i++){
        y[i] += x[i]*a;
    }
}

int main(){
    //Host addresses
    float *xh = 0, *yh = 0;

    //Device addresses
    float *xd = 0, *yd = 0;

    const float a = 0.5;
    
    // Allocate device memory
    cudaMalloc(&xd, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&yd, ARRAY_SIZE*sizeof(float));

    // Allocate host memory
    xh = (float *) malloc(ARRAY_SIZE*sizeof(float));
    yh = (float *) malloc(ARRAY_SIZE*sizeof(float));

    //Set some values
    for(int i = 0; i < ARRAY_SIZE; i++){
        xh[i] = i;
        yh[i] = i + 1;
    }

    //Copy memory to device
    cudaMemcpy(xd, xh, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(yd, yh, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    //Launch kernel
    printf("Computing SAXPY on the GPU...\n");
    saxpyKernel <<<(ARRAY_SIZE + TPB-1)/TPB, TPB>>>(xd, yd, a);

    //Sync
    cudaDeviceSynchronize();
    printf("Done!\n");
    
    //Run host verification
    printf("Computing SAXPY on the CPU...\n");
    verify(xh, yh, a, ARRAY_SIZE);
    printf("Done!\n");

    //Copy yd from device to xh, then compare that xh and yh are equal
    printf("Copying from device and comparing the output for device and host\n");
    cudaMemcpy(xh, yd, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    bool eq = true;
    for(int i = 0; i < ARRAY_SIZE; i++){
        if (abs(xh[i] - yh[i]) >= 1e-6){
            eq = false;
            break;
        }
        //printf("idx %6d: GPU = %f --- CPU = %f", i, xh[i], yh[i]);
    }

    if (eq){
        printf("Successfully compared everything within a 1e-6 error margin.\n");
    }
    else{
        printf("Comparison failed for 1e-6 error margin.\n");
    }
    //Free memory
    cudaFree(xd); cudaFree(yd);
    free(xh); free(yh);
    return 0;
}
