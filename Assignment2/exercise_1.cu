//cuda version of test.c

#include <stdio.h>
#define N 256
#define TPB 256


__global__ void helloWorldKernel(){
    
    const int   i = blockIdx.x*blockDim.x + threadIdx.x;

    
    printf("Hello World! My threadId is %2d\n", i);
    
}


int main(){

    helloWorldKernel <<<N/TPB, TPB>>>();

    return 0;
}
