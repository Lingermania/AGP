//SAXPY - Single-Precision A*X Plus Y

#include <stdio.h>
#define TPB 100000
#define ARRAY_SIZE 100000



__global__ void particle_kernel(float4 *particles){
    
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    float4 particle = particles[i];
    particle.w *= (1.0 + 1e-7); //some exponential update rule
    particle.x += particle.w;
    particle.y += particle.w;
    particle.z += particle.w;

    //printf("GPU %d: %f, %f, %f, %f \n",i, particle.x, particle.y, particle.z, particle.w);
}

__host__ void verify(float4 *particles){
    for(int i = 0; i < ARRAY_SIZE; i++){
        float4 particle = particles[i];
        particle.w *= (1.0 + 1e-7); //some exponential update rule
        particle.x += particle.w;
        particle.y += particle.w;
        particle.z += particle.w;

        //printf("CPU %d: %f, %f, %f, %f \n",i, particle.x, particle.y, particle.z, particle.w);
    }
}

__host__ void compare(float4 *a, float4 *b, float moe){
    //Compare a to b using a euclidian distance metric s.t. |a - b| <= moe

    for (int i = 0; i < ARRAY_SIZE; i++){
        float len = sqrt(pow((a[i].x - b[i].x), 2) + pow((a[i].y - b[i].y), 2) + pow((a[i].z - b[i].z), 2) + pow((a[i].w - b[i].w), 2));
        //printf("%d: %f, %f, %f, %f \n",i, a[i].x, a[i].y, a[i].z, a[i].w);
        //printf("%d: %f, %f, %f, %f \n",i, b[i].x, b[i].y, b[i].z, b[i].w);
        //printf("%f", len);
        if (len >= moe){
            printf("Comparison failed w. %f accuracy.\n", moe);
            return;
        }
    }
    printf("Comparison successful w. %f accuracy.\n", moe);
}



int main(){
    //Host address
    float4 *particles_h = 0, *intermediate_h;

    //Device addresses
    float4 *particles_d = 0;


    //Allocate host memory
    particles_h    = (float4 *) malloc(ARRAY_SIZE*sizeof(float4));
    intermediate_h = (float4 *) malloc(ARRAY_SIZE*sizeof(float4));

    // Allocate device memory
    cudaMalloc(&particles_d, ARRAY_SIZE*sizeof(float4));


    //Populate host particles w. some uniformly distributed numbers from [0,1]
    for(int i = 0; i < ARRAY_SIZE; i++){
        particles_h[i] = make_float4(1,2,3,4);//make_float4((rand() % 100)/100.0, (rand() % 100)/100.0, (rand() % 100)/100.0, (rand() % 100)/100.0);
    }


    //Copy particles_h to device memory
    cudaMemcpy(particles_d, particles_h, ARRAY_SIZE*sizeof(float4), cudaMemcpyHostToDevice);

    //Launch kernel
    printf("Computing simulation on the GPU...\n");
    particle_kernel <<<ARRAY_SIZE/TPB, TPB>>>(particles_d);

    //Sync
    cudaDeviceSynchronize();
    printf("Done!\n");
    
    //Run host verification
    printf("Computing simulation on the CPU...\n");
    verify(particles_h);
    printf("Done!\n");

    //Copy yd from device to xh, then compare that xh and yh are equal
    printf("Copying from device and comparing the output for device and host\n");
    cudaMemcpy(intermediate_h, particles_d, ARRAY_SIZE*sizeof(float4), cudaMemcpyDeviceToHost);
    
    compare(particles_h, intermediate_h, 1e-6);

    cudaFree(particles_d);
    free(particles_h); free(intermediate_h);
    return 0;
}
