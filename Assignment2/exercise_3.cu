//SAXPY - Single-Precision A*X Plus Y

#include <stdio.h>
#include <sys/time.h>

#define BLOCK_SIZE 400
#define NUM_PARTICLES 100000
#define NUM_ITERS 1000

struct particle{
    float3 pos;
    float3 v;
};
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__device__ __host__ void update(struct particle *p){
    p->v.x *= (1.0 + 1e-1);
    p->v.y *= (1.0 + 1e-1);
    p->v.z *= (1.0 + 1e-1);

    p->pos.x += p->v.x;
    p->pos.y += p->v.y;
    p->pos.z += p->v.z;
}
__global__ void particle_kernel(struct particle *particles){
    
    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < NUM_PARTICLES){
        update(&particles[i]);
    }


}

__host__ void verify(struct particle *particles){
    for(int i = 0; i < NUM_PARTICLES; i++){
        update(&particles[i]);
    }
}

__host__ void compare(struct particle *a, struct particle *b, float moe){
    //Compare a to b using a distance metric on particle points s.t. |a - b| <= moe

    for (int i = 0; i < NUM_PARTICLES; i++){
        float len = sqrt(pow((a[i].pos.x - b[i].pos.x), 2) + pow((a[i].pos.y - b[i].pos.y), 2) + pow((a[i].pos.z - b[i].pos.z), 2));
        if (len >= moe){
            printf("Comparison failed w. %f accuracy.\n", moe);
            return;
        }
    }
    printf("Comparison successful w. %f accuracy.\n", moe);
}


int main(){
    //Host address, intermediate memory and device adderss
    particle particles_h[NUM_PARTICLES], intermediate_h[NUM_PARTICLES];
    particle *particles_d = 0;


    //Allocate host memory
    //particles_h    = (particle *) malloc(NUM_PARTICLES*sizeof(particle));
    //intermediate_h = (particle *) malloc(ARRANUM_PARTICLESY_SIZE*sizeof(particle));

    // Allocate device memory
    cudaMalloc(&particles_d, NUM_PARTICLES*sizeof(particle));


    //Populate host particles w. some uniformly distributed numbers from [0,1]
    for(int i = 0; i < NUM_PARTICLES; i++){
        float3 pos = make_float3((rand() % 100)/100.0, (rand() % 100)/100.0, (rand() % 100)/100.0);
        float3 v   = make_float3((rand() % 100)/100.0, (rand() % 100)/100.0, (rand() % 100)/100.0);
        particles_h[i].pos = pos;
        particles_h[i].v = v;
        //= {pos: pos,  v: v};
    }


    //Copy particles_h to device memory
    cudaMemcpy(particles_d, particles_h, NUM_PARTICLES*sizeof(particle), cudaMemcpyHostToDevice);

    //Launch kernel
    printf("Computing simulation on the GPU...\n");
    double t = cpuSecond();
    for(int i = 0; i < NUM_ITERS; i++){
        particle_kernel <<<(NUM_PARTICLES + BLOCK_SIZE -1)/BLOCK_SIZE, BLOCK_SIZE>>>(particles_d);
    }
    //Sync
    cudaDeviceSynchronize();
    printf("Done in %f seconds!\n", cpuSecond() - t);
    
    //Run host verification
    printf("Computing simulation on the CPU...\n");
    t = cpuSecond();
    for(int i = 0; i < NUM_ITERS; i++){
        verify(particles_h);
    }
    printf("Done in %f seconds!\n", cpuSecond() - t);

    //Copy yd from device to xh, then compare that xh and yh are equal
    printf("Copying from device and comparing the output for device and host\n");
    cudaMemcpy(intermediate_h, particles_d, NUM_PARTICLES*sizeof(particle), cudaMemcpyDeviceToHost);
    
    compare(particles_h, intermediate_h, 1e-6);

    cudaFree(particles_d);
    //free(particles_h); free(intermediate_h);
    return 0;
}
