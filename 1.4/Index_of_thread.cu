#include <stdio.h>

__global__ void hello_from_gpu()
{
    int bid = blockIdx.x;
    int gid = threadIdx.x;
    printf("Hello World from the GPU! blockIdx is %d, threadIdx is %d!\n", bid, gid);
    
    
    //printf("There are %d threads in this block in x direction\n", blockDim.x);
    //printf("There are %d threads in this block in y direction\n", blockDim.y);
    
    //printf("There are %d blocks in this grid in x direction\n", gridDim.x);
    //printf("There are %d blocks in this grid in y direction\n", gridDim.y);
}

int main(void)
{
    hello_from_gpu<<<5, 33>>>();
    
    
    cudaDeviceSynchronize();
    return 0;
}