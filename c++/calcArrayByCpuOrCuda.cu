#include <cstdio>
#include "cuda_runtime.h"

#include <sys/time.h>
#include <time.h>

#include <math.h>

#include "utility/timecost.h"

__global__ void kernelAdd(float * A, float * B, float * C)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*blockDim.x*gridDim.x;
    C[idx]=cos(A[idx])+sin(B[idx]);
    // if(idx == 2077){
        // printf("idx[%d],ix[%d],iy[%d],bdx[%d],bdy[%d],bix[%d],biy[%d],gdx[%d],gdy[%d],C[%f]\r\n", \
        // idx,ix,iy,blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,gridDim.x,gridDim.y,C[idx]);
    // }
}


int main()
{

for(int cycle=0;cycle < 32;cycle++)
{
    int gridSize2Dx = 16*cycle;
    int gridSize2Dy = 16*cycle;
    //blocksize MAX 1024
    int blockSize2Dx = 32;
    int blockSize2Dy = 32;
    int sum = gridSize2Dx*gridSize2Dy*blockSize2Dx*blockSize2Dy;
    int sum_bytes = sum*sizeof(float);
    printf("size %d \r\n", sum);

    float* A_host=(float*)malloc(sum_bytes);
    for(int i=0;i<sum;i++){
        A_host[i]=(float)i;
    }
  
    float* B_host=(float*)malloc(sum_bytes);
    for(int i=0;i<sum;i++){
        B_host[i]=i;
    }

    float* C_host=(float*)malloc(sum_bytes);

    float *A_dev=NULL;
    float *B_dev=NULL;
    float *C_dev=NULL;
    {
        timecost t1("cuda");
        cudaMalloc((void**)&A_dev,sum_bytes);
        cudaMemcpy(A_dev,A_host,sum_bytes,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&B_dev,sum_bytes);
        cudaMemcpy(B_dev,B_host,sum_bytes,cudaMemcpyHostToDevice);

        cudaMalloc((void**)&C_dev,sum_bytes);
        dim3 gridSize2D(gridSize2Dx, gridSize2Dy);
        dim3 blockSize2D(blockSize2Dx, blockSize2Dy);
        kernelAdd<<<gridSize2D, blockSize2D>>>(A_dev,B_dev,C_dev);
        int ret = 0;
        ret = cudaMemcpy(C_host,C_dev,sum_bytes,cudaMemcpyDeviceToHost);
    }

    for(int i=0;i<sum;i++){
        // printf("C[%d]:%f \n", i, C_host[i]);
        C_host[i] = 0.0f;
    }

    {
        timecost t2("cpu");
        for(int i=0;i<sum;i++){
            C_host[i]=cos(A_host[i])+sin(B_host[i]);
        }
    }

    for(int i=0;i<sum;i++){
        // printf("C[%d]:%f \n", i, C_host[i]);
        C_host[i] = 0.0f;
    }

    cudaFree(A_dev);
    free(A_host);
    cudaFree(B_dev);
    free(B_host);
    cudaFree(C_dev);
    free(C_host);
}
    return 0;
}