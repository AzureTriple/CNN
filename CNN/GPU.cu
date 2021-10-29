#include "GPU.cuh"

cudaStream_t GPU::createStream()
{
    cudaStream_t out;
    check(cudaStreamCreate(&out));
    return out;
}
void GPU::destroyStream(cudaStream_t stream) {check(cudaStreamDestroy(stream));}

void GPU::destroyHostPinned(void *arr) {check(cudaFreeHost(arr));}

void GPU::destroyDeviceMem(void *arr,cudaStream_t stream) {check(cudaFreeAsync(arr,stream));}

void GPU::sync() {check(cudaDeviceSynchronize());}
void GPU::sync(cudaStream_t stream) {check(cudaStreamSynchronize(stream));}