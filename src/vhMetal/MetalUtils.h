/*
 * Metal device utilities for VerthashMiner
 * Provides Metal GPU detection and management for macOS
 */

#ifndef METALUTILS_H
#define METALUTILS_H

#ifdef __APPLE__
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
typedef void* id;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Metal device information structure (parallel to OpenCL cldevice_t)
typedef struct {
    id device;              // MTLDevice*
    id commandQueue;        // MTLCommandQueue*
    id computePipeline;     // MTLComputePipelineState*
    id verthashDataBuffer;  // MTLBuffer* for verthash data
    id inputBuffer;         // MTLBuffer* for input hashes
    id outputBuffer;        // MTLBuffer* for results
    id kStateBuffer;        // MTLBuffer* for keccak states
    char deviceName[256];
    size_t maxThreadsPerThreadgroup;
    size_t recommendedMaxWorkingSetSize;
    bool isLowPower;        // Integrated vs discrete GPU
} mtldevice_t;

// Metal device management functions
int mtl_get_device_count();
int mtl_get_device_info(int deviceIndex, mtldevice_t* device);
int mtl_create_compute_pipeline(mtldevice_t* device, const char* kernelSource, size_t sourceLen);
int mtl_allocate_buffers(mtldevice_t* device, size_t verthashDataSize, size_t workSize);
void mtl_cleanup_device(mtldevice_t* device);

// Metal kernel execution
int mtl_execute_verthash_kernel(
    mtldevice_t* device,
    uint32_t* inputHashes,
    uint32_t* kStates,
    uint8_t* verthashData,
    uint32_t in18,
    uint32_t firstNonce,
    uint32_t* results,
    uint64_t target,
    size_t workSize
);

#ifdef __cplusplus
}
#endif

#endif // __APPLE__
#endif // METALUTILS_H
