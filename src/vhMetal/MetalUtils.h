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
    id device;                      // MTLDevice*
    id commandQueue;                // MTLCommandQueue*

    // Compute pipelines for each kernel stage
    id precomputePipeline;          // MTLComputePipelineState* for sha3_512_precompute
    id sha3Pipeline;                // MTLComputePipelineState* for sha3_512_256
    id verthashPipeline;            // MTLComputePipelineState* for verthash_4w

    // Buffers
    id verthashDataBuffer;          // MTLBuffer* for verthash data (~1.2GB)
    id kStateBuffer;                // MTLBuffer* for 8 precomputed keccak states (8 * 50 * 8 bytes)
    id headerBuffer;                // MTLBuffer* for block header (18 * 4 bytes)
    id hashResultsBuffer;           // MTLBuffer* for SHA3-256 hashes (workSize * 8 * 4 bytes)
    id targetResultsBuffer;         // MTLBuffer* for target results ((workSize+1) * 4 bytes)

    char deviceName[256];
    size_t maxThreadsPerThreadgroup;
    size_t recommendedMaxWorkingSetSize;
    size_t workSize;                // Current work size
    bool isLowPower;                // Integrated vs discrete GPU
} mtldevice_t;

// Metal device management functions
int mtl_get_device_count();
int mtl_get_device_info(int deviceIndex, mtldevice_t* device);
int mtl_create_pipelines(mtldevice_t* device, const char* kernelSource, size_t sourceLen);
int mtl_allocate_buffers(mtldevice_t* device, size_t verthashDataSize, size_t workSize);
int mtl_upload_verthash_data(mtldevice_t* device, const uint8_t* verthashData, size_t dataSize);
void mtl_cleanup_device(mtldevice_t* device);

// Metal kernel execution - full pipeline
int mtl_run_precompute(mtldevice_t* device, const uint32_t* header);
int mtl_run_mining_batch(
    mtldevice_t* device,
    uint32_t in18,
    uint32_t firstNonce,
    uint64_t target,
    size_t batchSize,
    uint32_t* foundCount,
    uint32_t* foundNonces
);

#ifdef __cplusplus
}
#endif

#endif // __APPLE__
#endif // METALUTILS_H
