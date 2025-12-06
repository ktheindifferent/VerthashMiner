/*
 * Metal device utilities implementation for VerthashMiner
 */

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "MetalUtils.h"
#include <string.h>
#include <stdio.h>

// Get number of Metal devices
int mtl_get_device_count() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    int count = (int)[devices count];
    [devices release];
    return count;
}

// Get Metal device information
int mtl_get_device_info(int deviceIndex, mtldevice_t* mtlDev) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();

        if (deviceIndex >= (int)[devices count]) {
            [devices release];
            return -1;
        }

        id<MTLDevice> device = devices[deviceIndex];

        // Store device (retain it)
        mtlDev->device = (__bridge_retained id)device;

        // Get device name
        const char* name = [[device name] UTF8String];
        strncpy(mtlDev->deviceName, name, sizeof(mtlDev->deviceName) - 1);
        mtlDev->deviceName[sizeof(mtlDev->deviceName) - 1] = '\0';

        // Get device capabilities
        mtlDev->maxThreadsPerThreadgroup = [device maxThreadsPerThreadgroup].width;
        mtlDev->recommendedMaxWorkingSetSize = (size_t)[device recommendedMaxWorkingSetSize];
        mtlDev->isLowPower = [device isLowPower];

        // Create command queue
        id<MTLCommandQueue> queue = [device newCommandQueue];
        mtlDev->commandQueue = (__bridge_retained id)queue;

        // Initialize pipeline pointers to nil
        mtlDev->precomputePipeline = nil;
        mtlDev->sha3Pipeline = nil;
        mtlDev->verthashPipeline = nil;
        mtlDev->verthashDataBuffer = nil;
        mtlDev->kStateBuffer = nil;
        mtlDev->headerBuffer = nil;
        mtlDev->hashResultsBuffer = nil;
        mtlDev->targetResultsBuffer = nil;
        mtlDev->workSize = 0;

        [devices release];

        printf("[Metal] Device %d: %s\n", deviceIndex, mtlDev->deviceName);
        printf("[Metal]   Max threads: %zu\n", mtlDev->maxThreadsPerThreadgroup);
        printf("[Metal]   Low power: %s\n", mtlDev->isLowPower ? "Yes" : "No");

        return 0;
    }
}

// Create Metal compute pipelines from source
int mtl_create_pipelines(mtldevice_t* mtlDev, const char* kernelSource, size_t sourceLen) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDev->device;

        // Create string from source
        NSString* sourceString = [[NSString alloc] initWithBytes:kernelSource
                                                          length:sourceLen
                                                        encoding:NSUTF8StringEncoding];

        NSError* error = nil;

        // Compile options
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        [options setFastMathEnabled:YES];

        // Create library from source
        id<MTLLibrary> library = [device newLibraryWithSource:sourceString
                                                       options:options
                                                         error:&error];
        [sourceString release];
        [options release];

        if (!library) {
            printf("[Metal] Failed to compile kernel: %s\n", [[error localizedDescription] UTF8String]);
            return -1;
        }

        // Create sha3_512_precompute pipeline
        id<MTLFunction> precomputeFunc = [library newFunctionWithName:@"sha3_512_precompute"];
        if (!precomputeFunc) {
            printf("[Metal] Failed to find kernel function 'sha3_512_precompute'\n");
            [library release];
            return -1;
        }

        id<MTLComputePipelineState> precomputePipeline = [device newComputePipelineStateWithFunction:precomputeFunc
                                                                                                error:&error];
        [precomputeFunc release];
        if (!precomputePipeline) {
            printf("[Metal] Failed to create precompute pipeline: %s\n", [[error localizedDescription] UTF8String]);
            [library release];
            return -1;
        }
        mtlDev->precomputePipeline = (__bridge_retained id)precomputePipeline;

        // Create sha3_512_256 pipeline
        id<MTLFunction> sha3Func = [library newFunctionWithName:@"sha3_512_256"];
        if (!sha3Func) {
            printf("[Metal] Failed to find kernel function 'sha3_512_256'\n");
            [library release];
            return -1;
        }

        id<MTLComputePipelineState> sha3Pipeline = [device newComputePipelineStateWithFunction:sha3Func
                                                                                          error:&error];
        [sha3Func release];
        if (!sha3Pipeline) {
            printf("[Metal] Failed to create SHA3 pipeline: %s\n", [[error localizedDescription] UTF8String]);
            [library release];
            return -1;
        }
        mtlDev->sha3Pipeline = (__bridge_retained id)sha3Pipeline;

        // Create verthash_4w pipeline
        id<MTLFunction> verthashFunc = [library newFunctionWithName:@"verthash_4w"];
        if (!verthashFunc) {
            printf("[Metal] Failed to find kernel function 'verthash_4w'\n");
            [library release];
            return -1;
        }

        id<MTLComputePipelineState> verthashPipeline = [device newComputePipelineStateWithFunction:verthashFunc
                                                                                              error:&error];
        [verthashFunc release];
        [library release];

        if (!verthashPipeline) {
            printf("[Metal] Failed to create verthash pipeline: %s\n", [[error localizedDescription] UTF8String]);
            return -1;
        }
        mtlDev->verthashPipeline = (__bridge_retained id)verthashPipeline;

        printf("[Metal] All compute pipelines created successfully\n");
        return 0;
    }
}

// Allocate Metal buffers
int mtl_allocate_buffers(mtldevice_t* mtlDev, size_t verthashDataSize, size_t workSize) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDev->device;

        mtlDev->workSize = workSize;

        // Allocate verthash data buffer (shared with CPU for upload)
        id<MTLBuffer> verthashBuf = [device newBufferWithLength:verthashDataSize
                                                         options:MTLResourceStorageModeShared];
        if (!verthashBuf) {
            printf("[Metal] Failed to allocate verthash data buffer\n");
            return -1;
        }
        mtlDev->verthashDataBuffer = (__bridge_retained id)verthashBuf;

        // Allocate kState buffer: 8 keccak states * 25 uint64s each = 8 * 25 * 8 = 1600 bytes
        // But we use kstate2x_t with 50 uint64s for 4 lanes (4 * 50 * 8 = 1600 bytes)
        size_t kStateSize = 4 * 50 * sizeof(uint64_t);
        id<MTLBuffer> kStateBuf = [device newBufferWithLength:kStateSize
                                                       options:MTLResourceStorageModeShared];
        if (!kStateBuf) {
            printf("[Metal] Failed to allocate kState buffer\n");
            return -1;
        }
        mtlDev->kStateBuffer = (__bridge_retained id)kStateBuf;

        // Allocate header buffer: 18 uint32s
        size_t headerSize = 18 * sizeof(uint32_t);
        id<MTLBuffer> headerBuf = [device newBufferWithLength:headerSize
                                                       options:MTLResourceStorageModeShared];
        if (!headerBuf) {
            printf("[Metal] Failed to allocate header buffer\n");
            return -1;
        }
        mtlDev->headerBuffer = (__bridge_retained id)headerBuf;

        // Allocate hash results buffer: workSize * 8 uint32s (256-bit hash per work item)
        size_t hashResultsSize = workSize * 8 * sizeof(uint32_t);
        id<MTLBuffer> hashResultsBuf = [device newBufferWithLength:hashResultsSize
                                                            options:MTLResourceStorageModeShared];
        if (!hashResultsBuf) {
            printf("[Metal] Failed to allocate hash results buffer\n");
            return -1;
        }
        mtlDev->hashResultsBuffer = (__bridge_retained id)hashResultsBuf;

        // Allocate target results buffer: (workSize + 1) uint32s
        size_t targetResultsSize = (workSize + 1) * sizeof(uint32_t);
        id<MTLBuffer> targetResultsBuf = [device newBufferWithLength:targetResultsSize
                                                              options:MTLResourceStorageModeShared];
        if (!targetResultsBuf) {
            printf("[Metal] Failed to allocate target results buffer\n");
            return -1;
        }
        mtlDev->targetResultsBuffer = (__bridge_retained id)targetResultsBuf;

        printf("[Metal] Buffers allocated: verthash=%zu MB, kState=%zu bytes, hashResults=%zu KB\n",
               verthashDataSize / (1024*1024), kStateSize, hashResultsSize / 1024);

        return 0;
    }
}

// Upload verthash data to GPU
int mtl_upload_verthash_data(mtldevice_t* mtlDev, const uint8_t* verthashData, size_t dataSize) {
    @autoreleasepool {
        id<MTLBuffer> verthashBuf = (__bridge id<MTLBuffer>)mtlDev->verthashDataBuffer;
        memcpy([verthashBuf contents], verthashData, dataSize);
        printf("[Metal] Verthash data uploaded: %zu MB\n", dataSize / (1024*1024));
        return 0;
    }
}

// Run SHA3 precompute stage (once per new work)
int mtl_run_precompute(mtldevice_t* mtlDev, const uint32_t* header) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlDev->commandQueue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)mtlDev->precomputePipeline;
        id<MTLBuffer> kStateBuf = (__bridge id<MTLBuffer>)mtlDev->kStateBuffer;
        id<MTLBuffer> headerBuf = (__bridge id<MTLBuffer>)mtlDev->headerBuffer;

        // Copy header to buffer
        memcpy([headerBuf contents], header, 18 * sizeof(uint32_t));

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set pipeline and buffers
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:kStateBuf offset:0 atIndex:0];
        [encoder setBuffer:headerBuf offset:0 atIndex:1];

        // Dispatch 8 threads (one per keccak state)
        MTLSize threadsPerThreadgroup = MTLSizeMake(8, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(1, 1, 1);

        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        return 0;
    }
}

// Run mining batch (SHA3-256 + Verthash kernels)
int mtl_run_mining_batch(
    mtldevice_t* mtlDev,
    uint32_t in18,
    uint32_t firstNonce,
    uint64_t target,
    size_t batchSize,
    uint32_t* foundCount,
    uint32_t* foundNonces
) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlDev->commandQueue;
        id<MTLComputePipelineState> sha3Pipeline = (__bridge id<MTLComputePipelineState>)mtlDev->sha3Pipeline;
        id<MTLComputePipelineState> verthashPipeline = (__bridge id<MTLComputePipelineState>)mtlDev->verthashPipeline;

        id<MTLBuffer> headerBuf = (__bridge id<MTLBuffer>)mtlDev->headerBuffer;
        id<MTLBuffer> hashResultsBuf = (__bridge id<MTLBuffer>)mtlDev->hashResultsBuffer;
        id<MTLBuffer> kStateBuf = (__bridge id<MTLBuffer>)mtlDev->kStateBuffer;
        id<MTLBuffer> verthashBuf = (__bridge id<MTLBuffer>)mtlDev->verthashDataBuffer;
        id<MTLBuffer> targetResultsBuf = (__bridge id<MTLBuffer>)mtlDev->targetResultsBuffer;

        // Clear target results counter
        memset([targetResultsBuf contents], 0, sizeof(uint32_t));

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        // Stage 1: SHA3-256 kernel
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:sha3Pipeline];
            [encoder setBuffer:hashResultsBuf offset:0 atIndex:0];
            [encoder setBuffer:headerBuf offset:0 atIndex:1];
            [encoder setBytes:&in18 length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&firstNonce length:sizeof(uint32_t) atIndex:3];

            MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((batchSize + 255) / 256, 1, 1);

            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        // Stage 2: Verthash 4-way kernel
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:verthashPipeline];
            [encoder setBuffer:hashResultsBuf offset:0 atIndex:0];
            [encoder setBuffer:kStateBuf offset:0 atIndex:1];
            [encoder setBuffer:verthashBuf offset:0 atIndex:2];
            [encoder setBytes:&in18 length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&firstNonce length:sizeof(uint32_t) atIndex:4];
            [encoder setBuffer:targetResultsBuf offset:0 atIndex:5];
            [encoder setBytes:&target length:sizeof(uint64_t) atIndex:6];

            // Verthash uses 4x threads (4-way kernel)
            size_t verthashWorkSize = batchSize * 4;
            MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((verthashWorkSize + 255) / 256, 1, 1);

            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Read results
        uint32_t* results = (uint32_t*)[targetResultsBuf contents];
        *foundCount = results[0];
        if (*foundCount > 0 && *foundCount < 16) {
            for (uint32_t i = 0; i < *foundCount; i++) {
                foundNonces[i] = results[i + 1];
            }
        }

        return 0;
    }
}

// Cleanup Metal device
void mtl_cleanup_device(mtldevice_t* mtlDev) {
    @autoreleasepool {
        if (mtlDev->targetResultsBuffer) {
            CFRelease(mtlDev->targetResultsBuffer);
            mtlDev->targetResultsBuffer = nil;
        }
        if (mtlDev->hashResultsBuffer) {
            CFRelease(mtlDev->hashResultsBuffer);
            mtlDev->hashResultsBuffer = nil;
        }
        if (mtlDev->headerBuffer) {
            CFRelease(mtlDev->headerBuffer);
            mtlDev->headerBuffer = nil;
        }
        if (mtlDev->kStateBuffer) {
            CFRelease(mtlDev->kStateBuffer);
            mtlDev->kStateBuffer = nil;
        }
        if (mtlDev->verthashDataBuffer) {
            CFRelease(mtlDev->verthashDataBuffer);
            mtlDev->verthashDataBuffer = nil;
        }
        if (mtlDev->verthashPipeline) {
            CFRelease(mtlDev->verthashPipeline);
            mtlDev->verthashPipeline = nil;
        }
        if (mtlDev->sha3Pipeline) {
            CFRelease(mtlDev->sha3Pipeline);
            mtlDev->sha3Pipeline = nil;
        }
        if (mtlDev->precomputePipeline) {
            CFRelease(mtlDev->precomputePipeline);
            mtlDev->precomputePipeline = nil;
        }
        if (mtlDev->commandQueue) {
            CFRelease(mtlDev->commandQueue);
            mtlDev->commandQueue = nil;
        }
        if (mtlDev->device) {
            CFRelease(mtlDev->device);
            mtlDev->device = nil;
        }
    }
}

#endif // __APPLE__
