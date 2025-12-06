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

        if (deviceIndex >= [devices count]) {
            [devices release];
            return -1;
        }

        id<MTLDevice> device = devices[deviceIndex];

        // Store device (retain it)
        mtlDev->device = (__bridge id)device;

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
        mtlDev->commandQueue = (__bridge id)queue;

        [devices release];

        printf("[Metal] Device %d: %s\n", deviceIndex, mtlDev->deviceName);
        printf("[Metal]   Max threads: %zu\n", mtlDev->maxThreadsPerThreadgroup);
        printf("[Metal]   Low power: %s\n", mtlDev->isLowPower ? "Yes" : "No");

        return 0;
    }
}

// Create Metal compute pipeline from source
int mtl_create_compute_pipeline(mtldevice_t* mtlDev, const char* kernelSource, size_t sourceLen) {
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

        // Get the kernel function
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"verthash_4w"];
        if (!kernelFunction) {
            printf("[Metal] Failed to find kernel function 'verthash_4w'\n");
            [library release];
            return -1;
        }

        // Create compute pipeline
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunction
                                                                                      error:&error];
        [kernelFunction release];
        [library release];

        if (!pipeline) {
            printf("[Metal] Failed to create pipeline: %s\n", [[error localizedDescription] UTF8String]);
            return -1;
        }

        mtlDev->computePipeline = (__bridge id)pipeline;

        printf("[Metal] Compute pipeline created successfully\n");
        return 0;
    }
}

// Allocate Metal buffers
int mtl_allocate_buffers(mtldevice_t* mtlDev, size_t verthashDataSize, size_t workSize) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDev->device;

        // Allocate verthash data buffer (shared with CPU)
        id<MTLBuffer> verthashBuf = [device newBufferWithLength:verthashDataSize
                                                         options:MTLResourceStorageModeShared];
        if (!verthashBuf) {
            printf("[Metal] Failed to allocate verthash data buffer\n");
            return -1;
        }
        mtlDev->verthashDataBuffer = (__bridge id)verthashBuf;

        // Allocate input buffer (hashes)
        size_t inputSize = workSize * 8 * sizeof(uint32_t);  // 8 uint32s per hash
        id<MTLBuffer> inputBuf = [device newBufferWithLength:inputSize
                                                      options:MTLResourceStorageModeShared];
        if (!inputBuf) {
            printf("[Metal] Failed to allocate input buffer\n");
            return -1;
        }
        mtlDev->inputBuffer = (__bridge id)inputBuf;

        // Allocate output buffer (results)
        size_t outputSize = workSize * sizeof(uint32_t);
        id<MTLBuffer> outputBuf = [device newBufferWithLength:outputSize
                                                       options:MTLResourceStorageModeShared];
        if (!outputBuf) {
            printf("[Metal] Failed to allocate output buffer\n");
            return -1;
        }
        mtlDev->outputBuffer = (__bridge id)outputBuf;

        // Allocate kState buffer
        size_t kStateSize = (workSize / 4) * 50 * sizeof(uint64_t);  // 50 uint64s per state, 4 work items per state
        id<MTLBuffer> kStateBuf = [device newBufferWithLength:kStateSize
                                                       options:MTLResourceStorageModeShared];
        if (!kStateBuf) {
            printf("[Metal] Failed to allocate kState buffer\n");
            return -1;
        }
        mtlDev->kStateBuffer = (__bridge id)kStateBuf;

        printf("[Metal] Buffers allocated: verthash=%zu MB, input=%zu KB, output=%zu KB\n",
               verthashDataSize / (1024*1024), inputSize / 1024, outputSize / 1024);

        return 0;
    }
}

// Execute verthash kernel
int mtl_execute_verthash_kernel(
    mtldevice_t* mtlDev,
    uint32_t* inputHashes,
    uint32_t* kStates,
    uint8_t* verthashData,
    uint32_t in18,
    uint32_t firstNonce,
    uint32_t* results,
    uint64_t target,
    size_t workSize
) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDev->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlDev->commandQueue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)mtlDev->computePipeline;

        id<MTLBuffer> verthashBuf = (__bridge id<MTLBuffer>)mtlDev->verthashDataBuffer;
        id<MTLBuffer> inputBuf = (__bridge id<MTLBuffer>)mtlDev->inputBuffer;
        id<MTLBuffer> outputBuf = (__bridge id<MTLBuffer>)mtlDev->outputBuffer;
        id<MTLBuffer> kStateBuf = (__bridge id<MTLBuffer>)mtlDev->kStateBuffer;

        // Copy data to buffers
        memcpy([inputBuf contents], inputHashes, workSize * 8 * sizeof(uint32_t));
        memcpy([kStateBuf contents], kStates, (workSize / 4) * 50 * sizeof(uint64_t));
        memcpy([verthashBuf contents], verthashData, [verthashBuf length]);
        memset([outputBuf contents], 0, workSize * sizeof(uint32_t));

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Set pipeline
        [encoder setComputePipelineState:pipeline];

        // Set buffers
        [encoder setBuffer:inputBuf offset:0 atIndex:0];
        [encoder setBuffer:kStateBuf offset:0 atIndex:1];
        [encoder setBuffer:verthashBuf offset:0 atIndex:2];
        [encoder setBytes:&in18 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&firstNonce length:sizeof(uint32_t) atIndex:4];
        [encoder setBuffer:outputBuf offset:0 atIndex:5];
        [encoder setBytes:&target length:sizeof(uint64_t) atIndex:6];

        // Calculate thread configuration
        MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);  // Start with 256
        MTLSize threadgroupsPerGrid = MTLSizeMake((workSize + 255) / 256, 1, 1);

        // Dispatch
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy results back
        memcpy(results, [outputBuf contents], workSize * sizeof(uint32_t));

        return 0;
    }
}

// Cleanup Metal device
void mtl_cleanup_device(mtldevice_t* mtlDev) {
    @autoreleasepool {
        if (mtlDev->outputBuffer) {
            CFBridgingRelease(mtlDev->outputBuffer);
            mtlDev->outputBuffer = nil;
        }
        if (mtlDev->inputBuffer) {
            CFBridgingRelease(mtlDev->inputBuffer);
            mtlDev->inputBuffer = nil;
        }
        if (mtlDev->kStateBuffer) {
            CFBridgingRelease(mtlDev->kStateBuffer);
            mtlDev->kStateBuffer = nil;
        }
        if (mtlDev->verthashDataBuffer) {
            CFBridgingRelease(mtlDev->verthashDataBuffer);
            mtlDev->verthashDataBuffer = nil;
        }
        if (mtlDev->computePipeline) {
            CFBridgingRelease(mtlDev->computePipeline);
            mtlDev->computePipeline = nil;
        }
        if (mtlDev->commandQueue) {
            CFBridgingRelease(mtlDev->commandQueue);
            mtlDev->commandQueue = nil;
        }
        if (mtlDev->device) {
            CFBridgingRelease(mtlDev->device);
            mtlDev->device = nil;
        }
    }
}

#endif // __APPLE__
