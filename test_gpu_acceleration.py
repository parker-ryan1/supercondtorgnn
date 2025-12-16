#!/usr/bin/env python3
"""
Test script for GPU acceleration in the superconductor prediction pipeline.

This script tests GPU acceleration by:
1. Checking GPU availability and properties
2. Running a simple benchmark with and without GPU
3. Testing mixed precision training
"""

import os
import sys
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_gpu_info():
    """Check GPU information and display detailed properties."""
    logger.info("Checking GPU information...")
    
    if not torch.cuda.is_available():
        logger.warning("No GPU available. CUDA is not installed or no compatible GPU found.")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPU(s):")
    
    for i in range(gpu_count):
        # Get device properties
        props = torch.cuda.get_device_properties(i)
        
        # Display detailed information
        logger.info(f"  GPU {i}: {props.name}")
        logger.info(f"    - Compute Capability: {props.major}.{props.minor}")
        logger.info(f"    - Total Memory: {props.total_memory / (1024**3):.2f} GB")
        logger.info(f"    - Multi Processors: {props.multi_processor_count}")
        
        # Check current memory usage
        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        logger.info(f"    - Current Memory Usage: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    # Check CUDA version
    logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Check cuDNN version if available
    if hasattr(torch.backends, 'cudnn'):
        logger.info(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        if torch.backends.cudnn.enabled:
            logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
            logger.info(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    # Check if AMP (Automatic Mixed Precision) is available
    amp_available = hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast')
    logger.info(f"Mixed Precision (AMP) Available: {amp_available}")
    
    return True

def run_matrix_multiplication_benchmark(sizes=[1000, 2000, 4000, 8000], 
                                       use_gpu=True, 
                                       use_mixed_precision=True):
    """
    Run a matrix multiplication benchmark to test GPU performance.
    
    Args:
        sizes: List of matrix sizes to test
        use_gpu: Whether to use GPU
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running matrix multiplication benchmark (GPU: {use_gpu}, Mixed Precision: {use_mixed_precision})")
    
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    results = {"sizes": sizes, "times": [], "device": str(device)}
    
    for size in sizes:
        # Create random matrices
        logger.info(f"Testing size {size}x{size}...")
        
        # Measure time to create and transfer matrices
        start_time = time.time()
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.cuda.synchronize() if use_gpu and torch.cuda.is_available() else None
        setup_time = time.time() - start_time
        
        # Warmup
        if use_mixed_precision and use_gpu and torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            with torch.amp.autocast(device_type='cuda'):
                _ = torch.matmul(a, b)
        else:
            _ = torch.matmul(a, b)
        torch.cuda.synchronize() if use_gpu and torch.cuda.is_available() else None
        
        # Measure multiplication time
        start_time = time.time()
        if use_mixed_precision and use_gpu and torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            with torch.amp.autocast(device_type='cuda'):
                c = torch.matmul(a, b)
        else:
            c = torch.matmul(a, b)
        torch.cuda.synchronize() if use_gpu and torch.cuda.is_available() else None
        mult_time = time.time() - start_time
        
        # Log memory usage if using GPU
        if use_gpu and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"  Memory usage: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        # Log times
        logger.info(f"  Setup time: {setup_time:.4f} s, Multiplication time: {mult_time:.4f} s")
        results["times"].append(mult_time)
    
    return results

def plot_benchmark_results(results_list):
    """
    Plot benchmark results.
    
    Args:
        results_list: List of benchmark result dictionaries
    """
    plt.figure(figsize=(10, 6))
    
    for results in results_list:
        plt.plot(results["sizes"], results["times"], marker='o', label=results["device"])
    
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix Multiplication Benchmark")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    
    # Save plot
    plt.savefig("benchmark_results.png", dpi=300)
    logger.info("Benchmark plot saved to benchmark_results.png")

def test_gpu_memory_limits():
    """Test GPU memory limits by allocating increasingly large tensors."""
    if not torch.cuda.is_available():
        logger.warning("No GPU available. Skipping memory limit test.")
        return
    
    logger.info("Testing GPU memory limits...")
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024**3)
    logger.info(f"Total GPU memory: {total_memory_gb:.2f} GB")
    
    # Start with 10% of total memory and increase
    sizes = []
    allocated = []
    
    try:
        for i in range(1, 9):  # Limit to 8 steps (80% of memory) to avoid OOM errors
            # Calculate tensor size (i*10% of total memory)
            target_bytes = int(total_memory * i * 0.1)
            # Calculate dimensions for a square matrix that would use approximately this memory
            # Each float32 element uses 4 bytes
            dim = int(np.sqrt(target_bytes / 4))
            sizes.append(dim)
            
            logger.info(f"Allocating tensor of size {dim}x{dim} ({target_bytes/(1024**3):.2f} GB)...")
            
            # Clear cache before allocation
            torch.cuda.empty_cache()
            
            # Allocate tensor
            start_time = time.time()
            x = torch.zeros(dim, dim, device="cuda")
            torch.cuda.synchronize()
            allocation_time = time.time() - start_time
            
            # Record allocated memory
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            allocated.append(mem_allocated)
            
            logger.info(f"  Allocated {mem_allocated:.2f} GB in {allocation_time:.4f} s")
            
            # Free memory
            del x
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        logger.warning(f"Memory allocation failed: {str(e)}")
    finally:
        # Plot results if we have data
        if sizes and allocated:
            plt.figure(figsize=(10, 6))
            plt.plot(sizes, allocated, marker='o')
            plt.xlabel("Matrix Dimension")
            plt.ylabel("Allocated Memory (GB)")
            plt.title("GPU Memory Allocation Test")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig("memory_test_results.png", dpi=300)
            logger.info("Memory test plot saved to memory_test_results.png")

def main():
    """Main entry point."""
    logger.info("Starting GPU acceleration test...")
    
    # Check GPU information
    has_gpu = check_gpu_info()
    
    if has_gpu:
        # Run benchmarks with smaller matrices due to limited GPU memory
        sizes = [1000, 2000, 3000, 4000]
        cpu_results = run_matrix_multiplication_benchmark(sizes=sizes, use_gpu=False, use_mixed_precision=False)
        gpu_results = run_matrix_multiplication_benchmark(sizes=sizes, use_gpu=True, use_mixed_precision=False)
        
        # Run mixed precision benchmark if available
        if hasattr(torch.cuda, 'amp'):
            gpu_mp_results = run_matrix_multiplication_benchmark(sizes=sizes, use_gpu=True, use_mixed_precision=True)
            plot_benchmark_results([cpu_results, gpu_results, gpu_mp_results])
        else:
            plot_benchmark_results([cpu_results, gpu_results])
        
        # Test GPU memory limits
        test_gpu_memory_limits()
        
        # Test GPU accelerator
        logger.info("GPU acceleration test completed successfully")
        logger.info(f"The GPU is working properly and can be used for training")
    else:
        logger.warning("GPU acceleration test skipped because no GPU is available.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)