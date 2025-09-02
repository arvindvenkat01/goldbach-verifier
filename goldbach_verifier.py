"""
High-Performance Goldbach Conjecture Verification Framework
=========================================================

This script provides an efficient, single-node implementation for verifying the
Goldbach Conjecture up to a specified integer limit. It leverages a structural
decomposition of the problem based on prime residue classes modulo 6, combined
with JIT compilation via Numba for high performance.

Usage (Command-Line):
    python goldbach_verifier.py <target_number> [options]

    Example:
        python goldbach_verifier.py 1e10 --chunk_size 1e8

Usage (Jupyter/Colab):
    In a notebook cell, you can call one of the helper functions:
    - run_verification(target_n=1e7)
    - quick_test()
    - run_benchmark_comparison()

Key Optimizations:
1.  Residue class decomposition (mod 6) to prune search space.
2.  Optimized Sieve of Eratosthenes for prime generation.
3.  JIT compilation with Numba for C-like performance in critical loops.
4.  Parallel processing of verification chunks using `numba.prange`.
5.  Hash tables (numba.typed.Dict) for O(1) average-case prime lookups.
"""

import numpy as np
import numba as nb
from numba import jit, prange
from numba.typed import Dict
import time
import math
import psutil
import platform
import os
import gc
import argparse

def get_system_info():
    """Prints detailed system hardware and software information."""
    print("System Hardware and Software Information")
    print("=" * 50)
    try:
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_model = line.split(':')[1].strip()
                        print(f"CPU Model: {cpu_model}")
                        break
                else:
                    print(f"CPU Model: {platform.processor()}")
        else:
            print(f"CPU Model: {platform.processor()}")
        
        print(f"Physical CPU Cores: {psutil.cpu_count(logical=False)}")
        print(f"Logical CPU Cores: {psutil.cpu_count(logical=True)}")
        
        memory_info = psutil.virtual_memory()
        print(f"Total RAM: {memory_info.total / (1024**3):.1f} GB")
        
        print(f"Platform: {platform.platform()}")
        print(f"Python: {platform.python_version()}")
        print(f"Numba: {nb.__version__}")
        print(f"NumPy: {np.__version__}")
    except Exception as e:
        print(f"Could not retrieve full system info: {e}")
    print()

@jit(nopython=True)
def fast_sieve(limit):
    """
    Generates primes up to a given limit using an optimized Sieve of Eratosthenes.
    """
    is_prime = np.ones(limit + 1, dtype=nb.boolean)
    is_prime[0] = is_prime[1] = False
    
    sqrt_limit = int(np.sqrt(limit)) + 1
    for i in range(2, sqrt_limit):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    return is_prime

@jit(nopython=True)
def extract_primes_classified(is_prime, limit):
    """
    Extracts and classifies primes as Type A (p=1 mod 6) or Type B (p=5 mod 6).
    """
    count_a = 0
    count_b = 0
    for i in range(5, limit + 1):
        if is_prime[i]:
            if i % 6 == 1:
                count_a += 1
            elif i % 6 == 5:
                count_b += 1
    
    type_a = np.empty(count_a, dtype=np.int64)
    type_b = np.empty(count_b, dtype=np.int64)
    
    idx_a = 0
    idx_b = 0
    for i in range(5, limit + 1):
        if is_prime[i]:
            if i % 6 == 1:
                type_a[idx_a] = i
                idx_a += 1
            elif i % 6 == 5:
                type_b[idx_b] = i
                idx_b += 1
    
    return type_a, type_b

@jit(nopython=True)
def create_hash_tables(type_a, type_b):
    """
    Creates Numba-compatible hash tables for fast prime lookups (Optimized).
    """
    hash_a = Dict.empty(key_type=nb.int64, value_type=nb.boolean)
    for p in type_a:
        hash_a[p] = True
        
    hash_b = Dict.empty(key_type=nb.int64, value_type=nb.boolean)
    for p in type_b:
        hash_b[p] = True
    
    return hash_a, hash_b

@jit(nopython=True)
def verify_goldbach_fast(n, type_a, type_b, hash_a, hash_b):
    """
    Verifies a single even number `n` using the optimized decomposition.
    """
    residue = n % 6
    
    if residue == 0:
        if n == 6: return True
        for p in type_a:
            if p > n // 2: break
            if (n - p) in hash_b: return True
        for p in type_b:
            if p > n // 2: break
            if (n - p) in hash_a: return True
                
    elif residue == 2:
        if n > 3 and (n - 3) in hash_b: return True
        for p in type_a:
            complement = n - p
            if p > complement: break
            if complement in hash_a: return True
            
    else:
        if n == 4: return True
        if n > 3 and (n - 3) in hash_a: return True
        for p in type_b:
            complement = n - p
            if p > complement: break
            if complement in hash_b: return True
    
    return False

@jit(nopython=True, parallel=True)
def batch_verify(start_n, end_n, type_a, type_b, hash_a, hash_b):
    """
    Verifies a batch of even numbers in parallel (Optimized).
    """
    num_even_numbers = (end_n - start_n) // 2 + 1
    results = np.ones(num_even_numbers, dtype=nb.boolean)
    
    for i in prange(num_even_numbers):
        n = start_n + i * 2
        if not verify_goldbach_fast(n, type_a, type_b, hash_a, hash_b):
            results[i] = False
    
    return results

# --- Baseline Functions for Comparison ---

@jit(nopython=True)
def create_hash_table_all(all_primes_arr):
    """Creates a Numba hash table for all primes (Baseline)."""
    hash_all = Dict.empty(key_type=nb.int64, value_type=nb.boolean)
    for p in all_primes_arr:
        hash_all[p] = True
    return hash_all

@jit(nopython=True)
def verify_goldbach_baseline(n, all_primes_arr, all_primes_hash):
    """Verifies a single even number `n` using the baseline method."""
    if n == 4: return True
    for p in all_primes_arr:
        complement = n - p
        if p > complement:
            break
        if complement in all_primes_hash:
            return True
    return False

# --- Main Application and Helper Functions ---

def run_verification(target_n, chunk_size=1e8):
    """Main pipeline to run the full Goldbach verification process."""
    target_n = int(target_n)
    chunk_size = int(chunk_size)
    
    print(f"Starting Goldbach Conjecture Verification")
    print(f"Target: {target_n:,}")
    
    # --- Phase 1: Sieving ---
    prep_start_time = time.time()
    print("\nPhase 1: Generating primes...")
    is_prime = fast_sieve(target_n)
    
    # --- Phase 2: Classification ---
    print("Phase 2: Classifying primes by residue class...")
    type_a, type_b = extract_primes_classified(is_prime, target_n)
    
    # --- Phase 3: Hashing ---
    print("Phase 3: Building hash tables for fast lookups...")
    hash_a, hash_b = create_hash_tables(type_a, type_b)
    prep_time = time.time() - prep_start_time
    del is_prime; gc.collect()

    # --- Phase 4: Verification ---
    total_to_verify = (target_n - 2) // 2
    print(f"\nPhase 4: Verifying {total_to_verify:,} even numbers...")
    
    num_chunks = (target_n + chunk_size - 1) // chunk_size
    all_passed = True
    
    ver_start_time = time.time()
    
    for i in range(int(num_chunks)):
        start_n = i * chunk_size
        end_n = min((i + 1) * chunk_size - 2, target_n)
        if start_n == 0: start_n = 4
        if start_n > end_n: break

        chunk_start_time = time.time()
        results = batch_verify(start_n, end_n, type_a, type_b, hash_a, hash_b)
        chunk_time = time.time() - chunk_start_time
        
        if not np.all(results):
            all_passed = False
            first_fail_index = np.argmin(results)
            failed_number = start_n + 2 * first_fail_index
            print(f"\n!!! COUNTEREXAMPLE FOUND: {failed_number} !!!")
            break
        
        rate = ((end_n - start_n) / 2) / chunk_time if chunk_time > 0 else 0
        progress = (end_n / target_n) * 100
        print(f"Chunk {i+1:3d}/{int(num_chunks)} [{start_n:11,d}-{end_n:11,d}] | Progress: {progress:5.1f}% | Rate: {rate:11,.0f} n/s")

    ver_time = time.time() - ver_start_time
    
    # --- Summary ---
    print("\n" + "="*50)
    print("Verification Summary")
    print("="*50)
    print(f"Total Preprocessing Time: {prep_time:.1f} seconds")
    print(f"Total Verification Time:  {ver_time:.1f} seconds ({ver_time/3600:.2f} hours)")
    
    if ver_time > 0:
        overall_rate = total_to_verify / ver_time
        print(f"Processing Rate: {overall_rate:,.0f} numbers/second")

    if all_passed:
        print("\nSUCCESS: No counterexamples found to the Goldbach Conjecture.")
    else:
        print("\nFAILURE: A counterexample was found.")
    return all_passed

@jit(nopython=True)
def serial_benchmark_optimized(limit, type_a, type_b, hash_a, hash_b):
    """Helper for serial benchmark of the optimized method."""
    # This function is not meant to be accurate for results, only for timing.
    # We don't check for counterexamples here to keep the loop tight.
    for n in range(4, limit + 1, 2):
        verify_goldbach_fast(n, type_a, type_b, hash_a, hash_b)

@jit(nopython=True)
def serial_benchmark_baseline(limit, all_primes_arr, all_primes_hash):
    """Helper for serial benchmark of the baseline method."""
    for n in range(4, limit + 1, 2):
        verify_goldbach_baseline(n, all_primes_arr, all_primes_hash)

def run_benchmark_comparison():
    """
    Runs a side-by-side SERIAL benchmark of the optimized vs. baseline algorithm.
    This provides a true measure of the algorithmic speedup by removing parallelization
    artifacts that can skew results.
    """
    print("--- Running Performance Benchmark Comparison (Serial) ---")
    get_system_info()
    sizes = [1e6, 1e7, 1e8] # Added 1e8 back for a more complete picture

    print(f"{'Target':>12} | {'Optimized Rate':>18} | {'Baseline Rate':>15} | {'Speedup':>10}")
    print("-" * 65)

    for size in sizes:
        target_n = int(size)
        
        # --- Common Preprocessing ---
        is_prime = fast_sieve(target_n)
        all_primes_arr = np.where(is_prime)[0]
        type_a, type_b = extract_primes_classified(is_prime, target_n)
        del is_prime; gc.collect()

        # --- Optimized Run ---
        hash_a, hash_b = create_hash_tables(type_a, type_b)
        # First run to compile the function
        serial_benchmark_optimized(100, type_a, type_b, hash_a, hash_b) 
        start_time = time.time()
        serial_benchmark_optimized(target_n, type_a, type_b, hash_a, hash_b)
        elapsed_opt = time.time() - start_time
        rate_opt = (target_n / 2) / elapsed_opt if elapsed_opt > 0 else 0
        del hash_a, hash_b; gc.collect()

        # --- Baseline Run ---
        all_primes_hash = create_hash_table_all(all_primes_arr)
        # First run to compile the function
        serial_benchmark_baseline(100, all_primes_arr, all_primes_hash)
        start_time = time.time()
        serial_benchmark_baseline(target_n, all_primes_arr, all_primes_hash)
        elapsed_base = time.time() - start_time
        rate_base = (target_n / 2) / elapsed_base if elapsed_base > 0 else 0
        del all_primes_hash; gc.collect()

        speedup = rate_opt / rate_base if rate_base > 0 else float('inf')

        print(f"{size:12.0e} | {rate_opt:15,.0f} n/s | {rate_base:12,.0f} n/s | {speedup:9.2f}x")

def quick_test():
    """Runs a quick verification up to 1 million."""
    print("--- Running Quick Test (target: 1,000,000) ---")
    run_verification(1_000_000)

def is_interactive():
    """Checks if the script is running in an interactive session."""
    import __main__ as main
    return not hasattr(main, '__file__')

if __name__ == "__main__":
    if is_interactive():
        print("Script is running in an interactive session (e.g., Jupyter or Colab).")
        print("You can now call the following functions from a cell:")
        print("  - run_verification(target_n): Run the full verification to a target.")
        print("      Example: run_verification(1e7)")
        print("  - run_benchmark_comparison(): Compare optimized vs. baseline method.")
        print("  - quick_test(): Run a small verification to 1 million.")
        print("  - get_system_info(): Display system details.")
    else:
        parser = argparse.ArgumentParser(description="High-Performance Goldbach Conjecture Verifier.")
        parser.add_argument("target", type=float, help="The target number for verification (e.g., 1e10).")
        parser.add_argument("--chunk_size", type=float, default=1e8, help="The size of each verification chunk (e.g., 1e8).")
        args = parser.parse_args()
        run_verification(int(args.target), int(args.chunk_size))

