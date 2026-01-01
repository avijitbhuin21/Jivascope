"""
System Information Script - Prints detailed hardware and software info.
"""

import platform
import psutil
import os


def get_size(bytes_val, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes_val < factor:
            return f"{bytes_val:.2f}{unit}{suffix}"
        bytes_val /= factor


def print_system_info():
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print("\n--- OS Info ---")
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    print("\n--- CPU Info ---")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical Cores: {psutil.cpu_count(logical=True)}")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    print("\n--- Memory Info ---")
    svmem = psutil.virtual_memory()
    print(f"Total RAM: {get_size(svmem.total)}")
    print(f"Available RAM: {get_size(svmem.available)}")
    print(f"Used RAM: {get_size(svmem.used)}")
    print(f"RAM Usage: {svmem.percent}%")
    
    swap = psutil.swap_memory()
    print(f"Swap Total: {get_size(swap.total)}")
    print(f"Swap Used: {get_size(swap.used)}")
    
    print("\n--- Disk Info ---")
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            print(f"  {partition.device}")
            print(f"    Total: {get_size(partition_usage.total)}")
            print(f"    Free: {get_size(partition_usage.free)}")
        except PermissionError:
            continue
    
    print("\n--- Python Info ---")
    print(f"Python Version: {platform.python_version()}")
    
    print("\n--- PyTorch/CUDA Info ---")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {get_size(props.total_memory)}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
    except ImportError:
        print("PyTorch not installed")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_system_info()
