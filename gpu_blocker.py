import torch
import time
import sys

# Configuration
TARGET_GB = 9

def occupy():
    if not torch.cuda.is_available():
        print("Error: No CUDA device found.")
        sys.exit(1)
        
    try:
        print(f"Allocating {TARGET_GB} GB CUDA memory...", flush=True)
        # Calculate number of float32 elements (4 bytes each)
        # 9 GB * 1024^3 bytes / 4 bytes per float
        num_elements = int((TARGET_GB * (1024**3)) / 4)
        
        # torch.empty is efficient: it reserves VRAM but doesn't initialize data (saves CPU RAM)
        buffer = torch.empty(num_elements, dtype=torch.float32, device='cuda:0')
        
        # Ensure allocation is complete
        torch.cuda.synchronize()
        
        print("Memory allocated. Sleeping forever...", flush=True)
        # Infinite sleep loop (minimal CPU usage)
        while True:
            time.sleep(60)
            
    except Exception as e:
        print(f"Failed to allocate memory: {e}")
        sys.exit(1)

if __name__ == "__main__":
    occupy()