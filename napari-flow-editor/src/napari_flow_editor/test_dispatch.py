import numpy as np
import dask.array as da
import sys
import os

# --- AUTO-IMPORT FIX ---
# This helps python find your file wherever it is
sys.path.append(os.getcwd()) 

# 👇 MODIFY THIS LINE to match where your decorators.py actually lives!
# If it is in src/napari_flow_editor/decorators.py:
from flow_nodes.decorator import smart_compute
# OR if it is in nodes/decorators.py:
# from nodes.decorators import smart_compute

# --- 1. Define Mock Functions ---
def dask_worker(image):
    return "I am a Lazy Dask Result"

@smart_compute(dask_func=dask_worker)
def numpy_worker(image):
    return "I am a RAM Numpy Result"

# --- 2. Run Tests ---
def test_logic():
    print("--- Testing Smart Compute Dispatcher ---")

    # TEST A: Numpy
    ram_data = np.zeros((10, 10))
    result_a = numpy_worker(ram_data)
    print(f"Input: Numpy -> Output: '{result_a}'")
    assert result_a == "I am a RAM Numpy Result", "FAILED: Did not route to Numpy!"

    # TEST B: Dask
    lazy_data = da.zeros((10, 10))
    result_b = numpy_worker(lazy_data)
    print(f"Input: Dask  -> Output: '{result_b}'")
    assert result_b == "I am a Lazy Dask Result", "FAILED: Did not route to Dask!"
    
    print("\n✅ SUCCESS: The Dispatcher is routing correctly!")

if __name__ == "__main__":
    test_logic()