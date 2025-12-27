import zarr
import numpy as np
import dask.array as da
from skimage.transform import downscale_local_mean
import os
import shutil

def create_ome_zarr(filename="test_data.zarr", shape=(100, 1024, 1024)):
    """
    Creates a fake OME-Zarr dataset with 3 pyramid levels.
    Data: Random noise + a bright square in the middle (so we can see blur working).
    """
    
    # 1. Clean up old test data if it exists
    if os.path.exists(filename):
        shutil.rmtree(filename)
    
    store = zarr.DirectoryStore(filename)
    root = zarr.group(store=store)
    
    print(f"Generating synthetic data {shape}...")
    
    # 2. Generate synthetic data (Noise + Bright Square)
    # Using dask to generate it lazily, then compute to numpy for writing
    # (In a real scenario, this would be huge, but here we keep it manageable)
    data = np.random.random(shape).astype('float32')
    
    # Add a bright square feature in the middle so we can visually check blurring later
    z, y, x = shape
    data[z//2-10:z//2+10, y//2-100:y//2+100, x//2-100:x//2+100] = 5.0

    # 3. Write Level 0 (Full Resolution)
    # We create a dataset named '0' inside the root
    root.create_dataset('0', data=data, chunks=(10, 256, 256), overwrite=True)
    print(" - Level 0 written (Full Res)")

    # 4. Create Level 1 (Downsampled x2)
    # Simple mean downsampling
    data_lvl1 = downscale_local_mean(data, (1, 2, 2))
    root.create_dataset('1', data=data_lvl1, chunks=(10, 128, 128), overwrite=True)
    print(f" - Level 1 written (Shape: {data_lvl1.shape})")

    # 5. Create Level 2 (Downsampled x4)
    data_lvl2 = downscale_local_mean(data, (1, 4, 4))
    root.create_dataset('2', data=data_lvl2, chunks=(10, 64, 64), overwrite=True)
    print(f" - Level 2 written (Shape: {data_lvl2.shape})")

    # 6. Write OME-Zarr Metadata
    # This JSON tells Napari that '0', '1', and '2' are parts of the same image
    root.attrs['multiscales'] = [
        {
            "version": "0.4",
            "name": "test_image",
            "datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]},
                {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 2.0, 2.0]}]},
                {"path": "2", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 4.0, 4.0]}]}
            ],
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
        }
    ]
    print(f"✅ Success! Created '{filename}' with OME metadata.")

if __name__ == "__main__":
    create_ome_zarr()