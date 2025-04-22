import h5py
import cv2
import numpy as np
import os

# Use relative paths
hdf_path = 'data/framesets.hdf'
output_dir = 'output'  # Just the directory name
output_file = 'rgb_trimmed_output.mp4'
output_path = os.path.join(output_dir, output_file)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Video parameters
frame_size = (640, 480)
fps = 30
start_second = 13
skip_frames = fps * start_second

with h5py.File(hdf_path, 'r') as f:
    color_group = f['color']
    frame_keys = sorted(color_group.keys())

    # Only keep frames after skipping
    usable_keys = frame_keys[skip_frames:]
    
    # Check if we have frames to process
    if not usable_keys:
        print("No frames found to process after skipping!")
        exit(1)
    
    print(f"Processing {len(usable_keys)} frames...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    if not out.isOpened():
        print(f"ERROR: Could not open video writer. Check path: {output_path}")
        exit(1)

    # Process each frame
    for i, key in enumerate(usable_keys):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing frame {i}/{len(usable_keys)}")
            
        rgb = color_group[key][:]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    
    # Verify file was created
    if os.path.exists(output_path):
        print(f"✅ Video successfully saved to {output_path}")
        print(f"Started from {start_second}s (skipped {skip_frames} frames).")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    else:
        print(f"❌ Failed to create video file at {output_path}")