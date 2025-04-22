import h5py
import numpy as np
import open3d as o3d
import os

# Step 1: Load the HDF file
hdf_file = 'framesets.hdf'
output_dir = 'point_clouds'
os.makedirs(output_dir, exist_ok=True)

# Camera intrinsics (update with your actual values)
fx = 600  # focal length in x
fy = 600  # focal length in y
cx = 320  # center x
cy = 240  # center y
depth_scale = 1000.0  # if depth is in millimeters

# Create pinhole camera intrinsic object
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)

# Create a combined point cloud for the entire scene
combined_pcd = o3d.geometry.PointCloud()

with h5py.File(hdf_file, 'r') as f:
    # Get all frame indices (assuming they are numbered sequentially)
    frame_indices = list(f['depth'].keys())
    
    print(f"Found {len(frame_indices)} frames in the HDF file")
    
    for frame_idx in frame_indices:
        print(f"Processing frame {frame_idx}...")
        
        # Load depth and color data
        depth = f['depth'][frame_idx][:].astype(np.float32)
        color = f['color'][frame_idx][:]
        
        # Create Open3D RGBD image
        depth_o3d = o3d.geometry.Image((depth / depth_scale).astype(np.float32))
        color_o3d = o3d.geometry.Image(color)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,  # already scaled
            convert_rgb_to_intensity=False
        )
        
        # Generate point cloud for this frame
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        
        # Option 1: Save individual frame as PLY
        frame_path = os.path.join(output_dir, f"frame_{frame_idx}.ply")
        o3d.io.write_point_cloud(frame_path, pcd)
        
        # Option 2: Add to combined point cloud
        # If you have transformations between frames, apply them here
        combined_pcd += pcd

# Save the combined point cloud
combined_path = os.path.join(output_dir, "combined_scene.ply")
o3d.io.write_point_cloud(combined_path, combined_pcd)

print(f"✅ Individual point clouds saved to {output_dir}/")
print(f"✅ Combined point cloud saved as {combined_path}")
