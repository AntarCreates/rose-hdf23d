import open3d as o3d
import numpy as np
import argparse
import os

def filter_pointcloud(input_file, output_file, neighbors=20, std_ratio=2.0, voxel_size=None):
    """
    Filter a pointcloud using statistical outlier removal and optional voxel grid downsampling.
    
    Parameters:
    -----------
    input_file : str
        Path to the input PLY file
    output_file : str
        Path to save the filtered PLY file
    neighbors : int
        Number of nearest neighbors to consider for statistical outlier removal
    std_ratio : float
        Standard deviation ratio threshold for outlier removal
    voxel_size : float or None
        Size of voxel for downsampling. If None, no downsampling is performed.
    """
    print(f"Loading pointcloud: {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    
    # Display basic info about the pointcloud
    print(f"Original pointcloud contains {len(np.asarray(pcd.points))} points")
    
    # Apply statistical outlier removal
    print(f"Applying statistical outlier removal (neighbors={neighbors}, std_ratio={std_ratio})...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    print(f"After statistical filtering: {len(np.asarray(filtered_pcd.points))} points")
    
    # Apply voxel grid downsampling if specified
    if voxel_size is not None:
        print(f"Applying voxel grid downsampling (voxel_size={voxel_size})...")
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"After voxel grid downsampling: {len(np.asarray(filtered_pcd.points))} points")
    
    # Save the filtered pointcloud
    print(f"Saving filtered pointcloud to {output_file}")
    o3d.io.write_point_cloud(output_file, filtered_pcd)
    
    return filtered_pcd

def visualize_comparison(original_pcd, filtered_pcd):
    """
    Visualize the original and filtered pointclouds side by side.
    Fixed version using copy instead of clone.
    """
    # Create copies of the point clouds
    original_display = o3d.geometry.PointCloud()
    original_display.points = o3d.utility.Vector3dVector(np.asarray(original_pcd.points))
    if original_pcd.has_colors():
        original_display.colors = o3d.utility.Vector3dVector(np.asarray(original_pcd.colors))
    
    filtered_display = o3d.geometry.PointCloud()
    filtered_display.points = o3d.utility.Vector3dVector(np.asarray(filtered_pcd.points))
    if filtered_pcd.has_colors():
        filtered_display.colors = o3d.utility.Vector3dVector(np.asarray(filtered_pcd.colors))
    
    # Set colors if there are no original colors
    if not original_pcd.has_colors():
        original_display.paint_uniform_color([1, 0, 0])  # Red
    if not filtered_pcd.has_colors():
        filtered_display.paint_uniform_color([0, 1, 0])  # Green
    
    # Move the filtered point cloud to the right for side-by-side comparison
    points = np.asarray(filtered_display.points)
    max_x = np.max(np.asarray(original_display.points)[:, 0])
    min_x = np.min(np.asarray(original_display.points)[:, 0])
    offset = (max_x - min_x) * 1.2
    points[:, 0] += offset
    filtered_display.points = o3d.utility.Vector3dVector(points)
    
    # Visualize
    o3d.visualization.draw_geometries([original_display, filtered_display],
                                     window_name="Original (left) vs Filtered (right)",
                                     width=1200, height=800)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter noise from PLY pointcloud files")
    parser.add_argument("input_file", help="Path to input PLY file")
    parser.add_argument("--output_file", help="Path to output filtered PLY file")
    parser.add_argument("--neighbors", type=int, default=20, help="Number of neighbors for statistical outlier removal")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Standard deviation ratio for outlier removal")
    parser.add_argument("--voxel_size", type=float, help="Voxel size for downsampling (optional)")
    parser.add_argument("--visualize", action="store_true", help="Visualize before and after comparison")
    
    args = parser.parse_args()
    
    # If output file is not specified, create one in the same directory as input
    if args.output_file is None:
        base_name = os.path.basename(args.input_file)
        file_name, ext = os.path.splitext(base_name)
        args.output_file = os.path.join(os.path.dirname(args.input_file), f"{file_name}_filtered{ext}")
    
    # Filter the pointcloud
    original_pcd = o3d.io.read_point_cloud(args.input_file)
    filtered_pcd = filter_pointcloud(args.input_file, args.output_file, 
                                 neighbors=args.neighbors, 
                                 std_ratio=args.std_ratio,
                                 voxel_size=args.voxel_size)
    
    print(f"Pointcloud filtering complete!")
    print(f"Original points: {len(np.asarray(original_pcd.points))}")
    print(f"Filtered points: {len(np.asarray(filtered_pcd.points))}")
    
    # Visualize if requested
    if args.visualize:
        visualize_comparison(original_pcd, filtered_pcd)