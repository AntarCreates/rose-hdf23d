import argparse
import numpy as np
import open3d as o3d
import h5py
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def load_hdf_pointcloud(hdf_file, frame_idx, fx=600, fy=600, cx=320, cy=240, depth_scale=1000.0):
    """
    Load a single frame from HDF file and convert to pointcloud
    
    Parameters:
    -----------
    hdf_file : str
        Path to the HDF file containing depth and color data
    frame_idx : str
        Frame index to extract
    fx, fy : float
        Focal length in x and y
    cx, cy : float
        Principal point coordinates
    depth_scale : float
        Scale factor for depth values (e.g., 1000.0 for mm to m conversion)
        
    Returns:
    --------
    pcd : o3d.geometry.PointCloud
        Pointcloud generated from the HDF file
    """
    print(f"Loading frame {frame_idx} from HDF file: {hdf_file}")
    
    with h5py.File(hdf_file, 'r') as f:
        # Load depth and color data
        try:
            depth = f['depth'][frame_idx][:].astype(np.float32)
            color = f['color'][frame_idx][:]
            
            print(f"Loaded depth shape: {depth.shape}, color shape: {color.shape}")
            
        except KeyError as e:
            print(f"Error: {e}")
            print("Available keys in the HDF file:")
            print(list(f.keys()))
            return None
    
    # Create Open3D intrinsics
    height, width = depth.shape
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # Create Open3D RGBD image
    depth_o3d = o3d.geometry.Image((depth / depth_scale).astype(np.float32))
    color_o3d = o3d.geometry.Image(color)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,  # already scaled
        convert_rgb_to_intensity=False
    )
    
    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    
    print(f"Generated pointcloud with {len(pcd.points)} points from HDF data")
    
    return pcd

def calculate_pointcloud_metrics(mono_pcd, hdf_pcd):
    """
    Calculate comparison metrics between two pointclouds.
    
    Parameters:
    -----------
    mono_pcd : o3d.geometry.PointCloud
        Pointcloud from monocular depth estimation
    hdf_pcd : o3d.geometry.PointCloud
        Pointcloud from HDF depth data
        
    Returns:
    --------
    metrics : dict
        Dictionary containing various comparison metrics
    """
    # Convert to numpy arrays
    mono_points = np.asarray(mono_pcd.points)
    hdf_points = np.asarray(hdf_pcd.points)
    
    print(f"Computing metrics between pointclouds of sizes {len(mono_points)} and {len(hdf_points)}")
    
    # Build KD-trees for efficient nearest neighbor search
    mono_tree = KDTree(mono_points)
    hdf_tree = KDTree(hdf_points)
    
    # Calculate one-way chamfer distance (mono to hdf)
    distances_m_h, _ = hdf_tree.query(mono_points)
    chamfer_m_h = np.mean(distances_m_h**2)
    
    # Calculate one-way chamfer distance (hdf to mono)
    distances_h_m, _ = mono_tree.query(hdf_points)
    chamfer_h_m = np.mean(distances_h_m**2)
    
    # Bidirectional chamfer distance
    chamfer_distance = chamfer_m_h + chamfer_h_m
    
    # Calculate the median distance
    median_distance = np.median(distances_m_h)
    
    # Calculate percentage of points with distance less than thresholds
    thresholds = [0.01, 0.02, 0.05, 0.1]  # meters
    accuracy_metrics = {}
    for threshold in thresholds:
        accuracy = np.mean(distances_m_h < threshold) * 100
        accuracy_metrics[f'accuracy_{threshold}m'] = accuracy
    
    # Collect all metrics
    metrics = {
        'chamfer_distance': chamfer_distance,
        'one_way_chamfer_m_h': chamfer_m_h,
        'one_way_chamfer_h_m': chamfer_h_m,
        'median_distance': median_distance,
        'rmse': np.sqrt(np.mean(distances_m_h**2)),
        'mae': np.mean(np.abs(distances_m_h)),
        **accuracy_metrics
    }
    
    return metrics

def align_pointclouds(source_pcd, target_pcd, max_iterations=100, voxel_size=0.05):
    """
    Align source pointcloud to target pointcloud using ICP.
    
    Parameters:
    -----------
    source_pcd : o3d.geometry.PointCloud
        Source pointcloud to be aligned
    target_pcd : o3d.geometry.PointCloud
        Target pointcloud to align with
    max_iterations : int
        Maximum number of ICP iterations
    voxel_size : float
        Voxel size for downsampling
        
    Returns:
    --------
    aligned_source : o3d.geometry.PointCloud
        Aligned source pointcloud
    transformation : numpy.ndarray
        4x4 transformation matrix
    """
    print("Aligning pointclouds using ICP...")
    
    # Create copies of the pointclouds for downsampling
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # Initial alignment using point-to-point ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, max_correspondence_distance=voxel_size * 2,
        init=np.identity(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    # Apply transformation to original source pointcloud
    aligned_source = source_pcd.clone()
    aligned_source.transform(result_icp.transformation)
    
    print(f"ICP finished with fitness: {result_icp.fitness}, RMSE: {result_icp.inlier_rmse}")
    
    return aligned_source, result_icp.transformation

def visualize_comparison(mono_pcd, hdf_pcd, aligned=True, point_size=2.0):
    """
    Visualize both pointclouds for comparison with adjustable point size.
    
    Parameters:
    -----------
    mono_pcd : o3d.geometry.PointCloud
        Pointcloud from monocular depth estimation
    hdf_pcd : o3d.geometry.PointCloud
        Pointcloud from HDF depth data
    aligned : bool
        Whether the pointclouds have been aligned
    point_size : float
        Size of points for visualization
    """
    # Create copies for visualization
    mono_vis = o3d.geometry.PointCloud()
    mono_vis.points = o3d.utility.Vector3dVector(np.asarray(mono_pcd.points))
    if mono_pcd.has_colors():
        mono_vis.colors = o3d.utility.Vector3dVector(np.asarray(mono_pcd.colors))
    else:
        mono_vis.paint_uniform_color([0, 0.651, 0.929])  # Blue for mono pointcloud
    
    hdf_vis = o3d.geometry.PointCloud()
    hdf_vis.points = o3d.utility.Vector3dVector(np.asarray(hdf_pcd.points))
    if hdf_pcd.has_colors():
        hdf_vis.colors = o3d.utility.Vector3dVector(np.asarray(hdf_pcd.colors))
    else:
        hdf_vis.paint_uniform_color([1, 0.706, 0])  # Yellow for HDF pointcloud
    
    # Visualize
    title = "Aligned Pointclouds (Mono=Blue, HDF=Yellow)" if aligned else "Unaligned Pointclouds (Mono=Blue, HDF=Yellow)"
    
    # Create visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=1200, height=800)
    
    # Add geometry
    vis.add_geometry(mono_vis)
    vis.add_geometry(hdf_vis)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    # Set view control
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def plot_histogram(metrics, output_file):
    """
    Plot a histogram of point-to-point distances.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing comparison metrics
    output_file : str
        Path to save the histogram plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram of accuracy at different thresholds
    thresholds = [0.01, 0.02, 0.05, 0.1]
    accuracy_values = [metrics[f'accuracy_{t}m'] for t in thresholds]
    
    plt.bar([f'< {t}m' for t in thresholds], accuracy_values, color='skyblue')
    plt.ylabel('Percentage of Points (%)')
    plt.xlabel('Distance Threshold')
    plt.title('Accuracy at Different Distance Thresholds')
    
    # Add text with additional metrics
    info_text = (
        f"RMSE: {metrics['rmse']:.4f}m\n"
        f"MAE: {metrics['mae']:.4f}m\n"
        f"Median Distance: {metrics['median_distance']:.4f}m\n"
        f"Chamfer Distance: {metrics['chamfer_distance']:.4f}"
    )
    plt.figtext(0.7, 0.25, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Histogram saved to {output_file}")

def extract_color_from_hdf(hdf_file, frame_idx, output_dir):
    """
    Extract color image from HDF file to use with DepthAnythingV2.
    
    Parameters:
    -----------
    hdf_file : str
        Path to the HDF file
    frame_idx : str
        Frame index to extract
    output_dir : str
        Directory to save the color image
        
    Returns:
    --------
    image_path : str
        Path to the saved color image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf_file, 'r') as f:
        color = f['color'][frame_idx][:]
    
    image_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
    cv2.imwrite(image_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    
    print(f"Extracted color image saved to {image_path}")
    
    return image_path

def main():
    parser = argparse.ArgumentParser(description='Compare pointclouds from monocular depth estimation and HDF depth data.')
    parser.add_argument('--hdf-file', type=str, default='framesets.hdf',
                        help='Path to the HDF file containing depth and color data')
    parser.add_argument('--frame-idx', type=str, required=True,
                        help='Frame index to extract from the HDF file')
    parser.add_argument('--mono-pcd', type=str, required=True,
                        help='Path to the pointcloud generated from monocular depth estimation (.ply file)')
    parser.add_argument('--fx', type=float, default=600,
                        help='Focal length in x-direction')
    parser.add_argument('--fy', type=float, default=600,
                        help='Focal length in y-direction')
    parser.add_argument('--cx', type=float, default=320,
                        help='Principal point x-coordinate')
    parser.add_argument('--cy', type=float, default=240,
                        help='Principal point y-coordinate')
    parser.add_argument('--depth-scale', type=float, default=1000.0,
                        help='Scale factor for depth values (e.g., 1000.0 for mm to m conversion)')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                        help='Directory to save comparison results')
    parser.add_argument('--align', action='store_true', default=True,
                        help='Align the pointclouds before comparison')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the pointcloud comparison')
    parser.add_argument('--point-size', type=float, default=2.0,
                        help='Point size for visualization')
    parser.add_argument('--extract-color', action='store_true',
                        help='Extract color image from HDF file for use with DepthAnythingV2')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract color image if requested
    if args.extract_color:
        extract_color_from_hdf(args.hdf_file, args.frame_idx, os.path.join(args.output_dir, 'images'))
    
    # Load monocular-generated pointcloud
    print(f"Loading monocular-generated pointcloud: {args.mono_pcd}")
    mono_pcd = o3d.io.read_point_cloud(args.mono_pcd)
    print(f"Monocular pointcloud contains {len(mono_pcd.points)} points")
    
    # Generate pointcloud from HDF file
    hdf_pcd = load_hdf_pointcloud(
        args.hdf_file, 
        args.frame_idx,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale
    )
    
    if hdf_pcd is None:
        print("Failed to generate pointcloud from HDF file. Exiting.")
        return
    
    # Save the HDF-generated pointcloud
    hdf_pcd_path = os.path.join(args.output_dir, f'hdf_pointcloud_frame_{args.frame_idx}.ply')
    o3d.io.write_point_cloud(hdf_pcd_path, hdf_pcd)
    print(f"HDF-generated pointcloud saved to {hdf_pcd_path}")
    
    # Align the pointclouds if requested
    if args.align:
        aligned_mono_pcd, transformation = align_pointclouds(
            mono_pcd, hdf_pcd, voxel_size=0.05
        )
        # Save the aligned pointcloud
        aligned_pcd_path = os.path.join(args.output_dir, f'aligned_mono_pointcloud_frame_{args.frame_idx}.ply')
        o3d.io.write_point_cloud(aligned_pcd_path, aligned_mono_pcd)
        print(f"Aligned monocular pointcloud saved to {aligned_pcd_path}")
        
        # Calculate metrics using the aligned pointcloud
        metrics = calculate_pointcloud_metrics(aligned_mono_pcd, hdf_pcd)
        comparison_pcd = aligned_mono_pcd
    else:
        # Calculate metrics without alignment
        metrics = calculate_pointcloud_metrics(mono_pcd, hdf_pcd)
        comparison_pcd = mono_pcd
    
    # Print metrics
    print("\nComparison Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, f'comparison_metrics_frame_{args.frame_idx}.txt')
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    print(f"Metrics saved to {metrics_path}")
    
    # Plot histogram
    histogram_path = os.path.join(args.output_dir, f'accuracy_histogram_frame_{args.frame_idx}.png')
    plot_histogram(metrics, histogram_path)
    
    # Visualize if requested
    if args.visualize:
        visualize_comparison(comparison_pcd, hdf_pcd, aligned=args.align, point_size=args.point_size)

if __name__ == '__main__':
    main()