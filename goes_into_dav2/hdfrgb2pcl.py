import argparse
import numpy as np
import open3d as o3d
import os
import h5py
import torch
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from RGB frames in an HDF file.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', default='', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--hdf-file', type=str, required=True,
                        help='Path to the HDF file containing the RGB frames.')
    parser.add_argument('--outdir', type=str, default='./point_clouds/monocular',
                        help='Directory to save the output point clouds.')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Open the HDF file and read the RGB frames
    with h5py.File(args.hdf_file, 'r') as f:
        # Get all frame indices (assuming they are numbered sequentially)
        frame_indices = list(f['color'].keys())

        print(f"Found {len(frame_indices)} frames in the HDF file.")

        # Process each RGB frame
        for frame_idx in frame_indices:
            print(f'Processing frame {frame_idx}...')

            # Load the RGB frame from the HDF file
            color = f['color'][frame_idx][:]  # Shape (height, width, 3)
            color_image = Image.fromarray(color)

            # Read the image and pass it through the model
            width, height = color_image.size
            image = np.array(color_image)

            # Generate depth map using DepthAnything
            pred = depth_anything.infer_image(image, height)

            # Resize depth prediction to match the original image size
            resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

            # Generate mesh grid and calculate point cloud coordinates
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / args.focal_length_x
            y = (y - height / 2) / args.focal_length_y
            z = np.array(resized_pred)
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.array(color_image).reshape(-1, 3) / 255.0

            # Create the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save the point cloud in the 'monocular' directory
            output_filename = os.path.join(args.outdir, f"frame_{frame_idx}.ply")
            o3d.io.write_point_cloud(output_filename, pcd)

            print(f"✅ Saved point cloud: {output_filename}")

if __name__ == '__main__':
    main()

