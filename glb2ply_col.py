import os
import numpy as np
import trimesh
import argparse

def convert_glb_to_ply(input_glb_path, output_ply_path, debug=True):
    """
    Convert a GLB file containing a point cloud to a PLY point cloud file,
    ensuring color information is preserved.
    """
    try:
        # Load the GLB file
        print(f"Loading GLB file from: {input_glb_path}")
        scene = trimesh.load(input_glb_path)
        
        if debug:
            print(f"Loaded object type: {type(scene)}")
        
        # Check if it's a point cloud directly
        if isinstance(scene, trimesh.PointCloud):
            print("GLB file contains a direct point cloud.")
            point_cloud = scene
            # Check if colors are present
            if hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
                print(f"Direct point cloud has color data: {point_cloud.colors.shape}")
            else:
                print("Direct point cloud has no color data")
        elif isinstance(scene, trimesh.Scene):
            # Look for point clouds in the scene
            print("Searching for point clouds in the scene...")
            point_clouds = []
            
            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.PointCloud):
                    print(f"Found point cloud: {name} with {len(geometry.vertices)} points")
                    # Check for colors in the point cloud
                    has_colors = False
                    
                    # Method 1: Direct colors attribute
                    if hasattr(geometry, 'colors') and geometry.colors is not None:
                        has_colors = True
                        print(f"  - Point cloud has colors attribute with {len(geometry.colors)} values")
                    
                    # Method 2: Colors in visual attribute
                    elif hasattr(geometry, 'visual'):
                        if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                            geometry.colors = geometry.visual.vertex_colors
                            has_colors = True
                            print(f"  - Point cloud has vertex colors in visual with {len(geometry.visual.vertex_colors)} values")
                    
                    point_clouds.append(geometry)
                    
                    if not has_colors:
                        print("  - No color data found for this point cloud")
                elif hasattr(geometry, 'vertices') and len(geometry.vertices) > 0:
                    print(f"Found geometry with vertices: {name}, creating point cloud")
                    pc = trimesh.PointCloud(geometry.vertices)
                    
                    # Try different methods to extract colors
                    has_colors = False
                    
                    # Method 1: Direct color attribute
                    if hasattr(geometry, 'colors') and geometry.colors is not None:
                        pc.colors = geometry.colors
                        has_colors = True
                        print(f"  - Copied colors attribute with {len(geometry.colors)} values")
                    
                    # Method 2: Colors from visual attribute
                    elif hasattr(geometry, 'visual'):
                        # For trimesh objects
                        if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                            pc.colors = geometry.visual.vertex_colors
                            has_colors = True
                            print(f"  - Copied vertex colors with {len(geometry.visual.vertex_colors)} values")
                    
                    point_clouds.append(pc)
                    
                    if not has_colors:
                        print("  - No color data found for this geometry")
            
            if point_clouds:
                # Combine all point clouds
                if len(point_clouds) > 1:
                    print(f"Combining {len(point_clouds)} point clouds...")
                    all_points = []
                    all_colors = []
                    has_any_colors = False
                    
                    for pc in point_clouds:
                        all_points.append(pc.vertices)
                        
                        # Check for colors
                        if hasattr(pc, 'colors') and pc.colors is not None and len(pc.colors) > 0:
                            # Ensure colors match vertices
                            colors = pc.colors
                            if len(colors) != len(pc.vertices):
                                # Repeat or truncate colors to match vertices
                                if len(colors) < len(pc.vertices):
                                    # Tile the colors to match
                                    repeat_count = int(np.ceil(len(pc.vertices) / len(colors)))
                                    colors = np.tile(colors, (repeat_count, 1))[:len(pc.vertices)]
                                else:
                                    # Truncate
                                    colors = colors[:len(pc.vertices)]
                            
                            all_colors.append(colors)
                            has_any_colors = True
                        else:
                            # If no colors, use white as default
                            white_colors = np.tile([255, 255, 255, 255], (len(pc.vertices), 1))
                            all_colors.append(white_colors)
                    
                    # Stack all points
                    combined_points = np.vstack(all_points)
                    point_cloud = trimesh.PointCloud(combined_points)
                    
                    # Combine colors
                    if has_any_colors:
                        try:
                            combined_colors = np.vstack(all_colors)
                            point_cloud.colors = combined_colors
                            print(f"Combined colors: {combined_colors.shape}")
                        except Exception as e:
                            print(f"Error combining colors: {str(e)}")
                else:
                    point_cloud = point_clouds[0]
            else:
                # If no point clouds found, try to create one from all vertices
                all_vertices = []
                
                for name, geometry in scene.geometry.items():
                    if hasattr(geometry, 'vertices') and len(geometry.vertices) > 0:
                        print(f"Extracting vertices from {name}")
                        all_vertices.append(geometry.vertices)
                
                if all_vertices:
                    combined_vertices = np.vstack(all_vertices)
                    print(f"Created point cloud from {len(combined_vertices)} combined vertices")
                    point_cloud = trimesh.PointCloud(combined_vertices)
                else:
                    print("No point data found in the scene")
                    return False
        else:
            print(f"Unsupported file format: {type(scene)}")
            return False
        
        # Final check for colors
        if hasattr(point_cloud, 'colors') and point_cloud.colors is not None and len(point_cloud.colors) > 0:
            print(f"Final point cloud has colors: {point_cloud.colors.shape}")
        else:
            print("Warning: Final point cloud has no color data")
        
        # Export the point cloud - without the custom keyword arguments
        print(f"Saving point cloud to: {output_ply_path}")
        point_cloud.export(output_ply_path)
        
        # Verify the export worked
        if os.path.exists(output_ply_path):
            file_size = os.path.getsize(output_ply_path) / (1024 * 1024)  # Size in MB
            print(f"✅ Conversion successful! PLY file saved with {len(point_cloud.vertices)} points")
            print(f"   File size: {file_size:.2f} MB")
            return True
        else:
            print("❌ Export failed: PLY file was not created")
            return False
    
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        return False

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert GLB point cloud file to PLY with color preservation')
    parser.add_argument('--input', default='data/lunarbed.glb', help='Path to input GLB file')
    parser.add_argument('--output', default='output/lunarbed_colored.ply', help='Path for output PLY file')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the conversion
    convert_glb_to_ply(args.input, args.output, args.debug)

if __name__ == "__main__":
    main()