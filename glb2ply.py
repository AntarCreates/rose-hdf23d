import os
import numpy as np
import trimesh
import argparse

def convert_glb_to_ply(input_glb_path, output_ply_path, debug=True):
    """
    Convert a GLB file containing a point cloud to a PLY point cloud file.
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
        elif isinstance(scene, trimesh.Scene):
            # Look for point clouds in the scene
            print("Searching for point clouds in the scene...")
            point_clouds = []
            
            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.PointCloud):
                    print(f"Found point cloud: {name} with {len(geometry.vertices)} points")
                    point_clouds.append(geometry)
                elif hasattr(geometry, 'vertices') and len(geometry.vertices) > 0:
                    print(f"Found geometry with vertices: {name}, creating point cloud")
                    pc = trimesh.PointCloud(geometry.vertices)
                    # Try to copy colors if available
                    if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'vertex_colors'):
                        pc.colors = geometry.visual.vertex_colors
                    point_clouds.append(pc)
            
            if point_clouds:
                # Combine all point clouds
                if len(point_clouds) > 1:
                    print(f"Combining {len(point_clouds)} point clouds...")
                    all_points = []
                    all_colors = []
                    
                    for pc in point_clouds:
                        all_points.append(pc.vertices)
                        if hasattr(pc, 'colors') and pc.colors is not None:
                            # Match the colors array length to the vertices
                            colors = pc.colors
                            if len(colors) < len(pc.vertices):
                                # Repeat the last color if needed
                                colors = np.vstack([colors, np.tile(colors[-1], (len(pc.vertices) - len(colors), 1))])
                            all_colors.append(colors[:len(pc.vertices)])
                    
                    # Stack all points
                    combined_points = np.vstack(all_points)
                    point_cloud = trimesh.PointCloud(combined_points)
                    
                    # Combine colors if available for all point clouds
                    if all(len(c) > 0 for c in all_colors):
                        combined_colors = np.vstack(all_colors)
                        point_cloud.colors = combined_colors
                else:
                    point_cloud = point_clouds[0]
            else:
                # If no point clouds found, try to create one from all vertices in the scene
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
        
        # Export the point cloud
        print(f"Saving point cloud to: {output_ply_path}")
        point_cloud.export(output_ply_path)
        
        print(f"✅ Conversion successful! PLY file saved with {len(point_cloud.vertices)} points")
        return True
    
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        return False

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert GLB point cloud file to PLY')
    parser.add_argument('--input', default='data/lunarbed.glb', help='Path to input GLB file')
    parser.add_argument('--output', default='output/lunarbed.ply', help='Path for output PLY file')
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