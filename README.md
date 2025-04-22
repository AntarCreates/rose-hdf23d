# 🌌 ROSE-HDF23D: HDF to 3D Point Clouds Pipeline

This repository provides a complete and modular pipeline to convert `.hdf` RGB and depth frame data into filtered and colored 3D point clouds. It supports both ground-truth and monocular depth estimation using [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main), and integrates with [Facebook's VGGT Transformer](https://huggingface.co/spaces/facebook/vggt) for transformer-based 3D scene reconstruction.


## 🔁 Workflow Overview

### 1. 🎞 Convert `.hdf` to RGB video

Use `hdf2vid_trim.py` to generate a video from the `color` dataset in an HDF file:
```bash
python hdf2vid_trim.py
```
> Outputs: `output/rgb_trimmed_output.mp4`

---

### 2. 🧠 Generate 3D GLB from RGB video

Upload the generated video to Facebook’s VGGT Visual Geometry Grounded Transformer on HuggingFace:

🔗 https://huggingface.co/spaces/facebook/vggt

> This returns a `.glb` file representing a 3D point cloud.

---

### 3. 📦 Convert `.glb` to `.ply` (with color)

Use `glb2ply_col.py` to convert the `.glb` to a `.ply` file while preserving color:

```bash
python glb2ply_col.py --input data/lunarbed.glb --output output/lunarbed_colored.ply
```

---

### 4. 🧹 Filter and downsample `.ply` file

Use `pointcloud_filter.py` to remove statistical outliers and optionally downsample the point cloud:

```bash
python pointcloud_filter.py output/lunarbed_colored.ply --voxel_size 0.01 --visualize
```

---

### 5. 📡 Generate point clouds from HDF RGB+Depth

Use `hdf2pcl.py` to create point clouds for each frame using RGB and depth data from HDF:

```bash
python hdf2pcl.py
```
> Saves output in `point_clouds/gt/` and a combined scene as `combined_scene.ply`.

---

### 6. 🌊 Monocular Pointclouds with Depth-Anything V2

Clone the DepthAnythingV2 repo from:
🔗 https://github.com/DepthAnything/Depth-Anything-V2/tree/main

Place the `hdfrgb2pcl.py` script in:
```
Depth-Anything-V2/metric_depth/
```

Run the script like so:
```bash
python hdfrgb2pcl.py --load-from <path_to_weights> --hdf-file data/framesets.hdf
```

> Outputs are saved to `point_clouds/monocular/`

---

## 🧪 Comparing Results

Use the `--visualize` flag in `pointcloud_filter.py` to visually compare filtered vs. unfiltered point clouds:
```bash
python pointcloud_filter.py input_file.ply --visualize
```

You can compare corresponding frames between `point_clouds/gt/` and `point_clouds/monocular/` to evaluate performance.

---

## 🛠️ Requirements

Install dependencies:
```bash
pip install open3d h5py numpy opencv-python trimesh torch
```

DepthAnythingV2 has additional requirements; refer to their setup guide:
🔗 https://github.com/DepthAnything/Depth-Anything-V2

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Antar Mazumder**

For inquiries or contributions, feel free to open an issue or pull request!
