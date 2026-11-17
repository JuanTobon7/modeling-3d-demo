"""Malla con Open3D: limpieza, normales orientadas hacia las cámaras, Poisson y recortes."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .common import PipelineError, ProgressFn
from .dense import DenseCloud


def build_mesh(cloud: DenseCloud, progress: ProgressFn, depth: int = 9,
               max_triangles: int = 250_000) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    if len(cloud.points) < 200:
        raise PipelineError(f"Nube densa muy pequeña ({len(cloud.points)} puntos).")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud.points))
    pcd.colors = o3d.utility.Vector3dVector(cloud.colors.astype(np.float64) / 255.0)
    diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    voxel = diag / 400.0
    progress("mesh", 0.1, "Limpiando nube de puntos")
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=voxel * 4)
    if len(pcd.points) < 200:
        raise PipelineError("Quedaron muy pocos puntos tras filtrar la nube.")

    progress("mesh", 0.3, "Estimando normales")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30))
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    _, nn = cKDTree(cloud.points).query(P)
    to_cam = cloud.centers[nn] - P
    flip = np.sum(N * to_cam, axis=1) < 0
    N[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(N)

    progress("mesh", 0.5, f"Reconstrucción de Poisson (profundidad {depth})")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.08))
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)

    progress("mesh", 0.75, "Quitando fragmentos sueltos")
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    if len(counts):
        small = counts[clusters] < 0.05 * counts.max()
        mesh.remove_triangles_by_mask(small)
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise PipelineError("La malla quedó vacía.")
    progress("mesh", 1.0, f"Malla: {len(mesh.vertices)} vértices, {len(mesh.triangles)} triángulos")
    return mesh, pcd
