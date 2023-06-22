import open3d

print('visualzing the mesh using open3d')
pcd = open3d.io.read_point_cloud('pedestrian.ply')
open3d.visualization.draw_geometries([pcd],
                                     mesh_show_wireframe=True,) 