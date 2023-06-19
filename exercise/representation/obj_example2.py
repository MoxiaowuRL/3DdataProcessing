import open3d
from pytorch3d.io import load_obj
import torch

mesh_file = "cube_texture.obj"

print('visualizing the mesh using open3D')
mesh = open3d.io.read_triangle_mesh(mesh_file)
open3d.visualization.draw_geometries([mesh],
                mesh_show_back_face=True,
                mesh_show_wireframe=True)

print("Loading the same file using pytorch3D")
verts, faces, aux = load_obj(mesh_file)
print('Type of verts = ', type(verts))  # <class 'torch.Tensor'>
print('Type of faces = ', type(faces))  # <class 'torch.Tensor'>
print('Type of aux = ', type(aux))      # <class 'dict'>

print('faces = ', faces)
print('verts = ', verts)    
print('aux = ', aux)

texture_image = getattr(aux, 'texture_images')

print('texture_image type = ', type(texture_image))
print('texture_image = ', texture_image)
print(texture_image['Skin'].shape)


