import torch
import math
import numpy as np

from pytorch3d.renderer import(
    FoVPerspectiveCameras,
    PointLights, 
    look_at_view_transform, 
    NDCGridRaysampler,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# define 10 cameras, all pointing to the object at the center of the world coordinates
num_views: int = 10
azimuth_range: float = 180.0
elev = torch.linspace(0, 0, num_views)
azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180 
lights = PointLights(device = device, location = [[0.0, 0.0, -3.0]])
R, T = look_at_view_transform(dist = 2.7, elev = elev, azim = azim)

cameras = FoVPerspectiveCameras(device = device, R = R, T = T)

# define the ray sampler
image_size = 64
volume_extent_world = 3
raysampler = NDCGridRaysampler(
    image_width = image_size,
    image_height = image_size,
    n_pts_per_ray= 50,
    min_depth = 0.1,
    max_depth = volume_extent_world,
)
# passing the camera locations to ray sampler
ray_bundle = raysampler(cameras)
print('ray_bundle origins tensor shape = ', ray_bundle.origins.shape)
print('ray_bundle directions tensor shape = ', ray_bundle.directions.shape)
print('ray_bundle lengths tensor shape = ', ray_bundle.lengths.shape)
print('ray_bundle xys tensor shape = ', ray_bundle.xys.shape)

torch.save({'ray_bundle': ray_bundle}, 'ray_sampling.pt')




