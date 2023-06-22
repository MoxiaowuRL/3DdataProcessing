import torch
from pytorch3d.structures import Volumes
from pytorch3d.renderer.implicit.renderer import VolumeSampler

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

checkpoint = torch.load('ray_sampling.pt')
raybundle = checkpoint['ray_bundle']   

# define a volume, density shape = [batch_size, 64, 64, 50]
batch_size = 10
densities = torch.zeros([batch_size, 1, 64, 64, 64]).to(device)
colors = torch.zeros([batch_size, 3, 64, 64, 64]).to(device)

voxel_size = 0.1
volumes = Volumes(densities = densities, features = colors, voxel_size = voxel_size)

# define the volume sampler
volume_sampler = VolumeSampler(volumes = volumes, sample_mode = "bilinear")
rays_densities, rays_features = volume_sampler(raybundle)
print('rays_densities.shape = ', rays_densities.shape)
print('rays_features.shape = ', rays_features.shape)

rays_densities_shape = torch.Size([10, 64, 64, 50, 1])
rays_features_shape = torch.Size([10, 64, 64, 50, 3])

torch.save({
    'rays_densities': rays_densities,
    'rays_features': rays_features,
}, 'volume_sampling.pt')



  
