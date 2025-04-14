import torch
from mambapy.mamba import Mamba, MambaConfig
from torchinfo import summary
from c66 import pp

# B, L, D = 7, 64, 16
B, L, D = 7, 201, 286
pp(B,L,D)

config = MambaConfig(d_model=D, n_layers=1, use_cuda=False)
model = Mamba(config)
# .to("cuda")

x = torch.randn(B, L, D)
# .to("cuda")
y = model(x)

assert y.shape == x.shape
pp(x.shape)