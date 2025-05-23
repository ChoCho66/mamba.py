import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.pscan import pscan

from c66 import pp, pps
# print, show_print
# show_print = False

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
- the convolution is done with torch.nn.Conv1d
- the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : 
  - its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
  - First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
  - Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
  - We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto' # 預設 D//16
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16) # D//16

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        print("---"*10)
        print("In ResidualBlock")
        print("x.shape:", x.shape)
        print("self.norm(x).shape:", self.norm(x).shape)
        print("---"*10)
        
        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        # nn.Linear(D,2ED)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        # nn.Linear(ED, dt_rank+2N)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        # nn.Linear(dt_rank, ED)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            # 手動地把 self.dt_proj 這個線性層（nn.Linear）的 bias 欄位設成 inv_dt 的值。
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        # d_state = N, d_inner = ED
        # 1,2, ..., N
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1) # (ED, N)
        # A 的 shape 為 (ED, N) 是因為
        # 對於每個 ED, 程式碼的 A 是 A 矩陣 NxN 的 對角線的值
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        # nn.Linear(ED,D)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        print("---"*10)
        print("In MambaBlock")
        
        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        # print(x.shape)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        # print(x.shape)
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x) # (B, L, ED)
        y = self.ssm(x, z) # (B, L, ED)

        if self.config.use_cuda:
            output = self.out_proj(y) # (B, L, D)
            return output # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = F.silu(z) # (B, L, ED)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        print("---"*10)

        return output
    
    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        # A 是
        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        # dt_rank = math.ceil(d_model / 16)
        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        
        print("B, L, ED, N, dt_rank:", \
            x.shape[0], x.shape[1], x.shape[2], A.shape[-1], deltaBC.shape[-1] - 2 * A.shape[-1])

        # 這裡很重要
        # deltaBC, B, C is depend of the input x
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        
        # 在這沒作用 也就是 identity map
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # print("delta.shape, B.shape, C.shape:", delta.shape, B.shape, C.shape)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            # print("x.shape, delta.shape, B.shape, C.shape, z.shape:", x.shape, delta.shape, B.shape, C.shape, z.shape)
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)
            print("x.shape, delta.shape, B.shape, C.shape, z.shape:", x.shape, delta.shape, B.shape, C.shape, z.shape)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            # selective_scan_cuda
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
            print("y.shape:", y.shape, "(B,L,ED)")
        
        else:
            delta = delta.transpose(1, 2) # (B, L, ED)
            # softplus: https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
            # softplus(x) = 1/β * log ( 1 + exp( β * x ) )
            # pps(self.dt_proj.bias)
            delta = F.softplus(delta + self.dt_proj.bias)

            pp(x.shape, delta.shape, A.shape, B.shape, C.shape, z.shape)
            
            # use parallel scan mode or sequential mode when training
            # pscan: parallel scan
            if self.config.pscan:
                print("self.selective_scan")
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                print("self.selective_scan_seq")
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    # 為生產環境或高性能場景設計的實現，假設 pscan 是一個優化的並行掃描函數。
    # 可能用於實際部署的模型中，特別是在需要處理大量數據時。
    def selective_scan(self, x, delta, A, B, C, D):
        # ssm(x) = selective_scan(x, Δ, A, B, C, D)
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # return y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N) * (B, L, ED, 1) = (B, L, ED, N)
        
        hs = pscan(deltaA, BX) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1) -> (B, L, ED)
        # 最後兩維是 (ED, N) 和 (N, 1)，可以做矩陣乘法
        # 前面 (B, L) 維度保留
        # 矩陣乘法 (ED, N) @ (N, 1) → (ED, 1)

        y = y + D * x # (B, L, ED)

        return y
    
    # 參考實現或用於驗證的版本，因為它的邏輯更直接，容易檢查計算是否正確。
    # 可能用於測試或教育目的，或者在 pscan 不可用的情況下作為備用方案。
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    
    有關自動迴歸推斷

    使用 Mamba 的酷炫之處是：推斷對序列長度是常數
    我們只需要為每層 cached 兩個東西：
    - 隱藏狀態 h（它是（B, ED, N）），就像使用 RNN 做推斷時通常所做的一樣
    - 該層最後 d_conv-1 個輸入，以便計算時間維度上的 1D 卷積
    （d_conv 是固定的，所以這不會導致緩存隨著序列生成的增加）
    （而且 d_conv 通常非常小，例如 4，所以我們只需要「記住」最後 3 個輸入）

    具體來說，這兩個量被放入緩存 tuple 中，並分別命名為 h 和 inputs。
    h 是（B, ED, N），inputs 是（B, ED, d_conv-1）
    MambaBlock.step() 接收這個緩存，並且除了輸出輸出外，還輸出下一個呼叫的更新緩存。

    緩存物件初始化如下：（None, torch.zeros()）。
    當 h 是 None 時，選擇性掃描函數會檢測到它並以 h=0 開始。
    torch.zeros() 不是問題（它與只輸入相同，因為 conv1d 是填充的）

    由於我們需要每層一個緩存變數，所以我們存儲一個緩存物件列表，稱為 caches。（見 mamba_lm.py）
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
    