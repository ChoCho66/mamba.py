- D = 64
- ED = 4D
  - E 預設 4
- dt_rank = ED/16
- N = 16


- self.in_proj: nn.Linear(D, 2ED)
  - D * 2 * ED
- self.x_proj: nn.Linear(ED, dt_rank+2D)
  - ED * (dt_rank+2N)
- self.dt_proj: nn.Linear(dt_rank, ED)
  - dt_rank * ED + ED
- self.A_log = nn.Parameter(torch.log(A)) (ED, N)
  - ED * N
- self.D = nn.Parameter(torch.ones(config.d_inner)) (ED, )
  - ED
- self.out_proj: nn.Linear(ED, D)
  - ED * D

