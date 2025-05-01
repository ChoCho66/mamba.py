## SEMamba

(mixer): Mamba(
  65.28 K = 2.89% Params, 2.41 GMACs = 1.29% MACs, 4.82 GFLOPS = 1.28% FLOPs
  (in_proj): Linear(32.77 K = 1.45% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs, in_features=64, out_features=512, bias=False)
  (conv1d): Conv1d(1.28 K = 0.06% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs, 256, 256, kernel_size=(4,), stride=(1,), padding=(3,), groups=256)
  (act): SiLU(0 = 0% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs)
  (x_proj): Linear(9.22 K = 0.41% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs, in_features=256, out_features=36, bias=False)
  (dt_proj): Linear(1.28 K = 0.06% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs, in_features=4, out_features=256, bias=True)
  (out_proj): Linear(16.38 K = 0.73% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs, in_features=256, out_features=64, bias=False)
)

## mamba.py

MambaBlock(
  65.28 K = 100% Params, 1.7 GMACs = 100% MACs, 3.83 GFLOPS = 100% FLOPs
  (in_proj): Linear(32.77 K = 50.2% Params, 937.16 MMACs = 55.16% MACs, 1.87 GFLOPS = 48.88% FLOPs, in_features=64, out_features=512, bias=False)
  (conv1d): Conv1d(1.28 K = 1.96% Params, 29.59 MMACs = 1.74% MACs, 66.59 MFLOPS = 1.74% FLOPs, 256, 256, kernel_size=(4,), stride=(1,), padding=(3,), groups=256)
  (x_proj): Linear(9.22 K = 14.12% Params, 263.58 MMACs = 15.51% MACs, 527.16 MFLOPS = 13.75% FLOPs, in_features=256, out_features=36, bias=False)
  (dt_proj): Linear(1.28 K = 1.96% Params, 0 MACs = 0% MACs, 0 FLOPS = 0% FLOPs, in_features=4, out_features=256, bias=True)
  (out_proj): Linear(16.38 K = 25.1% Params, 468.58 MMACs = 27.58% MACs, 937.16 MFLOPS = 24.44% FLOPs, in_features=256, out_features=64, bias=False)
)

ssm 裡面 包括了 x_proj, dt_proj 以及剩下的 

- 4.35k (6.66%) Params, 
- 41.89 (0.02%) MMACs, 
- 476.13 (12.14%) MFLOPS

這裡頭包括 
- self.A_log
- self.D