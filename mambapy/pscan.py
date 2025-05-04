import math

import torch
import torch.nn.functional as F
from c66 import pps, pp

"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""

def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)
        # only supports L that is a power of two (mainly for a clearer code)
        # 使用前要先把 A,X 填充成 L 是 2^power 次方的長度

        # 這函數會修改 X 的值
        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
        
        print("最初的X")
        pps(X)
        pp(X[0,0,:3,:3])
        pp(X.abs().mean())
        
        B, D, L, N = A.size()
        num_steps = int(math.log2(L))
        pp(B,D,L,N)

        # up sweep (last 2 steps unfolded)
        Aa = A # (B, D, L, N)
        Xa = X # (B, D, L, N)
        for _ in range(num_steps-2):
            T = Xa.size(2) # T = L
            Aa = Aa.view(B, D, T//2, 2, -1) # (B, D, L//2, 2, N)
            Xa = Xa.view(B, D, T//2, 2, -1) # (B, D, L//2, 2, N)
            # print("把 Aa, Xa 分成兩塊")
            # pps(Xa)
            
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1] # (B, D, L//2, N)
            Xa = Xa[:, :, :, 1] # (B, D, L//2, N)
            # print("把 Aa[:, :, :, 1], Xa[:, :, :, 1] 當成新的 Aa, Xa")
            # pps(Xa)

        # we have only 4, 2 or 1 nodes left
        # 這裡已經不需要再切割成 A0, A1, X0, X1
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        # 一開始長度就很小了 直接不需要 down sweep
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return
        
        print("up sweep finish.")
        print("中間的X")
        pps(Xa) # (B, D, 4 ,N)
        pps(X)  # (B, D, L, N)
        pp(X[0,0,:3,:3])
        pp(X.abs().mean())
        print("down sweep begin")

        # down sweep (first 2 steps unfolded)
        # A[:,:, start : end : steps ]
        # L = 2 ** num_steps
        Aa = A[:, :, 2**(num_steps-2)-1 : L : 2**(num_steps-2)] # (B, D, 4 ,N)
        Xa = X[:, :, 2**(num_steps-2)-1 : L : 2**(num_steps-2)] # (B, D, 4 ,N)
        # pps(Xa)
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        # k 從 num_steps-3, num_steps-4, ... , 1, 0
        for k in range(num_steps-3, -1, -1):
            
            # Xa: (B, D, ell, N)
            # pps(Xa)
            
            # steps = 2**k
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]
            # pps(Xa)
            # pp(max(0, (L - 2**k) // 2**k + 1))
            
            # Xa: (B, D, 2ell, N) 
            # 2ell = max(0, (L - 2**k) // 2**k + 1) = [ L-2^k / 2^k ] + 1 = 2^(num_steps-k)
            # 所以 2ell: 2^3, 2^4, ... , 2^num_steps 

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])
        
        pps(Xa)
        print("結束的X")
        pps(X)
        pp(X[0,0,:3,:3])
        pp(X.abs().mean())
        # Xa: (B, D, L/2, 2, N)
        # X: (B, D, L, N)

    @staticmethod
    def pscan_rev(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
                    
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """
        # (B, L, D, N)
        # mamba 使用時是 D 取 ED
        # 實際會使用的是 hs = pscan(deltaA, BX) # (B, L, ED, N)
        # deltaA, BX 都是 (B, L, ED, N)
        
        L = X_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in) # (B, npo2(L), D, N)
            X = pad_npo2(X_in) # (B, npo2(L), D, N)
        
        # prepare tensors
        A = A.transpose(2, 1) # (B, D, npo2(L), N)
        X = X.transpose(2, 1) # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        # 不會 return 東西
        # 會去修改 X 的內容
        PScan.pscan(A, X)
        
        # print("-------")
        # pps(X)
        # pp(X[0,0,:3,:3])
        # pp(X.abs().mean())
        # PScan.pscan(A, X)
        # pps(X)
        # pp(X[0,0,:3,:3])
        # pp(X.abs().mean())
        # print("-------")

        ctx.save_for_backward(A_in, X)
        
        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L] # (B, L, D, N)
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1) # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    
pscan = PScan.apply
