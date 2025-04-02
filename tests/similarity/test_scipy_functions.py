import torch
import numpy as np
from numba import cuda
import numba as nb
from simms.similarity.scipy_functions import compile_linear_sum_assignment_kernel, RECT_LSAP_INFEASIBLE

import scipy.optimize

def test_hungarian():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda'
    N_MEM = 20
    lsap_kernel = compile_linear_sum_assignment_kernel(N_MEM)

    @cuda.jit(debug=True, opt=False)
    def rlsap_kernel(cost_batch, meta_batch, outp_batch, signal_batch_d ):
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if tid < cost_batch.shape[0]:
            nc = meta_batch[tid, 0] # number of columns
            nr = meta_batch[tid, 1] # number of rows
            cost = cost_batch[tid] # flattened cost matrix
            signal = lsap_kernel(nc, nr, cost, outp_batch[tid, 0], outp_batch[tid, 1])
            signal_batch_d[tid,0] = signal # if signal is 0, all is well. -100 if LSAP is infeasible

    cost = torch.tensor([
        [4, 1, 3],
        [2, 0, 5], 
        [3, 2, 2]
    ], 
    dtype=torch.int32,
    device=device)

    n_problems = 8
    cost_batch_d = torch.zeros((n_problems, 20), dtype=torch.int32, device=device)
    meta_batch_d = torch.zeros((n_problems, 2), dtype=torch.int32, device=device)
    meta_batch_d[0, 0] = 3  # as it's 3x3
    meta_batch_d[0, 1] = 3  # as it's 3x3
    cost_batch_d[0, :cost.numel()] = torch.asarray(cost.ravel(), device=device)
    # shaped like N_problems, (rows,cols), max_rows_cols
    outp_batch_d = torch.zeros((n_problems, 2, 20), dtype=torch.int32, device=device)
    # shall contain RLSAP_INFEASIBLE (-100) if assignment is infeasible
    signal_batch_d = torch.zeros((n_problems, 1), dtype=torch.int32, device=device) 

    threads_per_block = 1#32
    blocks_per_grid = 1# (n_problems + threads_per_block - 1) // threads_per_block

    rlsap_kernel[blocks_per_grid, threads_per_block](
        cuda.as_cuda_array(cost_batch_d), 
        cuda.as_cuda_array(meta_batch_d), 
        cuda.as_cuda_array(outp_batch_d),
        cuda.as_cuda_array(signal_batch_d),
    )

    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    assert row_ind.tolist() == outp_batch_d[0,0,:3].tolist()
    assert col_ind.tolist() == outp_batch_d[0,1,:3].tolist()
    assert signal_batch_d[0,0] == 0
    # total = cost[row_ind, col_ind].sum()
    