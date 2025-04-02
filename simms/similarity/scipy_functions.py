import math
import numba
from numba import cuda
from numba import int32, float32, boolean
import numpy as np
import cupy as cp

RECT_LSAP_INFEASIBLE = -100

@cuda.jit(device=True)
def augmenting_path(
    nc, cost, u, v, path, row4col,
    shortestPathCosts, i, 
    SR, SC,
    remaining, p_minVal
):
    minVal = 0.0
    num_remaining = nc
    for it in range(nc):
        remaining[it] = nc - it - 1

    SR[:] = False
    SC[:] = False
    shortestPathCosts[:] = np.inf

    sink = -1
    while sink == -1:
        index = -1
        lowest = np.inf  # FIXED: Should be positive infinity
        SR[i] = True

        for it in range(num_remaining):
            j = remaining[it]
            r = minVal + cost[i * nc + j] - u[i] - v[j]
            if r < shortestPathCosts[j]:
                path[j] = i
                shortestPathCosts[j] = r

            if shortestPathCosts[j] < lowest or \
                (shortestPathCosts[j] == lowest and row4col[j] == -1):
                lowest = shortestPathCosts[j]
                index = it

        minVal = lowest
        if minVal == np.inf:  # FIXED: Correct infinity comparison
            return -1

        j = remaining[index]
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        SC[j] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]

    p_minVal[0] = minVal  # FIXED: Move this outside the loop
    return sink


def compile_linear_sum_assignment_kernel(N_MEM) -> callable:
    @cuda.jit(device=True)
    def linear_sum_assignment(nc, nr, cost, a, b):
        u = cuda.local.array((N_MEM,), float32)
        v = cuda.local.array((N_MEM,), float32)
        u[:] = 0
        v[:] = 0
        shortestPathCosts = cuda.local.array((N_MEM), float32)
        path = cuda.local.array((N_MEM,), int32)
        shortestPathCosts[:] = np.inf
        path[:] = -1
        col4row = cuda.local.array((N_MEM,), int32)
        row4col = cuda.local.array((N_MEM,), int32)
        col4row[:] = -1
        row4col[:] = -1
        SR = cuda.local.array((N_MEM,), boolean)
        SC = cuda.local.array((N_MEM,), boolean)
        remaining = cuda.local.array((N_MEM,), int32)

        for curRow in range(nr):
            minVal = cuda.local.array((1,), float32)
            sink = -1
            sink = augmenting_path(nc, cost, u, v, path, row4col,
                                        shortestPathCosts, curRow, SR, SC,
                                        remaining, minVal)
            if sink < 0:
                return RECT_LSAP_INFEASIBLE

            u[curRow] += minVal[0]
            for i in range(nr):
                if SR[i] and i != curRow:
                    u[i] += minVal[0] - shortestPathCosts[col4row[i]]
            
            for j in range(nc):
                if SC[j]:
                    v[j] -= minVal[0] - shortestPathCosts[j]
            j = int(sink)
            while True:
                i = path[j]
                row4col[j] = i
                tmp = col4row[i] # std::swap
                col4row[i] = j
                j = tmp
                if i == curRow:
                    break
        
        for i in range(nr):
            a[i] = i
            b[i] = col4row[i]
        return 0
    
    return linear_sum_assignment
