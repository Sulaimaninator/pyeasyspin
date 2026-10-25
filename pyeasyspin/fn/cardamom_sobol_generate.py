"""cardamom_sobol_generate generates a Sobol dataset (Burkardt's i4_sobol, Bratley-Fox direction numbers)."""
import math
import numpy as np


class _I4Sobol:
    """i4_sobol with MATLAB's persistent variables as attributes (state is kept across calls,
    exactly as in EasySpin; the values do not depend on it)."""

    def __init__(self):
        self.initialized = 0
        self.dim_num_save = -1

    def _init(self):
        self.initialized = 1
        self.dim_max = 40
        self.dim_num_save = -1
        self.log_max = 30
        self.seed_save = -1
        v = np.zeros((40, 30), dtype=np.int64)
        v[0:40, 0] = 1
        v[2:40, 1] = [1, 3, 1, 3, 1, 3, 3, 1,
                      3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
                      1, 3, 1, 3, 3, 1, 3, 1, 3, 1,
                      3, 1, 1, 3, 1, 3, 1, 3, 1, 3]
        v[3:40, 2] = [7, 5, 1, 3, 3, 7, 5,
                      5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
                      5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
                      5, 1, 1, 5, 7, 7, 5, 1, 3, 3]
        v[5:40, 3] = [1, 7, 9, 13, 11,
                      1, 3, 7, 9, 5, 13, 13, 11, 3, 15,
                      5, 3, 15, 7, 9, 13, 9, 1, 11, 7,
                      5, 15, 1, 15, 11, 5, 3, 1, 7, 9]
        v[7:40, 4] = [9, 3, 27,
                      15, 29, 21, 23, 19, 11, 25, 7, 13, 17,
                      1, 25, 29, 3, 31, 11, 5, 23, 27, 19,
                      21, 5, 1, 17, 13, 7, 15, 9, 31, 9]
        v[13:40, 5] = [37, 33, 7, 5, 11, 39, 63,
                       27, 17, 15, 23, 29, 3, 21, 13, 31, 25,
                       9, 49, 33, 19, 29, 11, 19, 27, 15, 25]
        v[19:40, 6] = [13,
                       33, 115, 41, 79, 17, 29, 119, 75, 73, 105,
                       7, 59, 65, 21, 3, 113, 61, 89, 45, 107]
        v[37:40, 7] = [7, 23, 39]
        self.poly = [1, 3, 7, 11, 13, 19, 25, 37, 59, 47,
                     61, 55, 41, 67, 97, 91, 109, 103, 115, 131,
                     193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
                     213, 191, 253, 203, 211, 239, 247, 285, 369, 299]
        self.atmost = 2 ** self.log_max - 1
        self.maxcol = _bit_hi1(self.atmost)
        v[0, 0:self.maxcol] = 1
        self.v = v
        self.lastq = np.zeros(40, dtype=np.int64)

    def __call__(self, DimNum, seed):
        DimNum = int(DimNum)
        if not self.initialized or DimNum != self.dim_num_save:
            self._init()
        v = self.v
        if DimNum != self.dim_num_save:
            if DimNum < 1 or self.dim_max < DimNum:
                raise ValueError(f'I4_SOBOL - Fatal error!\n  The spatial dimension DIM_NUM should satisfy:\n'
                                 f'    1 <= DIM_NUM <= {self.dim_max}\n  But this input value is DIM_NUM = {DimNum}')
            self.dim_num_save = DimNum
            includ = {}
            for i in range(2, DimNum + 1):
                j = self.poly[i - 1]
                m = 0
                while True:
                    j = j // 2
                    if j <= 0:
                        break
                    m += 1
                j = self.poly[i - 1]
                for k in range(m, 0, -1):
                    j2 = j // 2
                    includ[k] = (j != 2 * j2)
                    j = j2
                for j in range(m + 1, self.maxcol + 1):
                    newv = int(v[i - 1, j - m - 1])
                    l = 1
                    for k in range(1, m + 1):
                        l = 2 * l
                        if includ[k]:
                            newv = newv ^ (l * int(v[i - 1, j - k - 1]))
                    v[i - 1, j - 1] = newv
            l = 1
            for j in range(self.maxcol - 1, 0, -1):
                l = 2 * l
                v[0:DimNum, j - 1] = v[0:DimNum, j - 1] * l
            self.recipd = 1.0 / (2 * l)
            self.lastq[0:DimNum] = 0
        seed = math.floor(seed)
        if seed < 0:
            seed = 0
        lastq = self.lastq
        if seed == 0:
            l = 1
            lastq[0:DimNum] = 0
        elif seed == self.seed_save + 1:
            l = _bit_lo0(seed)
        elif seed <= self.seed_save:
            self.seed_save = 0
            l = 1
            lastq[0:DimNum] = 0
            for seed_temp in range(int(self.seed_save), int(seed)):
                l = _bit_lo0(seed_temp)
                for i in range(DimNum):
                    lastq[i] = lastq[i] ^ v[i, l - 1]
            l = _bit_lo0(seed)
        elif self.seed_save + 1 < seed:
            for seed_temp in range(int(self.seed_save) + 1, int(seed)):
                l = _bit_lo0(seed_temp)
                for i in range(DimNum):
                    lastq[i] = lastq[i] ^ v[i, l - 1]
            l = _bit_lo0(seed)
        if self.maxcol < l:
            raise ValueError(f'I4_SOBOL - Fatal error!\n  Too many calls!\n  MAXCOL = {self.maxcol}\n  L =      {l}')
        quasi = np.zeros(DimNum)
        for i in range(DimNum):
            quasi[i] = lastq[i] * self.recipd
            lastq[i] = lastq[i] ^ v[i, l - 1]
        self.seed_save = seed
        return quasi, seed + 1


def _bit_hi1(n):
    i = math.floor(n)
    bit = 0
    while i > 0:
        bit += 1
        i = i // 2
    return bit


def _bit_lo0(n):
    bit = 0
    i = math.floor(n)
    while True:
        bit += 1
        i2 = i // 2
        if i == 2 * i2:
            break
        i = i2
    return bit


_i4_sobol = _I4Sobol()


def cardamom_sobol_generate(m=None, n=None, skip=None):
    """cardamom_sobol_generate(m, n, skip) -> r (m x n): the Sobol points with seeds skip..skip+n-1 in
    dimension m (m <= 40), as Burkardt's sobol_generate/i4_sobol, which EasySpin uses."""
    m, n = int(m), int(n)
    if n < 1:
        raise ValueError("Output argument \"r\" (and possibly others) not assigned a value in the execution with \"sobol_generate\" function.")
    r = np.zeros((m, n))
    for j in range(1, n + 1):
        seed = skip + j - 1
        r[:, j - 1], seed = _i4_sobol(m, seed)
    return r
