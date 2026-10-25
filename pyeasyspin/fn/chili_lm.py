"""chili_lm mex: Liouvillian (diffusion + Hamiltonian superoperator) of an S=1/2 system with 0, 1 or
2 nuclei in the K-symmetrized Freed basis, returned as COO triplets. Line-by-line numba port of
easyspin/private/chili_lm.c, chili_lm0/1/2.inc and jjj.h: same loop nesting and element order,
same truncated constants, same floating-point evaluation order, 1-based output indices."""
import math
import numpy as np
from numba import njit
from .._registry import nout

# ln(N!) table of jjj.h (facln), N = 0..200, transcribed from the C source
_FACLN = np.array([
    0, 0, 0.69314718055994530942,
    1.7917594692280550008, 3.1780538303479456196, 4.7874917427820459942,
    6.5792512120101009951, 8.5251613610654143002, 10.604602902745250228,
    12.801827480081469611, 15.104412573075515295, 17.502307845873885839,
    19.987214495661886150, 22.552163853123422886, 25.191221182738681500,
    27.899271383840891566, 30.671860106080672804, 33.505073450136888884,
    36.395445208033053576, 39.339884187199494036, 42.335616460753485030,
    45.380138898476908026, 48.471181351835223880, 51.606675567764373570,
    54.784729398112319190, 58.003605222980519939, 61.261701761002001985,
    64.557538627006331059, 67.889743137181534983, 71.257038967168009010,
    74.658236348830164385, 78.092223553315310631, 81.557959456115037179,
    85.054467017581517414, 88.580827542197678804, 92.136175603687092483,
    95.719694542143202485, 99.330612454787426929, 102.96819861451381270,
    106.63176026064345913, 110.32063971475739543, 114.03421178146170323,
    117.77188139974507154, 121.53308151543863396, 125.31727114935689513,
    129.12393363912721488, 132.95257503561630988, 136.80272263732636847,
    140.67392364823425940, 144.56574394634488601, 148.47776695177303207,
    152.40959258449735784, 156.36083630307878519, 160.33112821663090703,
    164.32011226319518141, 168.32744544842765233, 172.35279713916280156,
    176.39584840699735172, 180.45629141754377105, 184.53382886144949050,
    188.62817342367159119, 192.73904728784490244, 196.86618167288999399,
    201.00931639928152668, 205.16819948264119854, 209.34258675253683565,
    213.53224149456326119, 217.73693411395422725, 221.95644181913033395,
    226.19054832372759333, 230.43904356577695232, 234.70172344281826774,
    238.97838956183432305, 243.26884900298271418, 247.57291409618688394,
    251.89040220972319438, 256.22113555000952546, 260.56494097186320931,
    264.92164979855280104, 269.29109765101982254, 273.67312428569370415,
    278.06757344036614291, 282.47429268763039603, 286.89313329542699395,
    291.32395009427030757, 295.76660135076062402, 300.22094864701413175,
    304.68685676566871547, 309.16419358014692194, 313.65282994987906178,
    318.15263962020932685, 322.66349912672617689, 327.18528770377521720,
    331.71788719692847314, 336.26118197919847703, 340.81505887079901787,
    345.37940706226685411, 349.95411804077023693, 354.53908551944080885,
    359.13420536957539878, 363.73937555556349014, 368.35449607240474959,
    372.97946888568902068, 377.61419787391865645, 382.25858877306002911,
    386.91254912321755248, 391.57598821732961963, 396.24881705179152580,
    400.93094827891574549, 405.62229616114488919, 410.32277652693730542,
    415.03230672824963956, 419.75080559954473410, 424.47819341825707467,
    429.21439186665157013, 433.95932399501482019, 438.71291418612118484,
    443.47508812091894096, 448.24577274538460572, 453.02489623849613510,
    457.81238798127818110, 462.60817852687492219, 467.41219957160817874,
    472.22438392698059624, 477.04466549258563310, 481.87297922988793423,
    486.70926113683941223, 491.55344822329800350, 496.40547848721762066,
    501.26529089157929278, 506.13282534203487520, 511.00802266523602674,
    515.89082458782239760, 520.78117371604415136, 525.67901351599506273,
    530.58428829443349218, 535.49694318016954419, 540.41692410599766910,
    545.34417779115487380, 550.27865172428556555, 555.22029414689486985,
    560.16905403727303813, 565.12488109487429886, 570.08772572513420614,
    575.05753902471020676, 580.03427276713078116, 585.01787938883911760,
    590.00831197561785390, 595.00552424938196897, 600.00947055532742811,
    605.02010584942368386, 610.03738568623860819, 615.06126620708488458,
    620.09170412847732004, 625.12865673089094920, 630.17208184781019582,
    635.22193785505973286, 640.27818366040804092, 645.34077869343500772,
    650.40968289565523925, 655.48485671088906617, 660.56626107587352917,
    665.65385741110591324, 670.74760761191267558, 675.84747403973687400,
    680.95341951363745461, 686.06540730199399784, 691.18340111441075295,
    696.30736509381401187, 701.43726380873708535, 706.57306224578734711,
    711.71472580229000695, 716.86222027910346000, 722.01551187360123894,
    727.17456717281576797, 732.33935314673928203, 737.50983714177743381,
    742.68598687435126295, 747.86777042464334810, 753.05515623048410309,
    758.24811308137431347, 763.44661011264013922, 768.65061679971693457,
    773.86010295255835551, 779.07503871016734113, 784.29539453524566594,
    789.52114120895886719, 794.75224982581345382, 799.98869178864340302,
    805.23043880370304540, 810.47746287586353154, 815.72973630391016142,
    820.98723167593794297, 826.24992186484282852, 831.51778002390615665,
    836.79077958246990345, 842.06889424170042068, 847.35209797043840919,
    852.64036500113294442, 857.93366982585743682, 863.23198719240547350,
])
_LNC = np.array([76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
                 0.1208650973866179e-2, -0.5395239384953e-5])

# constants exactly as in chili_lm.c (truncated decimals)
_SQRT12 = 0.70710678118655
_SQRT13 = 0.57735026918963
_SQRT23 = 0.81649658092773


@njit(cache=True, error_model='numpy')
def _facln(N):
    if N <= 200:
        return _FACLN[N]
    x = N + 1.0
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    y = N + 1.0
    Series = 1 + 1.90015e-10
    for j in range(6):
        y += 1.0
        Series += _LNC[j] / y
    return -tmp + math.log(2.5066282746310005024 * Series / x)


@njit(cache=True)
def _parity(k):
    return -1 if (k % 2) != 0 else 1


@njit(cache=True, error_model='numpy')
def _jjj(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol for integer arguments (jjj.h)."""
    prty = _parity(j1 + j2 + j3)
    phase = 1
    if j1 < j3:
        j1, j3 = j3, j1
        m1, m3 = m3, m1
        phase *= prty
    if j1 < j2:
        j1, j2 = j2, j1
        m1, m2 = m2, m1
        phase *= prty
    if j2 < j3:
        j2, j3 = j3, j2
        m2, m3 = m3, m2
        phase *= prty
    if m3 < 0:
        m1 = -m1
        m2 = -m2
        m3 = -m3
        phase *= prty
    if j1 - j2 > j3:
        return 0.0
    w3j = 0.0
    if j3 <= 2:
        jd = j1 - j2
        J = j2
        M = m1
        x = float(2 * J - 1 + jd)
        y = float(J - M)
        z = float(J + M)
        phase *= _parity(J - M)
        if j3 == 0:
            w3j = 1 / math.sqrt(2.0 * J + 1)
        elif j3 == 1:
            if jd == 0:
                if m3 == 0:
                    w3j = M / math.sqrt(J * (J + 1) * (2.0 * J + 1))
                else:
                    w3j = math.sqrt(2.0 * y * (z + 1) / ((x + 3) * (x + 2) * (x + 1)))
            else:
                tmp = (x + 1) * (x + 2) * (x + 3)
                if m3 == 0:
                    w3j = -math.sqrt(2 * (y + 1) * (z + 1) / tmp)
                else:
                    w3j = -math.sqrt(y * (y + 1) / tmp)
        else:
            tmp = x * (x + 1) * (x + 2) * (x + 3) * (x + 4)
            if m3 == 0:
                if jd == 0:
                    w3j = 2 * (3 * M * M - J * (J + 1)) / math.sqrt(tmp)
                elif jd == 1:
                    w3j = -2 * M * math.sqrt(6 * (z + 1) * (y + 1) / tmp)
                else:
                    w3j = math.sqrt(6 * (z + 2) * (z + 1) * (y + 2) * (y + 1) / tmp)
            elif m3 == 1:
                if jd == 0:
                    w3j = (2 * M + 1) * math.sqrt(6 * (z + 1) * y / tmp)
                elif jd == 1:
                    w3j = -(2 * J + 4 * M + 4) * math.sqrt(y * (y + 1) / tmp)
                else:
                    w3j = 2 * math.sqrt((z + 2) * y * (y + 1) * (y + 2) / tmp)
            elif m3 == 2:
                if jd == 0:
                    w3j = math.sqrt(6 * (y - 1) * y * (z + 1) * (z + 2) / tmp)
                elif jd == 1:
                    w3j = -2 * math.sqrt((y - 1) * y * (y + 1) * (z + 2) / tmp)
                else:
                    w3j = math.sqrt((y - 1) * y * (y + 1) * (y + 2) / tmp)
    else:
        delta_ln = _facln(j1 + j2 - j3) + _facln(j1 - j2 + j3) + _facln(-j1 + j2 + j3) - _facln(j1 + j2 + j3 + 1)
        fact_ln = _facln(j1 + m1) + _facln(j1 - m1) + _facln(j2 + m2) + _facln(j2 - m2) + _facln(j3 + m3) + _facln(j3 - m3)
        t1 = j2 - m1 - j3
        t2 = j1 + m2 - j3
        t3 = j1 + j2 - j3
        t4 = j1 - m1
        t5 = j2 + m2
        tmin = t1 if t1 > t2 else t2
        tmin = tmin if tmin > 0 else 0
        tmax = t3 if t3 < t4 else t4
        tmax = tmax if tmax < t5 else t5
        Sum = 0.0
        for t in range(tmin, tmax + 1):
            denom_ln = _facln(t) + _facln(t - t1) + _facln(t - t2) + _facln(t3 - t) + _facln(t4 - t) + _facln(t5 - t)
            Sum += _parity(t) * math.exp(delta_ln / 2 + fact_ln / 2 - denom_ln)
        w3j = _parity(j1 - j2 - m3) * Sum
    return phase * w3j


@njit(cache=True)
def _grow(a, n):
    b = np.zeros(n, a.dtype)
    b[:a.size] = a
    return b


@njit(cache=True, error_model='numpy')
def _potdiff(L1, L2, K1, K2, M1, Ld, Ks, Kd, jKd, jK2, parityLK2, NormFactor, Lband, xlk, maxL):
    """Potential-dependent diffusion term, Eq. (A40) (identical block in chili_lm0/1/2.inc)."""
    PotDiff = 0.0
    if abs(Ld) <= Lband and _parity(Ks) == 1 and jKd == 0:
        for L in range(0, Lband + 1, 2):
            Term1 = 0.0
            if abs(Kd) <= L:
                X = xlk[(Kd + L) * (maxL + 1) + L]
                if X != 0:
                    Term1 = X * _jjj(L1, L, L2, K1, -Kd, -K2)
            Term2 = 0.0
            if Ks <= L:
                X = xlk[(Ks + L) * (maxL + 1) + L]
                if X != 0:
                    Term2 = parityLK2 * jK2 * X * _jjj(L1, L, L2, K1, -Ks, K2)
            if Term1 != 0 or Term2 != 0:
                PotDiff += (Term1 + Term2) * _jjj(L1, L, L2, M1, 0, -M1)
        PotDiff *= NormFactor
    return PotDiff


@njit(cache=True, error_model='numpy')
def _isodiff(L1, K1, Rxx, Ryy, Rzz):
    IsoDiffKdiag = (Rxx + Ryy) / 2 * (L1 * (L1 + 1)) + K1 * K1 * (Rzz - (Rxx + Ryy) / 2)
    if Rxx != Ryy:
        KK = K1 - 2
        IsoDiffKm2 = (Rxx - Ryy) / 4 * math.sqrt((L1 - KK - 1) * (L1 - KK) * (L1 + KK + 1) * (L1 + KK + 2))
        KK = K1 + 2
        IsoDiffKp2 = (Rxx - Ryy) / 4 * math.sqrt((L1 + KK - 1) * (L1 + KK) * (L1 - KK + 1) * (L1 - KK + 2))
    else:
        IsoDiffKp2 = 0.0
        IsoDiffKm2 = 0.0
    return IsoDiffKdiag, IsoDiffKm2, IsoDiffKp2


@njit(cache=True, error_model='numpy')
def _rterm(L1, L2, K1, K2, Kd, Ks, jK1, jK2, parityLK2, Re, Im, hasIm):
    """R(mu;l=2) of Eq. (A42)/(A44) for one interaction (5 ISTO components Re/Im)."""
    g1 = 0.0
    if abs(Kd) <= 2:
        coeff = _jjj(L1, 2, L2, K1, -Kd, -K2)
        if jK1 == jK2:
            g1 = coeff * Re[Kd + 2]
        elif hasIm:
            g1 = coeff * Im[Kd + 2] * jK1
    g2 = 0.0
    if abs(Ks) <= 2:
        coeff = _jjj(L1, 2, L2, K1, -Ks, K2)
        if jK1 == jK2:
            g2 = coeff * Re[Ks + 2]
        elif hasIm:
            g2 = coeff * Im[Ks + 2] * jK1
    return g1 + jK2 * parityLK2 * g2


@njit(cache=True, error_model='numpy')
def _makematrix0(Lemax, Lomax, Kmax, Mmax, jKmin, pSmin, deltaK, MeirovitchSymm,
                 EZI0, ReEZI2, ImEZI2, hasImEZ, DirTilt, d2psi, Rxx, Ryy, Rzz, xlk, maxL, cap):
    ridx = np.zeros(cap)
    cidx = np.zeros(cap)
    MRe = np.zeros(cap)
    MIm = np.zeros(cap)
    Lband = maxL if maxL >= 4 else 2
    Potential = maxL >= 0
    iRow = 0
    iElement = 0
    for L1 in range(0, Lemax + 1):
        if (L1 % 2) and (L1 > Lomax):
            continue
        for jK1 in range(jKmin, 2, 2):
            K1max = min(Kmax, L1)
            for K1 in range(0, K1max + 1, deltaK):
                if K1 == 0 and _parity(L1) != jK1:
                    continue
                IsoDiffKdiag, IsoDiffKm2, IsoDiffKp2 = _isodiff(L1, K1, Rxx, Ryy, Rzz)
                M1max = min(Mmax, L1)
                for M1 in range(-M1max, M1max + 1):
                    for pS1 in range(pSmin, 2):
                        qS1max = 1 - abs(pS1)
                        for qS1 in range(-qS1max, qS1max + 1, 2):
                            if MeirovitchSymm and DirTilt == 0 and (0 + pS1 - 1) != M1:
                                continue
                            iCol = iRow
                            diagRC = True
                            L2max = min(Lemax, L1 + Lband)
                            for L2 in range(L1, L2max + 1):
                                if (L2 % 2) and (L2 > Lomax):
                                    continue
                                Ld = L1 - L2
                                Ld2 = abs(Ld) <= 2
                                N_L = math.sqrt(float((2.0 * L1 + 1.0) * (2.0 * L2 + 1.0)))
                                for jK2 in range(jK1 if diagRC else jKmin, 2, 2):
                                    jKd = jK1 - jK2
                                    K2max = min(Kmax, L2)
                                    for K2 in range(K1 if diagRC else 0, K2max + 1, deltaK):
                                        if K2 == 0 and _parity(L2) != jK2:
                                            continue
                                        diagLK = (L1 == L2) and (K1 == K2)
                                        Kd = K1 - K2
                                        Ks = K1 + K2
                                        parityLK2 = _parity(L2 + K2)
                                        R_EZI2 = 0.0
                                        if Ld2:
                                            R_EZI2 = _rterm(L1, L2, K1, K2, Kd, Ks, jK1, jK2, parityLK2, ReEZI2, ImEZI2, hasImEZ)
                                        N_K = 1.0
                                        if K1 == 0:
                                            N_K /= math.sqrt(2.0)
                                        if K2 == 0:
                                            N_K /= math.sqrt(2.0)
                                        NormFactor = N_L * N_K * _parity(M1 + K1)
                                        PotDiff = 0.0
                                        if Potential:
                                            PotDiff = _potdiff(L1, L2, K1, K2, M1, Ld, Ks, Kd, jKd, jK2, parityLK2,
                                                               NormFactor, Lband, xlk, maxL)
                                        M2max = min(Mmax, L2)
                                        for M2 in range(M1 if diagRC else -M2max, M2max + 1):
                                            Md = M1 - M2
                                            diagLKM = diagLK and (jKd == 0) and (Md == 0)
                                            Liou3j = _jjj(L1, 2, L2, M1, -Md, -M2) if (Ld2 and abs(Md) <= 2) else 0.0
                                            for pS2 in range(pS1 if diagRC else pSmin, 2):
                                                pSd = pS1 - pS2
                                                qS2max = 1 - abs(pS2)
                                                for qS2 in range(qS1 if diagRC else -qS2max, qS2max + 1, 2):
                                                    diagS = (pS1 == pS2) and (qS1 == qS2)
                                                    qSd = qS1 - qS2
                                                    if MeirovitchSymm and DirTilt == 0 and (0 + pS2 - 1) != M2:
                                                        continue
                                                    includeRank0 = diagLKM and (pSd == 0)
                                                    LiouvilleElement = 0.0
                                                    if Ld2 and abs(pSd) <= 1 and abs(Md) <= 2 and abs(pSd) == abs(qSd):
                                                        d2jjj = d2psi[(pSd + 2) + (Md + 2) * 5] * Liou3j
                                                        if pSd == 0:
                                                            C2 = _SQRT23
                                                            S_g = C2 * pS1
                                                        else:
                                                            C2 = _SQRT12
                                                            S_g = -C2 * qSd / math.sqrt(2.0)
                                                        LiouvilleElement += NormFactor * d2jjj * R_EZI2 * S_g
                                                        if includeRank0:
                                                            C0 = -_SQRT13
                                                            LiouvilleElement += C0 * EZI0 * pS1
                                                    GammaElement = 0.0
                                                    if diagS and Ld == 0 and Md == 0 and jKd == 0:
                                                        if Kd == 0:
                                                            GammaElement += IsoDiffKdiag
                                                        elif Kd == 2:
                                                            GammaElement += IsoDiffKm2 / N_K
                                                        elif Kd == -2:
                                                            GammaElement += IsoDiffKp2 / N_K
                                                    if Potential:
                                                        if diagS and Md == 0 and jKd == 0:
                                                            GammaElement += PotDiff
                                                    if GammaElement != 0 or LiouvilleElement != 0:
                                                        if iElement + 2 > ridx.size:
                                                            n = 2 * ridx.size + 2
                                                            ridx = _grow(ridx, n)
                                                            cidx = _grow(cidx, n)
                                                            MRe = _grow(MRe, n)
                                                            MIm = _grow(MIm, n)
                                                        MRe[iElement] = GammaElement
                                                        MIm[iElement] = -LiouvilleElement
                                                        ridx[iElement] = iRow
                                                        cidx[iElement] = iCol
                                                        iElement += 1
                                                        if not diagRC:
                                                            MRe[iElement] = GammaElement
                                                            MIm[iElement] = -LiouvilleElement
                                                            ridx[iElement] = iCol
                                                            cidx[iElement] = iRow
                                                            iElement += 1
                                                    iCol += 1
                                                    diagRC = False
                            iRow += 1
    return ridx, cidx, MRe, MIm, iRow, iElement


@njit(cache=True, error_model='numpy')
def _hf_coeffs(pSd, pId, qSd, qId, pS1, qS1, pI1, qI1, I, pd):
    """Clebsch-Gordan coefficients and S_A of Eq. (B7) for one nucleus (chili_lm1/2.inc)."""
    if pId == 0:
        if pSd == 0:
            S_A = (pS1 * qI1 + pI1 * qS1) / 2.0
            C0 = -_SQRT13
            C2 = _SQRT23
        else:
            S_A = -(pI1 * pSd + qI1 * qSd) / math.sqrt(8.0)
            C0 = 0.0
            C2 = _SQRT12
    else:
        t = qI1 * qId + pI1 * pId
        KI = math.sqrt(I * (I + 1.0) - t * (t - 2.0) / 4.0)
        if pSd == 0:
            S_A = -(pS1 * pId + qS1 * qId) * KI / math.sqrt(8.0)
            C0 = 0.0
            C2 = _SQRT12
        else:
            S_A = pSd * qId * KI / 2.0
            C0 = _SQRT13
            if pd == 0:
                C2 = math.sqrt(1.0 / 6.0)
            else:
                C2 = 1.0
    return C0, C2, S_A


@njit(cache=True, error_model='numpy')
def _makematrix1(Lemax, Lomax, Kmax, Mmax, jKmin, pSmin, deltaK, MeirovitchSymm, pImax,
                 EZI0, ReEZI2, ImEZI2, hasImEZ, I, NZI0, HFI0, ReHFI2, ImHFI2, hasImHF,
                 DirTilt, d2psi, Rxx, Ryy, Rzz, ExchangeFreq, xlk, maxL, cap):
    ridx = np.zeros(cap)
    cidx = np.zeros(cap)
    MRe = np.zeros(cap)
    MIm = np.zeros(cap)
    Lband = maxL if maxL >= 4 else 2
    Potential = maxL >= 0
    ExchangePresent = ExchangeFreq != 0
    twoI = int(2 * I)
    iRow = 0
    iElement = 0
    for L1 in range(0, Lemax + 1):
        if (L1 % 2) and (L1 > Lomax):
            continue
        for jK1 in range(jKmin, 2, 2):
            K1max = min(Kmax, L1)
            for K1 in range(0, K1max + 1, deltaK):
                if K1 == 0 and _parity(L1) != jK1:
                    continue
                IsoDiffKdiag, IsoDiffKm2, IsoDiffKp2 = _isodiff(L1, K1, Rxx, Ryy, Rzz)
                M1max = min(Mmax, L1)
                for M1 in range(-M1max, M1max + 1):
                    for pS1 in range(pSmin, 2):
                        qS1max = 1 - abs(pS1)
                        for qS1 in range(-qS1max, qS1max + 1, 2):
                            for pI1 in range(-pImax, pImax + 1):
                                if MeirovitchSymm and DirTilt == 0 and (pI1 + pS1 - M1) != 1:
                                    continue
                                qI1max = twoI - abs(pI1)
                                for qI1 in range(-qI1max, qI1max + 1, 2):
                                    iCol = iRow
                                    diagRC = True
                                    L2max = min(Lemax, L1 + Lband)
                                    for L2 in range(L1, L2max + 1):
                                        if (L2 % 2) and (L2 > Lomax):
                                            continue
                                        Ld = L1 - L2
                                        Ld2 = abs(Ld) <= 2
                                        N_L = math.sqrt(float((2.0 * L1 + 1.0) * (2.0 * L2 + 1.0)))
                                        for jK2 in range(jK1 if diagRC else jKmin, 2, 2):
                                            jKd = jK1 - jK2
                                            K2max = min(Kmax, L2)
                                            for K2 in range(K1 if diagRC else 0, K2max + 1, deltaK):
                                                if K2 == 0 and _parity(L2) != jK2:
                                                    continue
                                                diagLK = (L1 == L2) and (K1 == K2)
                                                Kd = K1 - K2
                                                Ks = K1 + K2
                                                parityLK2 = _parity(L2 + K2)
                                                R_EZI2 = 0.0
                                                R_HFI2 = 0.0
                                                if Ld2:
                                                    R_EZI2 = _rterm(L1, L2, K1, K2, Kd, Ks, jK1, jK2, parityLK2, ReEZI2, ImEZI2, hasImEZ)
                                                    R_HFI2 = _rterm(L1, L2, K1, K2, Kd, Ks, jK1, jK2, parityLK2, ReHFI2, ImHFI2, hasImHF)
                                                N_K = 1.0
                                                if K1 == 0:
                                                    N_K /= math.sqrt(2.0)
                                                if K2 == 0:
                                                    N_K /= math.sqrt(2.0)
                                                NormFactor = N_L * N_K * _parity(M1 + K1)
                                                PotDiff = 0.0
                                                if Potential:
                                                    PotDiff = _potdiff(L1, L2, K1, K2, M1, Ld, Ks, Kd, jKd, jK2, parityLK2,
                                                                       NormFactor, Lband, xlk, maxL)
                                                M2max = min(Mmax, L2)
                                                for M2 in range(M1 if diagRC else -M2max, M2max + 1):
                                                    Md = M1 - M2
                                                    diagLKM = diagLK and (jKd == 0) and (Md == 0)
                                                    Liou3j = _jjj(L1, 2, L2, M1, -Md, -M2) if (Ld2 and abs(Md) <= 2) else 0.0
                                                    for pS2 in range(pS1 if diagRC else pSmin, 2):
                                                        pSd = pS1 - pS2
                                                        qS2max = 1 - abs(pS2)
                                                        for qS2 in range(qS1 if diagRC else -qS2max, qS2max + 1, 2):
                                                            qSd = qS1 - qS2
                                                            diagS = (pS1 == pS2) and (qS1 == qS2)
                                                            for pI2 in range(pI1 if diagRC else -pImax, pImax + 1):
                                                                if MeirovitchSymm and DirTilt == 0 and (pI2 + pS2 - M2) != 1:
                                                                    continue
                                                                pId = pI1 - pI2
                                                                qI2max = twoI - abs(pI2)
                                                                for qI2 in range(qI1 if diagRC else -qI2max, qI2max + 1, 2):
                                                                    qId = qI1 - qI2
                                                                    diagI = (pId == 0) and (qId == 0)
                                                                    pd = pSd + pId
                                                                    includeRank0 = diagLKM and (pd == 0)
                                                                    LiouvilleElement = 0.0
                                                                    if (Ld2 and abs(pSd) <= 1 and abs(pId) <= 1 and abs(Md) <= 2
                                                                            and abs(pSd) == abs(qSd) and abs(pId) == abs(qId)):
                                                                        d2jjj = d2psi[(pd + 2) + (Md + 2) * 5] * Liou3j
                                                                        if diagI:
                                                                            if pSd == 0:
                                                                                C2 = _SQRT23
                                                                                S_g = float(pS1)
                                                                            else:
                                                                                C2 = _SQRT12
                                                                                S_g = -qSd / math.sqrt(2.0)
                                                                            LiouvilleElement += NormFactor * d2jjj * R_EZI2 * (C2 * S_g)
                                                                            if includeRank0:
                                                                                C0 = -_SQRT13
                                                                                LiouvilleElement += EZI0 * (C0 * pS1)
                                                                        if I > 0 and pSd * pId == qSd * qId:
                                                                            C0, C2, S_A = _hf_coeffs(pSd, pId, qSd, qId, pS1, qS1, pI1, qI1, I, pd)
                                                                            LiouvilleElement += NormFactor * d2jjj * R_HFI2 * C2 * S_A
                                                                            if includeRank0:
                                                                                LiouvilleElement += HFI0 * C0 * S_A
                                                                        if diagS and diagI and includeRank0:
                                                                            C0 = -_SQRT13
                                                                            LiouvilleElement += NZI0 * C0 * pI1
                                                                    GammaElement = 0.0
                                                                    if diagS and diagI:
                                                                        if Ld == 0 and Md == 0 and jKd == 0:
                                                                            if Kd == 0:
                                                                                GammaElement += IsoDiffKdiag
                                                                            elif Kd == 2:
                                                                                GammaElement += IsoDiffKm2 / N_K
                                                                            elif Kd == -2:
                                                                                GammaElement += IsoDiffKp2 / N_K
                                                                        if Potential:
                                                                            if Md == 0 and jKd == 0:
                                                                                GammaElement += PotDiff
                                                                    if ExchangePresent:
                                                                        if pSd == 0 and pId == 0 and diagLKM:
                                                                            t = 0.0
                                                                            if qId == 0 and qSd == 0:
                                                                                t += 1.0
                                                                            if qId == 0 and pS1 == 0:
                                                                                t -= 0.5
                                                                            if pI1 == 0 and qSd == 0:
                                                                                t -= 1.0 / (2.0 * I + 1)
                                                                            GammaElement += t * ExchangeFreq
                                                                    if GammaElement != 0 or LiouvilleElement != 0:
                                                                        if iElement + 2 > ridx.size:
                                                                            n = 2 * ridx.size + 2
                                                                            ridx = _grow(ridx, n)
                                                                            cidx = _grow(cidx, n)
                                                                            MRe = _grow(MRe, n)
                                                                            MIm = _grow(MIm, n)
                                                                        MRe[iElement] = GammaElement
                                                                        MIm[iElement] = -LiouvilleElement
                                                                        ridx[iElement] = iRow
                                                                        cidx[iElement] = iCol
                                                                        iElement += 1
                                                                        if not diagRC:
                                                                            MRe[iElement] = GammaElement
                                                                            MIm[iElement] = -LiouvilleElement
                                                                            ridx[iElement] = iCol
                                                                            cidx[iElement] = iRow
                                                                            iElement += 1
                                                                    iCol += 1
                                                                    diagRC = False
                                    iRow += 1
    return ridx, cidx, MRe, MIm, iRow, iElement


@njit(cache=True, error_model='numpy')
def _hf_coeffs2(pSd, pId, qSd, qId, pS1, qS1, pI1, qI1, I):
    """As _hf_coeffs, chili_lm2.inc variant (C2 test on pSd+pId)."""
    C0 = 0.0
    C2 = 0.0
    S_A = 0.0
    if pId == 0:
        if pSd == 0:
            S_A = (pS1 * qI1 + pI1 * qS1) / 2.0
            C0 = -_SQRT13
            C2 = _SQRT23
        else:
            S_A = -(pI1 * pSd + qI1 * qSd) / math.sqrt(8.0)
            C0 = 0.0
            C2 = _SQRT12
    else:
        t = qI1 * qId + pI1 * pId
        KI = math.sqrt(I * (I + 1.0) - t * (t - 2.0) / 4.0)
        if pSd == 0:
            S_A = -(pS1 * pId + qS1 * qId) * KI / math.sqrt(8.0)
            C0 = 0.0
            C2 = _SQRT12
        else:
            S_A = pSd * qId * KI / 2.0
            C0 = _SQRT13
            if pSd + pId == 0:
                C2 = math.sqrt(1.0 / 6.0)
            else:
                C2 = 1.0
    return C0, C2, S_A


@njit(cache=True, error_model='numpy')
def _makematrix2(Lemax, Lomax, Kmax, Mmax, jKmin, pSmin, deltaK, MeirovitchSymm, pImax, pIbmax,
                 EZI0, ReEZI2, ImEZI2, hasImEZ, I, NZI0, HFI0, ReHFI2, ImHFI2, hasImHF,
                 Ib, NZI0b, HFI0b, ReHFI2b, ImHFI2b, hasImHFb,
                 DirTilt, d2psi, Rxx, Ryy, Rzz, ExchangeFreq, xlk, maxL, cap):
    ridx = np.zeros(cap)
    cidx = np.zeros(cap)
    MRe = np.zeros(cap)
    MIm = np.zeros(cap)
    Lband = maxL if maxL >= 4 else 2
    Potential = maxL >= 0
    ExchangePresent = ExchangeFreq != 0
    twoI = int(2 * I)
    twoIb = int(2 * Ib)
    iRow = 0
    iElement = 0
    for L1 in range(0, Lemax + 1):
        if (L1 % 2) and (L1 > Lomax):
            continue
        for jK1 in range(jKmin, 2, 2):
            K1max = min(Kmax, L1)
            for K1 in range(0, K1max + 1, deltaK):
                if K1 == 0 and _parity(L1) != jK1:
                    continue
                IsoDiffKdiag, IsoDiffKm2, IsoDiffKp2 = _isodiff(L1, K1, Rxx, Ryy, Rzz)
                M1max = min(Mmax, L1)
                for M1 in range(-M1max, M1max + 1):
                    for pS1 in range(pSmin, 2):
                        qS1max = 1 - abs(pS1)
                        for qS1 in range(-qS1max, qS1max + 1, 2):
                            for pI1 in range(-pImax, pImax + 1):
                                qI1max = twoI - abs(pI1)
                                for qI1 in range(-qI1max, qI1max + 1, 2):
                                    for pI1b in range(-pIbmax, pIbmax + 1):
                                        if MeirovitchSymm and DirTilt == 0 and (pI1 + pI1b + pS1 - M1) != 1:
                                            continue
                                        qI1bmax = twoIb - abs(pI1b)
                                        for qI1b in range(-qI1bmax, qI1bmax + 1, 2):
                                            iCol = iRow
                                            diagRC = True
                                            L2max = min(Lemax, L1 + Lband)
                                            for L2 in range(L1, L2max + 1):
                                                if (L2 % 2) and (L2 > Lomax):
                                                    continue
                                                Ld = L1 - L2
                                                Ld2 = abs(Ld) <= 2
                                                N_L = math.sqrt(float((2.0 * L1 + 1.0) * (2.0 * L2 + 1.0)))
                                                for jK2 in range(jK1 if diagRC else jKmin, 2, 2):
                                                    jKd = jK1 - jK2
                                                    K2max = min(Kmax, L2)
                                                    for K2 in range(K1 if diagRC else 0, K2max + 1, deltaK):
                                                        if K2 == 0 and _parity(L2) != jK2:
                                                            continue
                                                        diagLK = (L1 == L2) and (K1 == K2)
                                                        Kd = K1 - K2
                                                        Ks = K1 + K2
                                                        parityLK2 = _parity(L2 + K2)
                                                        R_EZI2 = 0.0
                                                        R_HFI2 = 0.0
                                                        R_HFI2b = 0.0
                                                        if Ld2:
                                                            R_EZI2 = _rterm(L1, L2, K1, K2, Kd, Ks, jK1, jK2, parityLK2, ReEZI2, ImEZI2, hasImEZ)
                                                            R_HFI2 = _rterm(L1, L2, K1, K2, Kd, Ks, jK1, jK2, parityLK2, ReHFI2, ImHFI2, hasImHF)
                                                            R_HFI2b = _rterm(L1, L2, K1, K2, Kd, Ks, jK1, jK2, parityLK2, ReHFI2b, ImHFI2b, hasImHFb)
                                                        N_K = 1.0
                                                        if K1 == 0:
                                                            N_K /= math.sqrt(2.0)
                                                        if K2 == 0:
                                                            N_K /= math.sqrt(2.0)
                                                        NormFactor = N_L * N_K * _parity(M1 + K1)
                                                        PotDiff = 0.0
                                                        if Potential:
                                                            PotDiff = _potdiff(L1, L2, K1, K2, M1, Ld, Ks, Kd, jKd, jK2, parityLK2,
                                                                               NormFactor, Lband, xlk, maxL)
                                                        M2max = min(Mmax, L2)
                                                        for M2 in range(M1 if diagRC else -M2max, M2max + 1):
                                                            Md = M1 - M2
                                                            diagLKM = diagLK and (jKd == 0) and (Md == 0)
                                                            Liou3j = _jjj(L1, 2, L2, M1, -Md, -M2) if (Ld2 and abs(Md) <= 2) else 0.0
                                                            for pS2 in range(pS1 if diagRC else pSmin, 2):
                                                                pSd = pS1 - pS2
                                                                qS2max = 1 - abs(pS2)
                                                                for qS2 in range(qS1 if diagRC else -qS2max, qS2max + 1, 2):
                                                                    qSd = qS1 - qS2
                                                                    diagS = (pS1 == pS2) and (qS1 == qS2)
                                                                    for pI2 in range(pI1 if diagRC else -pImax, pImax + 1):
                                                                        pId = pI1 - pI2
                                                                        qI2max = twoI - abs(pI2)
                                                                        for qI2 in range(qI1 if diagRC else -qI2max, qI2max + 1, 2):
                                                                            qId = qI1 - qI2
                                                                            for pI2b in range(pI1b if diagRC else -pIbmax, pIbmax + 1):
                                                                                if MeirovitchSymm and DirTilt == 0 and (pI2 + pI2b + pS2 - 1) != M2:
                                                                                    continue
                                                                                pIbd = pI1b - pI2b
                                                                                qI2bmax = twoIb - abs(pI2b)
                                                                                for qI2b in range(qI1b if diagRC else -qI2bmax, qI2bmax + 1, 2):
                                                                                    qIbd = qI1b - qI2b
                                                                                    diagI = (pId == 0) and (qId == 0) and (pIbd == 0) and (qIbd == 0)
                                                                                    pd = pSd + pId + pIbd
                                                                                    LiouvilleElement = 0.0
                                                                                    if (Ld2 and abs(Md) <= 2 and (DirTilt != 0 or pd == Md)
                                                                                            and abs(pSd) <= 1 and abs(pId) <= 1 and abs(pIbd) <= 1
                                                                                            and abs(pSd) == abs(qSd) and abs(pId) == abs(qId)
                                                                                            and abs(pIbd) == abs(qIbd)):
                                                                                        includeRank0 = diagLKM and (pd == 0)
                                                                                        # |pd| = 3 reads outside d2psi in C; no term uses it then
                                                                                        d2jjj = d2psi[(pd + 2) + (Md + 2) * 5] * Liou3j if abs(pd) <= 2 else 0.0
                                                                                        if diagI:
                                                                                            if pSd == 0:
                                                                                                C2 = _SQRT23
                                                                                                S_g = float(pS1)
                                                                                            else:
                                                                                                C2 = _SQRT12
                                                                                                S_g = -qSd / math.sqrt(2.0)
                                                                                            LiouvilleElement += NormFactor * d2jjj * R_EZI2 * (C2 * S_g)
                                                                                            if includeRank0:
                                                                                                C0 = -_SQRT13
                                                                                                LiouvilleElement += EZI0 * (C0 * pS1)
                                                                                        if I > 0 and pSd * pId == qSd * qId and pIbd == 0 and qIbd == 0:
                                                                                            C0, C2, S_A = _hf_coeffs2(pSd, pId, qSd, qId, pS1, qS1, pI1, qI1, I)
                                                                                            LiouvilleElement += NormFactor * d2jjj * R_HFI2 * (C2 * S_A)
                                                                                            if includeRank0:
                                                                                                LiouvilleElement += HFI0 * (C0 * S_A)
                                                                                        if Ib > 0 and pSd * pIbd == qSd * qIbd and pId == 0 and qId == 0:
                                                                                            C0, C2, S_A = _hf_coeffs2(pSd, pIbd, qSd, qIbd, pS1, qS1, pI1b, qI1b, Ib)
                                                                                            LiouvilleElement += NormFactor * d2jjj * R_HFI2b * (C2 * S_A)
                                                                                            if includeRank0:
                                                                                                LiouvilleElement += HFI0b * (C0 * S_A)
                                                                                        if diagS and diagI and includeRank0:
                                                                                            C0 = -_SQRT13
                                                                                            LiouvilleElement += NZI0 * C0 * pI1
                                                                                            LiouvilleElement += NZI0b * C0 * pI1b
                                                                                    GammaElement = 0.0
                                                                                    if diagS and diagI:
                                                                                        if Ld == 0 and Md == 0 and jKd == 0:
                                                                                            if Kd == 0:
                                                                                                GammaElement += IsoDiffKdiag
                                                                                            elif Kd == 2:
                                                                                                GammaElement += IsoDiffKm2 / N_K
                                                                                            elif Kd == -2:
                                                                                                GammaElement += IsoDiffKp2 / N_K
                                                                                        if Potential:
                                                                                            if Md == 0 and jKd == 0:
                                                                                                GammaElement += PotDiff
                                                                                    if ExchangePresent:
                                                                                        if pSd == 0 and pId == 0 and pIbd == 0 and diagLKM:
                                                                                            t = 0.0
                                                                                            if qId == 0 and qIbd == 0 and qSd == 0:
                                                                                                t += 1.0
                                                                                            if qId == 0 and qIbd == 0 and pS1 == 0:
                                                                                                t -= 0.5
                                                                                            if pI1 == 0 and pI1b == 0 and qSd == 0:
                                                                                                t -= 1.0 / (2.0 * I + 1) / (2.0 * Ib + 1)
                                                                                            GammaElement += t * ExchangeFreq
                                                                                    if GammaElement != 0 or LiouvilleElement != 0:
                                                                                        if iElement + 2 > ridx.size:
                                                                                            n = 2 * ridx.size + 2
                                                                                            ridx = _grow(ridx, n)
                                                                                            cidx = _grow(cidx, n)
                                                                                            MRe = _grow(MRe, n)
                                                                                            MIm = _grow(MIm, n)
                                                                                        MRe[iElement] = GammaElement
                                                                                        MIm[iElement] = -LiouvilleElement
                                                                                        ridx[iElement] = iRow
                                                                                        cidx[iElement] = iCol
                                                                                        iElement += 1
                                                                                        if not diagRC:
                                                                                            MRe[iElement] = GammaElement
                                                                                            MIm[iElement] = -LiouvilleElement
                                                                                            ridx[iElement] = iCol
                                                                                            cidx[iElement] = iRow
                                                                                            iElement += 1
                                                                                    iCol += 1
                                                                                    diagRC = False
                                            iRow += 1
    return ridx, cidx, MRe, MIm, iRow, iElement


# ------------------------------------------------------------------ mex interface helpers
def _flat(x):
    a = np.asarray(x)
    return a.ravel(order='F') if a.ndim else a.reshape(1)


def _scalar(x):
    """mxGetScalar: first element of the real part (0 for an empty array)."""
    a = _flat(x)
    return float(np.real(a[0])) if a.size else 0.0


def _pr(x):
    """mxGetPr: real part as a contiguous float vector."""
    return np.ascontiguousarray(np.real(_flat(x)), dtype=float)


def _pi(x):
    """mxGetPi: (imaginary part, True) for complex storage, else (zeros, False) (C: NULL)."""
    a = _flat(x)
    if np.iscomplexobj(a):
        return np.ascontiguousarray(np.imag(a), dtype=float), True
    return np.zeros(max(a.size, 5)), False


@nout(4)
def chili_lm(*varargin):
    """chili_lm(Sys, basisopts, Dynamics, allocation) -> (r, c, Vals, nRows): 1-based row/column
    indices (column vectors), complex values Gamma - 1i*L, and the basis size. One argument
    (the mex's jjj debug entry) computes nothing and returns None."""
    nrhs = len(varargin)
    if nrhs == 1:
        return None
    if nrhs != 4:
        raise ValueError('4 input arguments expected.')
    S, basisopts, D, alloc = varargin
    nNuclei = int(_scalar(S['nNuclei']))
    if nNuclei > 2:
        raise ValueError('chili_lm0123 works only for 0, 1 or 2 nuclear spins.')
    EZI0 = _scalar(S['EZ0'])
    DirTilt = _scalar(S['DirTilt'])
    d2psi = _pr(S['d2psi'])
    ReEZI2 = _pr(S['EZ2'])
    ImEZI2, hasImEZ = _pi(S['EZ2'])
    bo = _pr(basisopts)
    Lemax, Lomax, Kmax, Mmax = int(bo[0]), int(bo[1]), int(bo[2]), int(bo[3])
    jKmin, pSmin, deltaK, Meir = int(bo[4]), int(bo[5]), int(bo[6]), bool(int(bo[7]))
    pImax = int(bo[8]) if nNuclei >= 1 else 0
    pIbmax = int(bo[9]) if nNuclei >= 2 else 0
    Exchange = _scalar(D['Exchange'])
    xlk = _pr(D['xlk'])
    if xlk.size == 0:
        xlk = np.zeros(1)
    maxL = int(_scalar(D['maxL']))
    R = _pr(D['Diff'])
    Rxx, Ryy, Rzz = float(R[0]), float(R[1]), float(R[2])
    blockSize = int(_scalar(alloc))
    cap = max(2, min(blockSize, 1 << 16))
    if nNuclei == 0:
        res = _makematrix0(Lemax, Lomax, Kmax, Mmax, jKmin, pSmin, deltaK, Meir, EZI0, ReEZI2, ImEZI2, hasImEZ,
                           DirTilt, d2psi, Rxx, Ryy, Rzz, xlk, maxL, cap)
    else:
        I = _scalar(S['I'])
        NZI0 = _scalar(S['NZ0'])
        HFI0 = _scalar(S['HF0'])
        ReHF = _pr(S['HF2'])
        ImHF, hasImHF = _pi(S['HF2'])
        if nNuclei == 1:
            res = _makematrix1(Lemax, Lomax, Kmax, Mmax, jKmin, pSmin, deltaK, Meir, pImax,
                               EZI0, ReEZI2, ImEZI2, hasImEZ, I, NZI0, HFI0, ReHF, ImHF, hasImHF,
                               DirTilt, d2psi, Rxx, Ryy, Rzz, Exchange, xlk, maxL, cap)
        else:
            Ib = _scalar(S['Ib'])
            NZI0b = _scalar(S['NZ0b'])
            HFI0b = _scalar(S['HF0b'])
            ReHFb = _pr(S['HF2b'])
            ImHFb, hasImHFb = _pi(S['HF2b'])
            res = _makematrix2(Lemax, Lomax, Kmax, Mmax, jKmin, pSmin, deltaK, Meir, pImax, pIbmax,
                               EZI0, ReEZI2, ImEZI2, hasImEZ, I, NZI0, HFI0, ReHF, ImHF, hasImHF,
                               Ib, NZI0b, HFI0b, ReHFb, ImHFb, hasImHFb,
                               DirTilt, d2psi, Rxx, Ryy, Rzz, Exchange, xlk, maxL, cap)
    ridx, cidx, MRe, MIm, nRows, nEl = res
    r = (ridx[:nEl] + 1).reshape(-1, 1)
    c = (cidx[:nEl] + 1).reshape(-1, 1)
    vals = np.empty((nEl, 1), dtype=complex)
    vals.real[:, 0] = MRe[:nEl]
    vals.imag[:, 0] = MIm[:nEl]
    return r, c, vals, float(nRows)
