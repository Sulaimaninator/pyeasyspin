# Differences from EasySpin 6.0.12

pyeasyspin reproduces EasySpin 6.0.12's results. This file lists every place where it differs on
purpose:

1. errors in EasySpin 6.0.12 that are corrected (for many, a module flag restores EasySpin's
   original behaviour);
2. interface differences that come from Python;
3. small numerical differences from different libraries and algorithms;
4. EasySpin limitations that are kept.

## 1. Corrected errors in EasySpin 6.0.12

### Resonance fields, frequencies and ENDOR

- **`resfields` with hybrid (first-order) nuclei: intensities on the wrong lines.** EasySpin builds
  `NucTransitionRates = abs(Vu'*Vv).^2`, whose rows are u states, but takes the line shifts as
  `dEv(vidx) - dEu(uidx)` with vidx the row index. So |<u_i|v_j>|² is attached to the line at
  dEv(i) − dEu(j) instead of dEv(j) − dEu(i). (`resfreqs_matrix` does the same step consistently.)
  Line positions are unchanged, and for I = 1/2 the output is EasySpin's. For I ≥ 1 with nuclear
  state mixing, EasySpin's intensities were off from full diagonalization by up to 25 % of the
  strongest line in a test with 14N (tilted A and Q, X band); the corrected pairing is within
  1e-3, the accuracy of the first-order treatment. Flag:
  `pyeasyspin.fn.resfields.EASYSPIN_HYBRID_PAIRING = True`.
- **`resfields`: gradients not re-sorted.** EasySpin re-sorts Pdat, Idat and Wdat by transition but
  not the gradient output Gdat, which then belongs to the wrong rows. The port sorts Gdat too.
- **`resfields` with A strain and one transition** crashed in a reshape; the port computes the
  per-transition mean.
- **"A strain cannot be used with a tilted A tensor"** (`resfields`, `resfields_perturb`,
  `resfreqs_perturb`, `resfreqs_matrix`) was raised for g strain with a tilted A tensor even when
  no A strain was given. It is now raised only when A strain is set; the limitation itself is kept.
- **`resfields_perturb`, `resfreqs_perturb`:**
  - a full 3x3 D matrix (`Sys.fullD`) left `D` undefined and crashed; the port uses `Sys.D`, as both
    functions already do for g;
  - correlated D/E strain (`Sys.DStrainCorr ~= 0`) computed the E component from the already
    rotated dH/dD; the port uses the original, as EasySpin's matrix route does;
  - D-strain widths with nuclei were laid out with `repmat` (cycling over mS transitions) while the
    rows are mS-major blocks of nuclear sub-lines, so S > 1/2 systems with nuclei got the wrong
    widths; the port uses the layout the same functions use for intensities.
- **`resfields_perturb`:** when any H-strain width was 0 (for example HStrain = [a b 0] at
  theta = 0, a grid point), all widths became empty. The port keeps the widths.
- **`resfreqs_perturb`:**
  - photoselection referred to an undefined variable and crashed;
  - circular polarization used an undefined variable and crashed;
  - H-strain widths were discarded in every branch except D strain; they are now combined as in
    `resfields_perturb`;
  - the g-strain width was 1000 times too large and used the field of one orientation only.

  Frequencies, intensities (all four microwave modes) and widths now equal `resfreqs_matrix` on the
  principal axes.
- **`resfreqs_perturb` with `Opt.ImmediateBinning`** crashed on undefined variables (and its axis
  would come from `Exp.Range`, which is undefined in frequency sweeps). The port raises a clear
  error instead.
- **`endorfrq_perturb`: second-order quadrupole term.** EasySpin used `2*trace(P)^2`, which is 0
  because P is traceless; the analogous zero-field term uses `trace(D^2)`, and the port uses
  `trace(P^2)`. Against matrix diagonalization (14N, 3.35 T) the error falls from 2.6e-2 to 2e-7 MHz.
- **`cubicsolve`:** for a root at t = 0, the C routine returned derivatives from uninitialised
  memory; the port returns the true derivatives there.

### cw EPR

- **`pepper` with `Opt.ImmediateBinning`** tested an undefined variable and always crashed; the port
  uses the intended one.
- **Always-shown warnings crashed simulations.** EasySpin logs seven warnings with level -Inf (for
  example pepper's "eigenfields method ignores strains", or a negative intensity in resfields), and
  its own `logmsg` turns level < 0 into an error whenever a log level is set, which is during every
  simulation. The port logs them as warnings.
- **`curry` at zero temperature:** EasySpin fixes only the (1,1) element of the population matrix,
  so for a T = 0 that is not the first temperature the whole output column is NaN. The port sets
  the whole ground-state row. Only results that are NaN in EasySpin change.
- **`chili`, `Opt.Rescale` (the default).** EasySpin divides each Liouvillian by its largest element
  and later divides the summed spectrum once, by the scale of the last Liouvillian. Where the scale
  varies (explicit field sweeps, MOMD orientation grids), every point keeps a factor
  scale_i/scale_last, so the spectrum depends on a purely numerical option (1.5 % across a 5 mT
  sweep). The port divides each solve by its own scale; Rescale on and off then agree to 1e-9.
  Results differ from EasySpin by 6e-4 to 2e-2 of the peak in those cases. Flag:
  `pyeasyspin.fn.chili.EASYSPIN_RESCALE = True`.
- **`chili`, director azimuth.** EasySpin feeds phi of `Exp.SampleFrame` (and of the MOMD grid) into
  the Liouvillian, where the fast path reads only the real part of a complex Wigner function and the
  general path builds an unphysical Liouvillian. A rotation of the director about the field cannot
  change a cw spectrum, and EasySpin's own LMK-basis route is phi-invariant; in EasySpin 6.0.12 the
  spectrum changed by 70 % of the peak (fast path) and 10 times the peak (general path). The port
  uses phi = 0 in the Liouvillian; the ordering weights still use phi. Flag:
  `pyeasyspin.fn.chili.EASYSPIN_DIRECTOR_PHI = True`.
- **`chili`, rank-1 (antisymmetric) tensor terms** in `liouvhamiltonian` and `liouvhamiltonian_LMK`
  were wrong in three ways (sign conventions of the 3j symbols, completion of a Hermitian block by
  a plain transpose and double counting, and a missing (-1)^(L1+j+L2) factor for j = 1). The port
  uses the rank-2 formula with j = 1 and computes all pairs. The block now equals a brute-force
  SO(3) quadrature to 1e-12, and rotating a g tensor with an antisymmetric part leaves the spectrum
  invariant to 5e-12 (EasySpin's code changed it by 0.47 of the peak).
- **`magint` symmetry flags** were set from the rank-2 components only, so chili's default basis
  dropped functions that rank-1 terms couple (12.5 % of the peak off for a g tensor with an
  antisymmetric part). The port clears the flags when rank-1 terms are present; symmetric tensors
  are unchanged.
- **`chili_lanczos` with `Opt.Lentz = false`** returned the spectrum evaluated on its 201-point
  convergence-test axis instead of the sweep axis (wrong length, or a crash). The port evaluates the
  returned spectrum on the sweep axis.
- **`diffsuperop_LMK` with a potential** collected nonzero terms by horizontal concatenation of
  column vectors, a dimension error for any real potential. The port concatenates the lists; the
  result equals `diffsuperop` in the same basis to 1e-16. (chili 6.0.12 does not call this function.)

### Pulse EPR

- **`sf_peaks`, scheme 15 (`[1 2 2 1]`, 2D-CP):** the C routine used the imaginary part of the wrong
  factor in one complex product. The port uses the matching factor and agrees with a brute-force
  pathway sum. Flag: `pyeasyspin.fn.sf_peaks.EASYSPIN_CASE15_TYPO = True`.
- **`propint` with stored propagators** used the propagators one interval off from the direct path
  (a difference of 2.8e-3 on a test pulse). Both paths now give the same result.
- **`evolve`, schemes `[1 2 1]`, `[1 2 2 1]` and `[1 2 -2 1]`:** the adjoint of the mixing propagator
  was computed once per outer step but the propagator was updated in the inner loop, so the final
  density matrix was not a similarity transform (not even Hermitian). The port uses the matching
  adjoint and agrees with brute-force propagation.

### Fitting

- **`esfit` uncertainties at a parameter bound.** EasySpin's `jacobianest` evaluates the model at
  x0 ± δ and ignores the bounds, so a width or strain fitted to a zero bound is evaluated at a
  negative value and the model fails after convergence. The port keeps the finite-difference
  stencil inside the bounds (one-sided where needed, as in DERIVEST, which jacobianest comes from);
  `esfit_levmar`'s forward step does the same. Uncertainties change slightly when the stencil would
  leave the box. Flags: `pyeasyspin.fn.esfit.BOUNDED_FD = False` and
  `pyeasyspin.fn.esfit_levmar.BOUNDED_FD = False` restore EasySpin's stencil.
- **`esfit` 'diff' and 'fft' targets were swapped** (fitting with 'diff' compared Fourier magnitudes
  and vice versa). The port computes what the names say.
- Minor fixes of crashes: `hccm` HC2–HC5 with row residuals; `esfit_levmar` messages for negative
  stop codes; `printparlist` without uncertainties.

### Trajectories and molecular dynamics

- **`cardamom_estimatedifftensor`** differentiated step-to-step rotations as if they were an
  orientation trajectory, which makes the diffusion tensor 50–1000 times too small. The port uses
  orientations relative to the first frame and recovers the rate within 25 %. Flag:
  `pyeasyspin.fn.cardamom_estimatedifftensor.EASYSPIN_INCREMENTS = True`.
- **`cardamom`, 'diffusion' model with `Sys.Potential`:** the equilibration half was dropped along
  the trajectory dimension instead of the time dimension, leaving no trajectories for
  `Opt.Method = 'ISTOs'`. Flag: `pyeasyspin.fn.cardamom.EASYSPIN_QTRAJ_INDEXING = True`.
- **`stochtraj_diffusion`, starting orientations for `Sys.Potential = [L M K lambda]`:** rejection
  sampling left out the volume element sin(beta), so the start was not the equilibrium distribution
  (<cos²β> = 1/2 instead of 1/3 for a negligible potential). Flag:
  `pyeasyspin.fn.stochtraj_diffusion.EASYSPIN_NO_SINBETA = True`.
- **`stochtraj_proprottraj`, Leimkuhler–Matthews integrator** read one page past its random numbers
  and stopped with an index error. The port draws one more set. Flag:
  `pyeasyspin.fn.stochtraj_proprottraj.EASYSPIN_LM_PAGES = True`.
- **`mdhmm_em` with several trajectories** took the means and covariances from the last trajectory
  only while normalising by all of them (covariances about nTraj times too small). The port
  accumulates over all trajectories (Baum–Welch for several sequences). Flag:
  `pyeasyspin.fn.mdhmm_em.EASYSPIN_LAST_TRAJECTORY = True`.
- **`md_readdcd`** failed on X-PLOR DCD files and CHARMM files without the extra-block flag. Flag:
  `pyeasyspin.fn.md_readdcd.EASYSPIN_EXTRABLOCK_FIELD = True`.
- **`md_savedcd`** could not write a file in EasySpin 6.0.12 (undefined variables, and reader code
  run on the write handle). The port writes the file its header describes (X-PLOR, VERSION 0).

### Other

- **`eprconvert`:** the eV input factor was inverted (`planck/echarge` instead of `echarge/planck`),
  turning 1 eV into about 4e-15 Hz. EasySpin's own display path and `unitconvert` use the correct
  direction, as does the port.

## 2. Interface differences

- **Outputs and nargout.** Several outputs come back as a tuple. Where EasySpin decides by the
  number of requested outputs, the default is noted in [API.md](API.md), for example:
  - `gaussian`, `lorentzian`, `voigtian` return the absorption only; `nargout=2` gives both parts;
  - `ham_ez`, `ham_nz`, `ham_oz` return one output when given a field vector, three otherwise;
  - `garlic`, `salt`, `pepper`, `chili`, `saffron` return `(x, spectrum)`;
  - `vec2ang` returns `(phi, theta)`; `nargout=1` gives the stacked 2 x N array;
  - `eprload` returns `(Abscissa, Data, Parameters)`;
  - `isotopologues` returns a single `Struct` for one isotopologue and a list otherwise.
- **Plotting.** Nothing is plotted unless `plot=True` is passed; figures are static Matplotlib
  figures that also return their outputs. EasySpin's GUIs are not included (the esfit window, the
  `isotopes` browser). In `pulse` and `exciteprofile` the interactive checkboxes become a static
  figure; plotting helpers return a `Struct` with the plotted data; viridis replaces parula; the
  hover and info-text callbacks of `levelsplot` and interactive parts of the saffron/spidyan plots
  are not reproduced. `apowin`, `basecorr`, `rescaledata` and `rcfilt` do not plot. `blochsteady`
  does not print its imaginary/real ratios.
- **Headless versions of GUI functions.** `eprconvert` takes two of frequency, field and g and returns
  the third plus the GUI's display strings. `nucfrq2d` returns the curve data. `eprload` with a
  folder name raises instead of opening a file dialog.
- **Random numbers** come from NumPy. Functions that draw random numbers (`addnoise`, `sphrand`,
  `cardamom`, `stochtraj_*`, `mdhmm`, `mdhmm_kmeans`, `cardamom_rejectionsample3d`) take an `rng`
  keyword (a NumPy Generator or RandomState; default the global stream) and agree with EasySpin
  statistically, not draw for draw. NumPy cannot reproduce MATLAB's randn, randi or randperm
  streams.
- **Multicore.** `esfit` computes the model points its algorithms will need (swarm particles, a
  generation, grid blocks, Jacobian steps, the simplex candidates) on worker processes, and the
  algorithm then uses them in its usual order, so fits give the same numbers as serial ones.
  `Opt.maxTime` counts the CPU time of the main process and its workers, as MATLAB's cputime
  counts all of MATLAB's threads. With Monte Carlo, trial points are drawn in batches in the usual
  order, so after an early stop the random stream has advanced past the last trial used.
- **`chili` diagnostics** (`Opt.Diagnostics`) are stored in `pyeasyspin.fn.chili.DIAGNOSTICS[name]`
  instead of the MATLAB base workspace.
- **Files.** `getmatrix` maps MATLAB fread precisions to NumPy types (MATLAB-only types such as
  `bitN` are not supported) and drops trailing singleton dimensions, so 1-D BES3T data come back
  N x 1. Binary JEOL strings are decoded as Latin-1; text files as UTF-8 with a Latin-1 fallback.
  ORCA text and property files with a UTF-16 or UTF-8 byte-order mark are decoded, as MATLAB's
  fopen does. `eprsave` writes DSC lines literally (MATLAB passes them through fprintf, so `%` or
  `\` in a title is interpreted there). `md_readtrr` parses the file in memory instead of writing
  temporary files to the working directory.
- **`ham` with order-0 higher-order Zeeman terms only and no field** returns the order-0 matrix,
  where EasySpin adds a cell array to a matrix and errors.

## 3. Numerical differences

These come from different numerical libraries or algorithms. Unless stated, they are at the level
of rounding.

- **Eigenvector phases and signs** depend on the LAPACK build. Energies and every phase-invariant
  quantity agree; the Euler angles that the ORCA readers derive for g, D, A and Q frames can differ
  by sign choices while the tensors are identical.
- **Least squares** (`basecorr`, `exponfit`, `transmitter`) uses NumPy's `lstsq` (SVD) instead of
  MATLAB's QR-based mldivide.
- **`gaussian` with diff = -1 and dispersion** is the integral of SciPy's Dawson function instead of
  MATLAB's symbolic hypergeom (which needs the Symbolic Math Toolbox).
- **`sphgrid` triangulation** uses SciPy's Delaunay (Qhull). For co-circular points the two may
  choose different triangles; the triangle lists and areas of EasySpin's test suite are identical.
- **Interpolation:** `interp1` 'spline' and `griddedInterpolant` 'spline' are SciPy not-a-knot
  cubic splines, 'pchip' is SciPy's PchipInterpolator (complex data as real and imaginary parts).
  `fresnelC`/`fresnelS` evaluate EasySpin's own spline table.
- **`xml2struct`** parses with Python's minidom instead of Java Xerces (attributes in sorted order,
  as Xerces stores them).
- **`resonatorprofile` at zero frequency** returns 0 where NumPy would give NaN, as MATLAB does.
- **Random fuzz.** `resfields` and `resfreqs_matrix` add random fuzz of relative size
  `Opt.FuzzLevel` (1e-10) to the Hamiltonian, as EasySpin does, but from NumPy's generator. Results
  agree to about 1e-10 times the conditioning of the problem.
- **Ties in transition selection** (`endorfrq`): when two transition rates are equal to rounding,
  their order follows eigensolver rounding and can differ from MATLAB's.
- **`pepper`** line positions come from spline interpolation of resonance fields (`cubicsolve`),
  which is ill-conditioned for nearly linear segments. Most spectra agree far better than 1e-6 of
  the peak; in a few cases of EasySpin's test suite the difference reaches 3e-5 of the peak.
- **`chili`'s Lanczos solver** (also used by `cardamom`) runs without reorthogonalization and
  amplifies rounding: a change of g by one part in 1e15 moves a nitroxide spectrum by about 6e-7 of
  the peak. Results usually agree with EasySpin to 1e-6–1e-5 of the peak; in large or poorly
  conditioned problems, where the iteration stops at a different step, differences of up to about
  3e-4 of the peak occur in EasySpin's test suite.
- **`s_thyme`** propagates in Numba kernels with a different summation order (agreement 4e-12); its
  relaxation steady states solve systems with condition numbers near 1e14, where LU factorizations
  from different libraries differ by about 1e-13 absolute.
- **`expm_(A, 'fast')`** for n ≥ 50 uses the exact 1-norm of A^m where MATLAB uses the randomized
  normest1; this only affects the choice of the scaling power.
- **`oripotentialplot`** computes the population normalization with a refined quadrature (relative
  1e-10) instead of MATLAB's integral2/integral3 (AbsTol 1e-6).
- **Optimizers.** `esfit`'s 'lsqnonlin' method (MATLAB Optimization Toolbox) is SciPy's
  `least_squares` with the trust-region-reflective algorithm, lsqnonlin's default; its trajectory
  differs from MATLAB's. `mdhmm` uses SciPy's L-BFGS-B (the same algorithm and settings as
  EasySpin's compiled routine): identical iterates for small models, slightly different stopping
  points for large ones.

## 4. EasySpin limitations that are kept

- **Spectra that run past `Exp.Range`.** EasySpin warns "Spectrum exceeds sweep range. Artifacts at
  lower and upper limits possible." Resonances outside the window are dropped, and the padding
  before convolution repeats the edge values. For a rhombic powder (g 2.05/2.00/1.95) cut by a
  335–345 mT window, the error was up to 43 % of the peak at the window edges and about 1 % at the
  centre with Lorentzian broadening. Workaround: simulate on a wider range and crop.
- **A strain with a tilted A tensor** is still not supported (see section 1 for the error message
  that is no longer raised without A strain).
- **`spidyan`:** the cell-array branch of its phase shift multiplies by conj(Phase), as EasySpin
  does, which is inconsistent with the array branch.
