# pyeasyspin

An unofficial Python port of [EasySpin](https://easyspin.org), the MATLAB toolbox for simulating
and analysing electron paramagnetic resonance (EPR) spectra. It follows EasySpin 6.0.12 function by
function: the same function names, the same `Sys`/`Exp`/`Opt` fields, the same units, and results
that agree with EasySpin to about 1e-6 of the peak (exceptions are listed in
[docs/DEVIATIONS.md](docs/DEVIATIONS.md)).

pyeasyspin is not affiliated with or endorsed by the EasySpin authors. If you use it in published
work, please cite EasySpin:

> S. Stoll, A. Schweiger, "EasySpin, a comprehensive software package for spectral simulation and
> analysis in EPR", *J. Magn. Reson.* **178**, 42–55 (2006).

## Installation

```bash
pip install pyeasyspin
```

or, for the latest version from GitHub:

```bash
pip install git+https://github.com/Sulaimaninator/pyeasyspin
```

Python 3.11 or later. Dependencies: NumPy, SciPy and Numba; Matplotlib for plotting
(`pip install "pyeasyspin[plot]"`). The first call of some functions takes a few seconds while
Numba compiles them.

## Example

A script reads much like its MATLAB version:

```python
from pyeasyspin import *
import matplotlib.pyplot as plt

# Nitroxide, rigid limit, X band
Sys = Struct()
Sys.g = [2.0088, 2.0061, 2.0027]
Sys.Nucs = '14N'
Sys.A = [16, 16, 86]          # MHz
Sys.lwpp = [0, 0.3]           # mT

Exp = Struct()
Exp.mwFreq = 9.5              # GHz
Exp.Range = [328, 349]        # mT

B, spc = pepper(Sys, Exp)     # field axis (mT) and first-derivative spectrum
plt.plot(B, spc)
```

`import pyeasyspin as es` (then `es.pepper`, `es.Struct(g=2.0, lwpp=1.0)`, ...) works as well, and
`pepper(Sys, Exp, plot=True)` plots the spectrum the way MATLAB's `pepper(Sys,Exp)` does.

## Speed

A single simulation runs at about the speed of EasySpin in MATLAB (median of five calls, Intel
Core i7-1255U laptop):

| | EasySpin 6.0.12 | pyeasyspin |
|---|---|---|
| Nitroxide powder (`pepper`) | 0.25 s | 0.13 s |
| Cu(II) powder (`pepper`) | 0.047 s | 0.035 s |
| Mn(II), S = 5/2 (`pepper`) | 0.23 s | 0.26 s |
| Three-pulse ESEEM (`saffron`) | 0.043 s | 0.019 s |
| Nitroxide, slow motion (`chili`) | 0.19 s | 0.58 s |

Work that consists of many independent simulations runs on all CPU cores, with no extra toolbox:

- `esfit` computes the model points of each step in parallel (the particles of a swarm iteration,
  a genetic generation, grid points, finite-difference Jacobians, simplex candidates). The fit
  follows the same path and gives the same numbers as a serial run.
- `pepper` splits large orientation grids across cores.
- `pyeasyspin.parallel_map` runs any batch of simulations, such as a parameter scan.

Fitting a Cu(II) powder spectrum (7 parameters; the same data, starting point and settings), on
the same laptop:

| | EasySpin 6.0.12 | pyeasyspin, 1 core | pyeasyspin, all cores |
|---|---|---|---|
| Levenberg–Marquardt | 125 s | 126 s | 38 s |
| Particle swarm, 20 particles | 2186 s | | 692 s |

Both fits end at EasySpin's result: the rmsd agrees to six digits, and the fitted parameters agree
to 2e-6 (Levenberg–Marquardt) and 4e-11 (swarm, which followed EasySpin's random-number path). The
speed-up depends on the number of cores.

`pyeasyspin.config.Workers` sets the number of worker processes (`'auto'`, the default, uses all
cores; `1` turns parallel evaluation off). The first parallel call starts the workers, which takes
a few seconds; later calls reuse them.

## What is included

All of EasySpin's simulation and analysis functions:

- cw EPR: `pepper` (solids), `garlic` (isotropic and fast motion), `chili` (slow motion),
  `cardamom` (trajectory-based);
- ENDOR (`salt`), pulse EPR (`saffron`, `spidyan`), resonance fields and frequencies, energy levels;
- fitting (`esfit`, without the GUI);
- spin operators and Hamiltonians, rotations and orientation grids, line shapes, isotope data,
  physical constants, data import (`eprload`) and processing.

Not included: EasySpin's interactive GUIs (the `esfit` window and the `isotopes` browser), the
deprecated `esfit_legacy`, and installation and update helpers.

## Differences from MATLAB

- Structures are `pyeasyspin.Struct`, a `dict` with attribute access (`Sys.g` and `Sys['g']`).
- Functions with several outputs return a tuple, in MATLAB's order. Where EasySpin's behaviour
  depends on the number of requested outputs, the function takes a `nargout` keyword.
- Row and column vectors come back as 1-D NumPy arrays, so they plot and index like ordinary
  vectors; matrices keep MATLAB's layout. `pyeasyspin.config.MatlabShapes = True` returns
  MATLAB's exact 1 × N and N × 1 shapes instead.
- EasySpin plots when a function is called without outputs; here that is `plot=True`.
- Indices that users see (transitions, sites, ...) stay 1-based, as in EasySpin.
- Random numbers come from NumPy, so most stochastic results (noise, trajectories) agree with
  EasySpin statistically rather than draw for draw.
- Some genuine errors in EasySpin 6.0.12 are corrected. Each correction is listed in
  [docs/DEVIATIONS.md](docs/DEVIATIONS.md). For many of them a module flag (for example
  `pyeasyspin.fn.chili.EASYSPIN_RESCALE = True`) restores EasySpin's original behaviour.

[docs/API.md](docs/API.md) lists every function with its Python signature.

## How it was validated

EasySpin 6.0.12's own test suite was run in MATLAB with every function call recorded, and each
recorded call was replayed in Python and compared output by output: 1e-6 of the peak for spectra
and 1e-10 for exact mathematics. The recorded data are too large to distribute. The tests in this
repository check the package on its own, including a comparison with EasySpin reference spectra
for standard systems (`tests/reference`):

```bash
pip install "pyeasyspin[test]"
pytest
```

## License

MIT, as EasySpin. See [LICENSE](LICENSE).
