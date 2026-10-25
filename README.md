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

```python
import pyeasyspin as es

# Nitroxide, rigid limit, X band
Sys = es.Struct(g=[2.0088, 2.0061, 2.0027], Nucs='14N', A=[16, 16, 86], lwpp=[0, 0.3])
Exp = es.Struct(mwFreq=9.5, Range=[328, 349], nPoints=2048)

B, spc = es.pepper(Sys, Exp)              # field axis (mT) and first-derivative spectrum
B, spc = es.pepper(Sys, Exp, plot=True)   # the same, plotted with Matplotlib
```

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
