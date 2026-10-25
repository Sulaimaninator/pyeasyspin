"""Default fit options and their descriptions for the esfit algorithms (private)."""
from .._registry import nout
from .._struct import Struct


@nout(2)
def esfit_algdefaults(algorithm=None):
    """esfit_algdefaults(algorithm) -> (FitOpt, info). algorithm is the long name
    ('Nelder-Mead simplex', 'Levenberg-Marquardt', 'Monte Carlo', 'genetic algorithm',
    'grid search', 'particle swarm'). MATLAB [] defaults are None."""
    info = Struct()
    FitOpt = Struct()
    info.TolFun = 'termination tolerance for error function value'
    FitOpt.TolFun = 1e-5
    info.maxTime = 'time, in minutes, after which esfit will terminate even if the fitting has not yet converged'
    FitOpt.maxTime = float('inf')
    FitOpt.IterFcn = None
    FitOpt.IterationPrintFunction = lambda str_: str_
    FitOpt.InfoPrintFunction = lambda str_: str_
    if algorithm == 'Nelder-Mead simplex':
        info.delta = 'edge length for initial simplex'
        FitOpt.delta = 0.1
        info.TolEdgeLength = 'termination tolerance for the length of the parameter step'
        FitOpt.TolEdgeLength = 1e-4
        info.SimplexPars = ('[rho, chi, psi, sigma](reflection coefficient, expansion coefficient,'
                            'contraction coefficient, reduction coefficient)')
        FitOpt.SimplexPars = None
        info.ScaleParams = 'rescale fitting parameters to within [-1,1] interval'
        FitOpt.ScaleParams = False
        FitOpt.RandomStart = 0
    elif algorithm == 'Levenberg-Marquardt':
        info['lambda'] = 'starting value of Marquardt parameter'   # 'lambda' is a Python keyword
        FitOpt['lambda'] = 1e-3
        info.delta = 'relative step for difference approximation'
        FitOpt.delta = 1e-3
        info.TolStep = 'termation tolerance for parameter step (small step stops)'
        FitOpt.TolStep = 1e-6
        info.Gradient = 'termation tolerance for gradient (small gradient stops)'
        FitOpt.Gradient = FitOpt.TolFun
        info.ScaleParams = 'rescale fitting parameters to within [-1,1] interval'
        FitOpt.ScaleParams = False
        FitOpt.RandomStart = 0
    elif algorithm == 'Monte Carlo':
        info.nTrials = 'number of random trial simulations before termination'
        FitOpt.nTrials = 20000
    elif algorithm == 'genetic algorithm':
        info.PopulationSize = ('size of the population, i.e. number of parameter sets and simulations '
                               'in one generation')
        FitOpt.PopulationSize = 20
        info.EliteCount = ('number of populations (ordered in terms of decreasing score) carried over to '
                           'the next generation without recombination and mutation')
        FitOpt.EliteCount = None
        info.maxGenerations = 'maximum number of generations the algorithm should run'
        FitOpt.maxGenerations = 10000
    elif algorithm == 'grid search':
        info.GridSize = 'number or an array that specifies how many grid points there should be for each parameter'
        FitOpt.GridSize = 7
        info.randGrid = 'randomize order of gridpoints'
        FitOpt.randGrid = True
        FitOpt.maxGridPoints = 1e5
    elif algorithm == 'particle swarm':
        info.nParticles = 'number of particles in the particle swarm'
        FitOpt.nParticles = None
        info.SwarmParams = ('[k, w, c1, c2](velocity limit, inertial coefficient,'
                            'cognitive coefficient, social coefficient)')
        FitOpt.SwarmParams = [0.2, 0.5, 2, 1]
        info.TolStallIter = ("maximum number of consecutive iterations over which the best function "
                             "value doesn't change")
        FitOpt.TolStallIter = 6
    return FitOpt, info
