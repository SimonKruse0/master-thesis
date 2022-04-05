# -*- coding: utf-8 -*-
from numpy import abs, sum, sign, arange, atleast_2d, cos, pi
from scr.benchmark_class import Benchmark

# (Continuous, Differentiable, Separable,Non-Scalable, Unimodal)
class Zirilli(Benchmark):

    r"""
    Zettl objective function.
    This class defines the Zirilli [1]_ global optimization problem. This is a
    unimodal minimization problem defined as follows:
    .. math::
        f_{\text{Zirilli}}(x) = 0.25x_1^4 - 0.5x_1^2 + 0.1x_1 + 0.5x_2^2
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.
    *Global optimum*: :math:`f(x) = -0.3523` for :math:`x = [-1.0465, 0]`
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])

        self.global_optimum = [[-1.0465, 0.0]]
        self.fglob = -0.35238603

    def fun(self, x, *args):
        self.nfev += 1

        return 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.1 * x[0] + 0.5 * x[1] ** 2


#Continuous, Differentiable, Non-Separable, Non-Scalable,Multimodal
class Alpine01(Benchmark):

    r"""
    Alpine01 objective function.
    The Alpine01 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:
    .. math::
        f_{\text{Alpine01}}(x) = \sum_{i=1}^{n} \lvert {x_i \sin \left( x_i
        \right) + 0.1 x_i} \rvert
    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-10, 10]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1

        return sum(abs(x * sin(x) + 0.1 * x))


class Weierstrass(Benchmark):

    r"""
    Weierstrass objective function.
    This class defines the Weierstrass [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:
    .. math::
       f_{\text{Weierstrass}}(x) = \sum_{i=1}^{n} \left [
                                   \sum_{k=0}^{kmax} a^k \cos 
                                   \left( 2 \pi b^k (x_i + 0.5) \right) - n
                                   \sum_{k=0}^{kmax} a^k \cos(\pi b^k) \right ]
    Where, in this exercise, :math:`kmax = 20`, :math:`a = 0.5` and
    :math:`b = 3`.
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-0.5, 0.5]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 4` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`
    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005
    TODO line 1591.
    TODO Jamil, Gavana have got it wrong.  The second term is not supposed to
    be included in the outer sum. Mishra code has it right as does the
    reference referred to in Jamil#166.
    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-0.5] * self.N, [0.5] * self.N))

        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1

        kmax = 20
        a, b = 0.5, 3.0

        k = atleast_2d(arange(kmax + 1.)).T
        t1 = a ** k * cos(2 * pi * b ** k * (x + 0.5))
        t2 = self.N * sum(a ** k.T * cos(pi * b ** k.T))

        return sum(sum(t1, axis=0)) - t2