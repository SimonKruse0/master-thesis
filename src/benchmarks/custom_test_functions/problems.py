# -*- coding: utf-8 -*-
from numpy import exp,floor, repeat,size,sqrt, tril, sin,abs, sum, sign, arange, atleast_2d, cos, pi, asarray
import numpy as np
from .benchmark_class import Benchmark
from scipy.optimize import rosen

#Whitley #Ackley01, #Cola, #Corana, #Hartmann6 #Hartmann3
#Corana, CosineMixture

class SimonsTest4_cosine_fuction(Benchmark):
    #sin more and more intensively
    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)
        assert dimensions == 1
        self._bounds = list(zip([-100] * self.N,
                           [100] * self.N))
        self.custom_bounds = ([-5, 5], [-5, 5])

        self.global_optimum = [[0. for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self,x, *args):
        k = np.sign((sin(x/10)>0.9)-0.5)*np.round(x)

        # if sin(x[0]/10)>0.9:
        #     k = np.round(x)
        # else:
        #     k = -np.round(x)

        self.nfev += 1
        return x*sin(k*x/100)+k

class SimonsTest3_cosine_fuction(Benchmark):
    #sin more and more intensively
    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)
        assert dimensions == 1
        self._bounds = list(zip([-100] * self.N,
                           [100] * self.N))
        self.custom_bounds = ([-5, 5], [-5, 5])

        self.global_optimum = [[0. for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self,x, *args):
        if sin(x)>0.9:
            k = np.round(x)
        else:
            k = -np.round(x)

        self.nfev += 1
        return x*sin(k*x)+k


class SimonsTest2(Benchmark):

    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)
        assert dimensions == 1
        self._bounds = list(zip([-100] * self.N,
                           [100] * self.N))
        self.custom_bounds = ([-5, 5], [-5, 5])

        self.global_optimum = [[0. for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True
    def fun(self,x, *args): 
        x = x/200 + 0.5
        self.nfev += 1
        return 50 * (np.sign(x-0.5) + 1)+np.sin(100*x)*10 + 100


class SimonsTest(Benchmark):

    def __init__(self, dimensions=1):
        Benchmark.__init__(self, dimensions)
        assert dimensions == 1
        self._bounds = list(zip([-0.5] * self.N,
                           [0.5] * self.N))
        self.custom_bounds = ([-5, 5], [-5, 5])

        self.global_optimum = [[0. for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True
    def fun(self,x, *args): 
        x = x+ 0.5
        self.nfev += 1
        return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

class Step(Benchmark):

    r"""
    Step objective function.
    This class defines the Step [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    .. math::
        f_{\text{Step}}(x) = \sum_{i=1}^{n} \left ( \lfloor x_i
                             + 0.5 \rfloor \right )^2
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0.5` for
    :math:`i = 1, ..., n`
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        self.custom_bounds = ([-5, 5], [-5, 5])

        self.global_optimum = [[0. for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1

        return sum(floor(abs(x)))

#Discontinuous, Non-Differentiable, Separable, Scalable, Multimodal
class CosineMixture(Benchmark):
    r"""
    Cosine Mixture objective function.
    This class defines the Cosine Mixture global optimization problem. This
    is a multimodal minimization problem defined as follows:
    .. math::
        f_{\text{CosineMixture}}(x) = -0.1 \sum_{i=1}^n \cos(5 \pi x_i)
        - \sum_{i=1}^n x_i^2
    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-1, 1]` for :math:`i = 1, ..., N`.
    *Global optimum*: :math:`f(x) = -0.1N` for :math:`x_i = 0` for
    :math:`i = 1, ..., N`
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    TODO, Jamil #38 has wrong minimum and wrong fglob. I plotted it.
    -(x**2) term is always negative if x is negative.
     cos(5 * pi * x) is equal to -1 for x=-1.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self.change_dimensionality = True
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[-1. for _ in range(self.N)]]
        self.fglob = -0.9 * self.N

    def fun(self, x, *args):
        self.nfev += 1

        return -0.1 * sum(cos(5.0 * pi * x)) - sum(x ** 2.0)


#Discontinuous, Non-Differentiable, Separable, Scalable, Multimodal)
class Corana(Benchmark):
    r"""
    Corana objective function.
    This class defines the Corana [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    .. math::
        f_{\text{Corana}}(x) = \begin{cases} \sum_{i=1}^n 0.15 d_i
        [z_i - 0.05\textrm{sgn}(z_i)]^2 & \textrm{if }|x_i-z_i| < 0.05 \\
        d_ix_i^2 & \textrm{otherwise}\end{cases}
    Where, in this exercise:
    .. math::
        z_i = 0.2 \lfloor |x_i/s_i|+0.49999\rfloor\textrm{sgn}(x_i),
        d_i=(1,1000,10,100, ...)
    with :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., 4`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., 4`
    ..[1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1

        d = [1., 1000., 10., 100.]
        r = 0
        for j in range(4):
            zj = floor(abs(x[j] / 0.2) + 0.49999) * sign(x[j]) * 0.2
            if abs(x[j] - zj) < 0.05:
                r += 0.15 * ((zj - 0.05 * sign(zj)) ** 2) * d[j]
            else:
                r += d[j] * x[j] * x[j]
        return r


class Hartmann3(Benchmark):

    r"""
    Hartmann3 objective function.
    This class defines the Hartmann3 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:
    .. math::
        f_{\text{Hartmann3}}(x) = -\sum\limits_{i=1}^{4} c_i 
        e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}
    Where, in this exercise:
    .. math::
        \begin{array}{l|ccc|c|ccr}
        \hline
        i & & a_{ij}&  & c_i & & p_{ij} &  \\
        \hline
        1 & 3.0 & 10.0 & 30.0 & 1.0 & 0.3689  & 0.1170 & 0.2673 \\
        2 & 0.1 & 10.0 & 35.0 & 1.2 & 0.4699 & 0.4387 & 0.7470 \\
        3 & 3.0 & 10.0 & 30.0 & 3.0 & 0.1091 & 0.8732 & 0.5547 \\
        4 & 0.1 & 10.0 & 35.0 & 3.2 & 0.03815 & 0.5743 & 0.8828 \\
        \hline
        \end{array}
    with :math:`x_i \in [0, 1]` for :math:`i = 1, 2, 3`.
    *Global optimum*: :math:`f(x) = -3.8627821478` 
    for :math:`x = [0.11461292,  0.55564907,  0.85254697]`
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    TODO Jamil #62 has an incorrect coefficient. p[1, 1] should be 0.4387
    """

    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0.11461292, 0.55564907, 0.85254697]]
        self.fglob = -3.8627821478

        self.a = asarray([[3.0, 10., 30.],
                          [0.1, 10., 35.],
                          [3.0, 10., 30.],
                          [0.1, 10., 35.]])

        self.p = asarray([[0.3689, 0.1170, 0.2673],
                          [0.4699, 0.4387, 0.7470],
                          [0.1091, 0.8732, 0.5547],
                          [0.03815, 0.5743, 0.8828]])

        self.c = asarray([1., 1.2, 3., 3.2])

    def fun(self, x, *args):
        self.nfev += 1

        XX = np.atleast_2d(x)
        d = sum(self.a * (XX - self.p) ** 2, axis=1)
        return -sum(self.c * exp(-d))


class Hartmann6(Benchmark):

    r"""
    Hartmann6 objective function.
    This class defines the Hartmann6 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:
    .. math::
        f_{\text{Hartmann6}}(x) = -\sum\limits_{i=1}^{4} c_i
        e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}
    Where, in this exercise:
    .. math::
        \begin{array}{l|cccccc|r}
        \hline
        i & &   &   a_{ij} &  &  & & c_i  \\
        \hline
        1 & 10.0  & 3.0  & 17.0 & 3.50  & 1.70  & 8.00  & 1.0 \\
        2 & 0.05  & 10.0 & 17.0 & 0.10  & 8.00  & 14.00 & 1.2 \\
        3 & 3.00  & 3.50 & 1.70 & 10.0  & 17.00 & 8.00  & 3.0 \\
        4 & 17.00 & 8.00 & 0.05 & 10.00 & 0.10  & 14.00 & 3.2 \\
        \hline
        \end{array}
        \newline
        \
        \newline
        \begin{array}{l|cccccr}
        \hline
        i &  &   & p_{ij} &  & & \\
        \hline
        1 & 0.1312 & 0.1696 & 0.5569 & 0.0124 & 0.8283 & 0.5886 \\
        2 & 0.2329 & 0.4135 & 0.8307 & 0.3736 & 0.1004 & 0.9991 \\
        3 & 0.2348 & 0.1451 & 0.3522 & 0.2883 & 0.3047 & 0.6650 \\
        4 & 0.4047 & 0.8828 & 0.8732 & 0.5743 & 0.1091 & 0.0381 \\
        \hline
        \end{array}
    with :math:`x_i \in [0, 1]` for :math:`i = 1, ..., 6`.
    *Global optimum*: :math:`f(x_i) = -3.32236801141551` for
    :math:`{x} = [0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162,
    0.65730054]`
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=6):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0.20168952, 0.15001069, 0.47687398, 0.27533243,
                                0.31165162, 0.65730054]]

        self.fglob = -3.32236801141551

        self.a = asarray([[10., 3., 17., 3.5, 1.7, 8.],
                          [0.05, 10., 17., 0.1, 8., 14.],
                          [3., 3.5, 1.7, 10., 17., 8.],
                          [17., 8., 0.05, 10., 0.1, 14.]])

        self.p = asarray([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                          [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                          [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                          [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

        self.c = asarray([1.0, 1.2, 3.0, 3.2])

    def fun(self, x, *args):
        self.nfev += 1

        XX = np.atleast_2d(x)
        d = sum(self.a * (XX - self.p) ** 2, axis=1)
        return -sum(self.c * exp(-d))

class Corana(Benchmark):
    r"""
    Corana objective function.
    This class defines the Corana [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    .. math::
        f_{\text{Corana}}(x) = \begin{cases} \sum_{i=1}^n 0.15 d_i
        [z_i - 0.05\textrm{sgn}(z_i)]^2 & \textrm{if }|x_i-z_i| < 0.05 \\
        d_ix_i^2 & \textrm{otherwise}\end{cases}
    Where, in this exercise:
    .. math::
        z_i = 0.2 \lfloor |x_i/s_i|+0.49999\rfloor\textrm{sgn}(x_i),
        d_i=(1,1000,10,100, ...)
    with :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., 4`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., 4`
    ..[1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1

        d = [1., 1000., 10., 100.]
        r = 0
        for j in range(4):
            zj = floor(abs(x[j] / 0.2) + 0.49999) * sign(x[j]) * 0.2
            if abs(x[j] - zj) < 0.05:
                r += 0.15 * ((zj - 0.05 * sign(zj)) ** 2) * d[j]
            else:
                r += d[j] * x[j] * x[j]
        return r


class Cola(Benchmark):
    r"""
    Cola objective function.
    This class defines the Cola global optimization problem. The 17-dimensional
    function computes indirectly the formula :math:`f(n, u)` by setting
    :math:`x_0 = y_0, x_1 = u_0, x_i = u_{2(i2)}, y_i = u_{2(i2)+1}` :
    .. math::
        f_{\text{Cola}}(x) = \sum_{i<j}^{n} \left (r_{i,j} - d_{i,j} \right )^2
    Where :math:`r_{i, j}` is given by:
    .. math::
        r_{i, j} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
    And :math:`d` is a symmetric matrix given by:
    .. math::
        \{d} = \left [ d_{ij} \right ] = \begin{pmatrix}
        1.27 &  &  &  &  &  &  &  & \\
        1.69 & 1.43 &  &  &  &  &  &  & \\
        2.04 & 2.35 & 2.43 &  &  &  &  &  & \\
        3.09 & 3.18 & 3.26 & 2.85  &  &  &  &  & \\
        3.20 & 3.22 & 3.27 & 2.88 & 1.55 &  &  &  & \\
        2.86 & 2.56 & 2.58 & 2.59 & 3.12 & 3.06  &  &  & \\
        3.17 & 3.18 & 3.18 & 3.12 & 1.31 & 1.64 & 3.00  & \\
        3.21 & 3.18 & 3.18 & 3.17 & 1.70 & 1.36 & 2.95 & 1.32  & \\
        2.38 & 2.31 & 2.42 & 1.94 & 2.85 & 2.81 & 2.56 & 2.91 & 2.97
        \end{pmatrix}
    This function has bounds :math:`x_0 \in [0, 4]` and :math:`x_i \in [-4, 4]`
    for :math:`i = 1, ..., n-1`.
    *Global optimum* 11.7464.
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=17):
        Benchmark.__init__(self, dimensions)

        self._bounds = [[0.0, 4.0]] + list(zip([-4.0] * (self.N - 1),
                                               [4.0] * (self.N - 1)))

        self.global_optimum = [[0.651906, 1.30194, 0.099242, -0.883791,
                                -0.8796, 0.204651, -3.28414, 0.851188,
                                -3.46245, 2.53245, -0.895246, 1.40992,
                                -3.07367, 1.96257, -2.97872, -0.807849,
                                -1.68978]]
        self.fglob = 11.7464

        self.d = asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.27, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1.69, 1.43, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2.04, 2.35, 2.43, 0, 0, 0, 0, 0, 0, 0],
                 [3.09, 3.18, 3.26, 2.85, 0, 0, 0, 0, 0, 0],
                 [3.20, 3.22, 3.27, 2.88, 1.55, 0, 0, 0, 0, 0],
                 [2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0, 0, 0, 0],
                 [3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00, 0, 0, 0],
                 [3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32, 0, 0],
                 [2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97, 0.]])

    def fun(self, x, *args):
        self.nfev += 1

        xi = atleast_2d(asarray([0.0, x[0]] + list(x[1::2])))
        xj = repeat(xi, size(xi, 1), axis=0)
        xi = xi.T

        yi = atleast_2d(asarray([0.0, 0.0] + list(x[2::2])))
        yj = repeat(yi, size(yi, 1), axis=0)
        yi = yi.T

        inner = (sqrt(((xi - xj) ** 2 + (yi - yj) ** 2)) - self.d) ** 2
        inner = tril(inner, -1)
        return sum(sum(inner, axis=1))

class Ackley01(Benchmark):

    r"""
    Ackley01 objective function.
    The Ackley01 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:
    .. math::
        f_{\text{Ackley01}}(x) = -20 e^{-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n
         x_i^2}} - e^{\frac{1}{n} \sum_{i=1}^n \cos(2 \pi x_i)} + 20 + e
    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-35, 35]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`
    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005
    TODO: the -0.2 factor in the exponent of the first term is given as
    -0.02 in Jamil et al.
    """

    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-35.0] * self.N, [35.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        u = sum(x ** 2)
        v = sum(cos(2 * pi * x))
        return (-20. * exp(-0.2 * sqrt(u / self.N))
                - exp(v / self.N) + 20. + exp(1.))

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

class Whitley(Benchmark):

    r"""
    Whitley objective function.
    This class defines the Whitley [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    .. math::
        f_{\text{Whitley}}(x) = \sum_{i=1}^n \sum_{j=1}^n
                                \left[\frac{(100(x_i^2-x_j)^2
                                + (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2
                                + (1-x_j)^2)+1 \right]
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10.24, 10.24]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`
    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    TODO Jamil#167 has '+ 1' inside the cos term, when it should be outside it.
    """

    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.24] * self.N,
                           [10.24] * self.N))
        self.custom_bounds = ([-1, 2], [-1, 2])

        self.global_optimum = [[1.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1

        XI = x
        XJ = atleast_2d(x).T

        temp = 100.0 * ((XI ** 2.0) - XJ) + (1.0 - XJ) ** 2.0
        inner = (temp ** 2.0 / 4000.0) - cos(temp) + 1.0
        return sum(sum(inner, axis=0))


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


class Rosenbrock(Benchmark):

    r"""
    Rosenbrock objective function.
    This class defines the Rosenbrock [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:
    .. math::
       f_{\text{Rosenbrock}}(x) = \sum_{i=1}^{n-1} [100(x_i^2
       - x_{i+1})^2 + (x_i - 1)^2]
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5, 10]` for :math:`i = 1, ..., n`.
    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-30.] * self.N, [30.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]

        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1

        return rosen(x)
