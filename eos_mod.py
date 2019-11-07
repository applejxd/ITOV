import math
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, integrate
from scipy import optimize
from scipy.interpolate import interp1d

C_CONST: float = 2.99792458e+10  # cm/s


class EoS:
    """
    状態方程式 p=p(rho) とその log-log 関数 zeta=zeta(xi)
    """

    @staticmethod
    def f0(x: float) -> float:
        return 1. / (math.exp(x) + 1.)

    def zeta_from_xi(self, xi: float) -> float:
        """
        状態方程式の log-log 関数
        :param xi: 密度 / [g/cm^3]の常用対数
        :return: 圧力 / [dyn/cm^2]の常用対数
        """
        a: List[float] = [6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105, 0.8938, 6.54,
                          11.4950, -22.775, 1.5707, 4.3, 14.08, 27.80, -1.653, 1.50, 14.67]
        zeta: float = ((a[0] + a[1] * xi + a[2] * xi ** 3) / (1 + a[3] * xi)
                       * self.f0(a[4] * (xi - a[5]))
                       + (a[6] + a[7] * xi) * self.f0(a[8] * (a[9] - xi))
                       + (a[10] + a[11] * xi) * self.f0(a[12] * (a[13] - xi))
                       + (a[14] + a[15] * xi) * self.f0(a[16] * (a[17] - xi)))
        return zeta

    def zeta_from_xi_modify(self, xi: float) -> float:
        """
        低密度側に修正を行った状態方程式の log-log 関数
        :param xi: 密度 / [g/cm^3]の常用対数
        :return: 圧力 / [dyn/cm^2]の常用対数
        """
        zeta: float = self.zeta_from_xi(xi)
        if xi < 5.:
            rho: float = 10. ** xi
            p: float = 10. ** zeta
            zeta = math.log10(p + 3.5e+14 * rho)
        return zeta

    def p_from_rho(self, rho: float) -> float:
        """
        :param rho: 密度 / rho_c[g/cm^3]
        :return: 圧力 / [dyn/cm^2]
        """
        xi: float = math.log10(rho)
        zeta: float = self.zeta_from_xi_modify(xi)
        p: float = 10. ** zeta
        return p

    def demo_log_eos_plot(self):
        xi_samples = np.arange(0., 16., 0.1)
        zeta_samples = np.frompyfunc(self.zeta_from_xi_modify, 1, 1)(xi_samples)

        plt.xlabel("log10(rho/[g cm^3])")
        plt.ylabel("log10(p/[dyn cm^2])")
        plt.plot(xi_samples, zeta_samples, marker='+')
        plt.show()
        # plt.savefig("EoS.png")


class AbsIEoS(ABC, EoS):
    @abstractmethod
    def xi_from_zeta(self, zeta: float) -> float:
        pass

    def rho_from_p(self, p: float) -> float:
        """
        :param p: 圧力 / [dyn/cm^2]
        :return: 密度 / [g/cm^3]
        """
        try:
            zeta: float = math.log10(p)
        except ValueError as e:
            print(e, ": p=", p)
            zeta: float = -np.inf
        xi: float = self.xi_from_zeta(zeta)
        rho: float = 10. ** xi
        return rho

    def demo_log_ieos_plot(self):
        zeta_samples: np.ndarray = np.arange(15., 37., 1e-1)
        xi_samples: np.ndarray = np.frompyfunc(self.xi_from_zeta, 1, 1)(zeta_samples)

        plt.xlabel("log10(p/[dyn cm^2])")
        plt.ylabel("log10(rho/[g cm^3])")
        plt.plot(zeta_samples, xi_samples, marker='+')
        plt.show()


class IEoS(AbsIEoS):
    """
    状態方程式の逆関数 rho=rho(p) とそのデモ
    """
    _xi_from_zeta: interp1d

    def make_ieos(self, xi_max):
        xi_samples = np.arange(0., xi_max, 1.e-2)
        zeta_samples = np.frompyfunc(self.zeta_from_xi_modify, 1, 1)(xi_samples)
        self._xi_from_zeta = interpolate.interp1d(zeta_samples, xi_samples,
                                                  kind='linear', fill_value='extrapolate')

    def __init__(self, xi_max: float = 16.):
        self.make_ieos(xi_max)

    def xi_from_zeta(self, zeta: float) -> float:
        return self._xi_from_zeta(zeta)[()]


class IEoS2(AbsIEoS):
    def xi_from_zeta(self, zeta: float) -> float:
        ans: Tuple[float, Any] = optimize.brentq(lambda xi: zeta - self.zeta_from_xi_modify(xi), -0.1, 16.1,
                                                 full_output=True)
        return ans[0]


class AbsPEnthalpyEoS(ABC):
    ieos_ins: Union[IEoS, IEoS2]

    def __init__(self, ieos_ins: Union[IEoS, IEoS2] = IEoS()):
        self.ieos_ins = ieos_ins

    @abstractmethod
    def h_integrand(self, value: float) -> float:
        pass

    @abstractmethod
    def h_from_p(self, p: float) -> float:
        pass

    @abstractmethod
    def p_from_h(self, h: float) -> float:
        pass

    def h_from_rho(self, rho: float) -> float:
        p = self.ieos_ins.p_from_rho(rho)
        h = self.h_from_p(p)
        return h

    def rho_from_h(self, h: float) -> float:
        p = self.p_from_h(h)
        rho = self.ieos_ins.rho_from_p(p)
        return rho

    @abstractmethod
    def demo_h_integrand_plot(self, zeta_max: float):
        pass

    def demo_h_plot(self, xi_max: float):
        """
        擬エンタルピーの log-log プロット
        """
        xi_samples = np.arange(0, xi_max, 0.1)
        zeta_samples = np.frompyfunc(self.ieos_ins.zeta_from_xi_modify, 1, 1)(xi_samples)
        p_samples = np.frompyfunc(lambda zeta: 10 ** zeta, 1, 1)(zeta_samples)
        h_samples = np.frompyfunc(self.h_from_p, 1, 1)(p_samples)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("log10(p/[dyn cm^2])")
        plt.ylabel("psudo-enthalpy")
        plt.plot(p_samples, h_samples, marker="+")
        plt.show()


class PEnthalpyEoS(AbsPEnthalpyEoS):
    """
    擬エンタルピー h(p) を用いて状態方程式を記述
    """
    _h_from_p: interp1d
    _p_from_h: interp1d

    def h_from_p(self, zeta: float) -> float:
        h: float = self._h_from_p(zeta)[()]  # 0次元 ndarray
        return h

    def p_from_h(self, h: float) -> float:
        return self._p_from_h(h)[()]

    def make_eos(self):
        xi_samples = np.arange(0, 16., 0.1)
        zeta_samples = np.frompyfunc(self.ieos_ins.zeta_from_xi_modify, 1, 1)(xi_samples)
        p_samples = np.frompyfunc(lambda zeta: 10 ** zeta, 1, 1)(zeta_samples)
        h_samples = self.make_h_samples(p_samples)

        self._h_from_p = interpolate.interp1d(p_samples, h_samples,
                                              kind='linear', fill_value='extrapolate')
        self._p_from_h = interpolate.interp1d(h_samples, p_samples,
                                              kind='linear', fill_value='extrapolate')

    def __init__(self, ieos_ins=IEoS()):
        super().__init__(ieos_ins)
        self.make_eos()

    def h_integrand(self, p: float) -> float:
        return 1. / (self.ieos_ins.rho_from_p(p) * C_CONST ** 2 + p)

    def make_h_samples(self, p_samples: np.ndarray) -> np.ndarray:
        h_list: List[float] = [0.]

        for index, p in enumerate(p_samples):
            if index == 0:
                continue
            p_pre: float = p_samples[index - 1]
            h_diff: float = integrate.quad(self.h_integrand, p_pre, p)[0]
            h_list.append(h_list[-1] + h_diff)

        return np.array(h_list)

    def demo_h_integrand_plot(self, xi_max: float = 16.):
        """
        擬エンタルピーの被積分関数の log プロット
        """
        xi_samples = np.arange(0, xi_max, 0.1)
        zeta_samples = np.frompyfunc(self.ieos_ins.zeta_from_xi_modify, 1, 1)(xi_samples)
        p_samples = np.frompyfunc(lambda zeta: 10 ** zeta, 1, 1)(zeta_samples)
        h_samples = np.frompyfunc(self.h_integrand, 1, 1)(p_samples)

        plt.xscale('log')
        plt.yscale('log')
        plt.plot(p_samples, h_samples, marker="+")
        plt.show()


class PEnthalpyEoS2(AbsPEnthalpyEoS):
    zeta_min: float
    zeta_max: float

    def __init__(self, ieos_ins):
        super().__init__(ieos_ins)
        self.zeta_min = self.ieos_ins.zeta_from_xi_modify(0.)
        self.zeta_max = self.ieos_ins.zeta_from_xi_modify(16.)

    def h_integrand(self, t: float) -> float:
        x = math.tanh(np.pi / 2 * math.sinh(t))
        zeta = 1 / 2 * ((self.zeta_max - self.zeta_min) * x + self.zeta_max + self.zeta_min)
        dx_dt = np.pi / 2 * math.cosh(t) / math.cosh(np.pi / 2 * math.sinh(t)) ** 2
        dz_dx = (self.zeta_max - self.zeta_min) / 2
        integrand = 1. / (1. + 10. ** (self.ieos_ins.xi_from_zeta(zeta) - zeta) * C_CONST ** 2)
        return integrand * dz_dx * dx_dt

    # TODO: なぜか値が足りない
    def h_from_p(self, p: float) -> float:
        self.zeta_max = math.log10(p)
        # 積分幅は端点の値が最大値に対して1e-16倍以上になる値(丸め誤差程度)に設定
        h: float = integrate.quad(self.h_integrand, -3.5, 3.5, limit=1000)[0]
        return h

    # TODO: 境界の設定が不適切
    def p_from_h(self, h: float) -> float:
        ans: Tuple[float, Any] = optimize.brentq(lambda p: h - self.h_from_p(p), 14.5, 37., full_output=True)
        return ans[0]

    def demo_h_integrand_plot(self, xi_max):
        self.zeta_max = self.ieos_ins.zeta_from_xi_modify(xi_max)
        t_samples = np.arange(-3.5, 3.5, 0.1)
        h_samples = np.frompyfunc(self.h_integrand, 1, 1)(t_samples)

        plt.yscale("log")
        plt.plot(t_samples, h_samples, marker="+")
        plt.show()
