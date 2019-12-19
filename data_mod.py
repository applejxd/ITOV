"""
質量-半径関係を作成するモジュール
"""
from typing import List, Union
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from eos_mod import IEoS, IEoS2, PEnthalpyEoS, PEnthalpyEoS2
from tov_mod import Solution, TOVEqs, TOVEqs2


class AbsMassRadius(ABC):
    """
    質量-半径関係を生成するための抽象基底クラス
    """
    tov_ins: Union[TOVEqs, TOVEqs2]
    results: np.ndarray

    def __init__(self, tov_ins: Union[TOVEqs, TOVEqs2] = TOVEqs2):
        self.tov_ins = tov_ins
        self.results = np.array([[], [], []])

    def reset_mr(self):
        self.results = np.array([[], [], []])

    @abstractmethod
    def extract_radius(self, sol: Solution) -> float:
        """
        解から天体の半径を読み取るメソッド.
        子クラスで実装.

        :param sol: TOV方程式の解
        :return: 天体の半径 [km]
        """
        pass

    def make_mr_data(self, xi_min) -> None:
        """
        質量-半径関係のデータセットを作成するメソッド

        :param xi_min: 最小密度 / [g/cm^3] の常用対数
        """
        xi_c_samples: np.ndarray = np.array([])
        xi_flag: bool = False

        # サンプリング点の幅を調節する
        if xi_min < 14.1:
            xi_tmp = np.arange(xi_min, 14.1, 5.e-2)
            xi_c_samples = np.hstack([xi_c_samples, xi_tmp])
            xi_flag = True

        if xi_min < 14.3:
            if xi_flag:
                xi_tmp = np.arange(14.1, 14.3, 1.e-3)
            else:
                xi_tmp = np.arange(xi_min, 14.3, 1.e-3)
            xi_c_samples = np.hstack([xi_c_samples, xi_tmp])
            xi_flag = True

        if xi_flag:
            xi_tmp = np.arange(14.3, 16., 1.e-2)
        else:
            xi_tmp = np.arange(xi_min, 16., 1.e-2)
        xi_c_samples = np.hstack([xi_c_samples, xi_tmp])

        # 質量-半径関係を作成する
        mass_tmp: List[float] = []
        radius_tmp: List[float] = []
        for xi_init in xi_c_samples:
            sol = self.tov_ins.solve_tov(xi_init)
            mass_tmp.append(sol.y[0][-1])
            radius_tmp.append(self.extract_radius(sol))
        self.results = np.array([xi_c_samples, mass_tmp, radius_tmp])

    def plot_mr(self, xi_min: float) -> None:
        """
        質量-半径関係をプロットするメソッド

        :param xi_min: 密度 / [g/cm^3] の常用対数
        """
        self.reset_mr()
        self.make_mr_data(xi_min)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_\odot$]")
        plt.plot(self.results[2], self.results[1], marker="+")
        if xi_min < 14.5:
            plt.xscale('log')
            plt.xlabel(r"$\log_{10}R$ [km]")
        plt.show()


class MassRadius(AbsMassRadius):
    """
    TOVEqs クラスのインスタンスに対応した実装
    """
    tov_eqs: TOVEqs

    def __init__(self, ieos_ins: Union[IEoS, IEoS2] = IEoS):
        super().__init__(TOVEqs(ieos_ins))

    def extract_radius(self, sol: Solution) -> float:
        return sol.t_events[0][()]


class MassRadius2(AbsMassRadius):
    """
    TOVEqs2 クラスのインスタンスに対応した実装
    """
    tov_eqs: TOVEqs2

    def __init__(self, enthalpy_ins: Union[PEnthalpyEoS, PEnthalpyEoS2] = PEnthalpyEoS):
        super().__init__(TOVEqs2(enthalpy_ins))

    def extract_radius(self, sol: Solution) -> float:
        return sol.y[1][-1]


# TODO: 未完成
# class MR2EoS:
#     tov_eqs: tov_mod.TOVEqs
#
#     def __init__(self):
#         self.tov_eqs = tov_mod.TOVEqs(1)
#
#     def ext_func(self, m, r):
#         rho = self.rho_samples[-1]
#         p = self.p_samples[-1]
#
#         sol = self.inverse_tov(m, r)
#         h = sol[0]
#         x = [sol[1], sol[2]]
#
#         rho_ext = rho + 5/2 * ((3*x[0])/(4*np.pi*x[1]**3) - rho)
#         p_ext = p + (2*np.pi/3 * (rho+p)*(rho+3*p) * x[1]**2 * (1 + 2*np.pi/3 * (4*rho+3*p) * x[1]**2)
#                      + np.pi/3 * (6*rho+11*p) * ((3*x[0])/(4*np.pi*x[1]**3)-rho) * x[1]**2)
#         h_ext = h + (p_ext-p)/(2*(rho+p)) * (3-(rho_ext+p_ext)/(rho+p))
#         return [h_ext, rho_ext, p_ext]
