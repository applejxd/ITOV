import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from eos_mod import IEoS, IEoS2, PEnthalpyEoS, PEnthalpyEoS2

Solution = Tuple[np.ndarray, np.ndarray,
                 Optional[integrate.OdeSolution], Optional[Iterable[np.ndarray]],
                 int, int, int, int, str, bool]

G_CONST: float = 6.67430e-8  # cm^3/(g*s^2)
C_CONST: float = 2.99792458e+10  # cm/s
M_SOLAR: float = 1.9891e+33  # g

R_G: float = G_CONST * M_SOLAR / C_CONST ** 2

RHO_SOLAR: float = M_SOLAR / R_G ** 3
P_SOLAR: float = RHO_SOLAR * C_CONST ** 2

XI_CONST: float = math.log10(RHO_SOLAR)
ZETA_CONST: float = math.log10(P_SOLAR)


class AbsTOVEqs(ABC):
    @abstractmethod
    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        pass

    @abstractmethod
    def solve_tov(self, xi_c: float) -> Solution:
        pass

    @abstractmethod
    def tov_demo(self, xi_c: float):
        pass


class TOVEqs(AbsTOVEqs):
    """
    TOV 方程式とそのデモ
    """
    ieos_ins: Union[IEoS, IEoS2]
    p_c: float

    def __init__(self, ieos_ins: Union[IEoS, IEoS2] = IEoS):
        self.ieos_ins = ieos_ins
        self.p_c = self.ieos_ins.p_from_rho(10. ** 16.)

    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        """
        TOV 方程式
        :param t: 動径座標
        :param y: y = (m, p)
        :return: x の導関数
        """
        rho = self.ieos_ins.rho_from_p(y[1] * self.p_c) * C_CONST ** 2 / self.p_c

        dm_dr: float = 4. * np.pi * t ** 2. * rho
        dp_dr: float = -(rho + y[1]) * (y[0] + 4. * np.pi * t ** 3. * y[1]) / (t * (t - 2. * y[0]))
        return [dm_dr, dp_dr]

    def solve_tov(self, xi_c: float) -> Solution:
        """
        TOV方程式を初期値問題として解く
        :param xi_c: 中心密度[g/cm^3]の常用対数
        :return: 解に関連する bunch object
        """
        self.p_c = self.ieos_ins.p_from_rho(10. ** xi_c)

        r_init: float = 1.e-5
        r_span: List[float] = [r_init, 50.]

        rho_init = 10. ** xi_c * C_CONST ** 2. / self.p_c
        p_init: float = 1.

        y0: List[float] = [4. / 3. * np.pi * r_init ** 3. * rho_init,
                           p_init - 2. / 3. * np.pi * (rho_init + p_init) * (rho_init + 3. * p_init) * r_init ** 2.]

        def is_surface(r: float, x: List[float]) -> float:
            """
            星表面(付近)で 0 になる関数
            :param r: 半径
            :param x: x=(m, p)
            :return: 圧力 / (rho_c * C^2 [dyn/cm^2]) と cut-off の差
            """
            return x[1] - 1 / self.p_c

        is_surface.terminal = True  # 条件.が0のとき終了する
        is_surface.direction = -1  # 正から負にクロス

        m_ratio: float = C_CONST ** 4. / math.sqrt(G_CONST ** 3. * self.p_c) / M_SOLAR
        r_ratio: float = C_CONST ** 2. / math.sqrt(G_CONST * self.p_c) * 1.e-5

        # TODO: 刻み幅を最適化したい
        # TODO: 負の圧力を削除
        sol: Solution = integrate.solve_ivp(self.dy_dt, r_span, y0, events=is_surface, dense_output=True,
                                            max_step=1e-3 / r_ratio, method="Radau")

        sol.y[0] *= m_ratio  # solar_mass
        sol.t *= r_ratio  # km
        sol.t_events = list(map(lambda t: t*r_ratio, sol.t_events))
        return sol

    def tov_demo(self, xi_c: float):
        """
        TOV 方程式を解いて圧力分布を表示するデモ
        :param xi_c: 密度[g/cm^3] の常用対数
        """
        plt.xlabel("radius [km]")
        # plt.yscale("log")
        sol = self.solve_tov(xi_c)
        plt.plot(sol.t, sol.y[1], marker="+")
        plt.show()


class TOVEqs2(AbsTOVEqs):
    """
    擬エンタルピー関数を引数とした TOV 方程式とそのデモ
    """
    enthalpy_ins: Union[PEnthalpyEoS, PEnthalpyEoS2]
    xi_c: float

    def __init__(self, enthalpy_ins: Union[PEnthalpyEoS, PEnthalpyEoS2] = PEnthalpyEoS):
        self.enthalpy_ins = enthalpy_ins
        self.xi_c = 16.

    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        """
        TOV 方程式
        :param t: 擬エンタルピー
        :param y: y = (m, r)
        :return: y の導関数
        """
        rho = self.enthalpy_ins.rho_from_h(t) / 10 ** self.xi_c
        p = self.enthalpy_ins.p_from_h(t) / (10 ** self.xi_c * C_CONST ** 2)

        dr_dh: float = -y[1] * (y[1] - 2 * y[0]) / (y[0] + 4 * np.pi * y[1] ** 3 * p)
        dm_dh: float = 4 * np.pi * rho * y[1] ** 2 * dr_dh
        return [dm_dh, dr_dh]

    def solve_tov(self, xi_c: float) -> Solution:
        """
        擬エンタルピー関数を引数とした TOV 方程式を解く
        :param xi_c: 中心密度[g/cm^3]の常用対数
        :return: 解に関連する bunch object
        """
        self.xi_c = xi_c

        h_c: float = self.enthalpy_ins.h_from_rho(10**self.xi_c)
        h_diff: float = h_c * 1.e-2
        h_span: List[float] = [h_c - h_diff, 0.]

        rho_c = 1.
        p_c: float = self.enthalpy_ins.p_from_h(h_c) / (10 ** xi_c * C_CONST ** 2)

        # 星中心付近で初期条件を構成
        r_init: float = math.sqrt(3 * h_diff / (2 * np.pi) / (rho_c + 3 * p_c))
        m_init: float = 4/3 * np.pi * rho_c * r_init ** 3
        y0: List[float] = [m_init, r_init]

        sol: Solution = integrate.solve_ivp(self.dy_dt, h_span, y0, first_step=h_diff,
                                            max_step=(h_span[0] - h_span[1]) / 50)

        m_ratio: float = C_CONST ** 3 / math.sqrt(G_CONST ** 3 * 10 ** xi_c) / M_SOLAR
        r_ratio: float = C_CONST / math.sqrt(G_CONST * 10 ** xi_c) * 1e-5
        sol.y[0] *= m_ratio  # solar_mass
        sol.y[1] *= r_ratio  # km

        return sol

    def tov_demo(self, xi_c: float):
        """
        擬エンタルピー関数を引数としたTOV方程式のデモ
        :param xi_c: 中心密度[g/cm^3]の常用対数
        """
        sol: Solution = self.solve_tov(xi_c)

        plt.xlabel(r"$h$")
        plt.plot(sol.t, sol.y[0], label=r"$m$ [$M_\odot$]")
        plt.plot(sol.t, sol.y[1], label="$r$ [km]")
        plt.legend()
        plt.show()


class InverseTOV:
    enthalpy_ins: Union[PEnthalpyEoS, PEnthalpyEoS2]

    def __init__(self, enthalpy_ins: Union[PEnthalpyEoS, PEnthalpyEoS2] = PEnthalpyEoS):
        self.enthalpy_ins = enthalpy_ins

    def dy_dt(self, t: float, y: List[float]) -> List[float]:
        """
        TOV 方程式
        :param t: 擬エンタルピー
        :param y: y = (m, r)
        :return: y の導関数
        """
        rho = self.enthalpy_ins.rho_from_h(t) / RHO_SOLAR
        p = self.enthalpy_ins.p_from_h(t) / P_SOLAR

        dr_dh: float = -y[1] * (y[1] - 2 * y[0]) / (y[0] + 4 * np.pi * y[1] ** 3 * p)
        dm_dh: float = 4 * np.pi * rho * y[1] ** 2 * dr_dh
        return [dm_dh, dr_dh]

    def solve_tov(self, mass: float, radius: float, h_i: float) -> Solution:
        radius *= 1.e+5 / R_G

        t_span: List[float] = [0., h_i]
        y0: List[float] = [mass, radius]

        sol: Solution = integrate.solve_ivp(self.dy_dt, t_span, y0, max_step=(t_span[1] - t_span[0]) / 50)

        sol.y[1] *= R_G * 1.e-5  # km

        return sol

    def inverse_tov_demo(self):
        plt.xlabel("pseudo-enthalpy")
        sol = self.solve_tov(2., 10., 0.7)
        plt.plot(sol.t, sol.y[0], label="mass [solar_mass]")
        plt.plot(sol.t, sol.y[1], label="radius [km]")
        plt.legend()
        plt.show()
