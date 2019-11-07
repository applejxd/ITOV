from eos_mod import EoS, IEoS, IEoS2, PEnthalpyEoS, PEnthalpyEoS2
import tov_mod
import data_mod

if __name__ == "__main__":
    # 状態方程式のデモ
    eos_ins = EoS()
    eos_ins.demo_log_eos_plot()

    # # 状態方程式の逆関数のデモ
    # ieos_ins = IEoS()
    # ieos_ins.demo_log_ieos_plot()

    # # 擬エンタルピー関連のデモ
    # enthalpy_ins = PEnthalpyEoS(IEoS())
    # enthalpy_ins.demo_h_integrand_plot(16.)
    # enthalpy_ins.demo_h_plot(16.)

    # # TOV 方程式のデモ
    # tov2_demo = tov_mod.TOVEqs2(PEnthalpyEoS(IEoS()))
    # tov2_demo.tov_demo(15.5)

    # # M-R 関係の生成
    # mr2 = data_mod.MassRadius2(PEnthalpyEoS(IEoS()))
    # mr2.plot_mr(14.6)

    # # TOV逆写像のデモ
    # inverse_tov_ins = tov_mod.InverseTOV(PEnthalpyEoS(IEoS()))
    # inverse_tov_ins.inverse_tov_demo()

    print("finished")
