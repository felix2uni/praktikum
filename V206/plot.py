import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat

data = pd.read_csv("./assets/temperature.csv", delimiter=";")

# to kelvin
data["Q_1"] = data["Q_1"] + 273.15
data["Q_2"] = data["Q_2"] + 273.15

# pressure add 1
data["p_a"] = data["p_a"] + 1
data["p_b"] = data["p_b"] + 1

data["t"] = data["t"] * 60

cold_C = data["Q_1"][0]
hot_C = data["Q_2"][0]

c_w = 4182
m_l = 3

c_k = 750
m_k = 1


plt.plot(
    data["t"],
    data["Q_1"],
    "X",
    label=r"Reservoir 1",
)
plt.plot(
    data["t"],
    data["Q_2"],
    "X",
    label=r"Reservoir 2",
)

plt.xlabel("t/s")
plt.ylabel("T/K")
plt.legend(loc="best")
plt.grid()
plt.savefig("build/plot_0.pdf")


def f_1(t, A, B, C):
    return A * t**2 + B * t + C


def d_d_1(t, A, B):
    return 2 * A * t + B


def gueteziffer_1(t, A, B, N):
    return (1 / N) * (m_l * c_w + m_k * c_k) * d_d_1(t, A, B)


def gueteziffer_ideal(T_1, T_2):
    return T_1 / (T_1 - T_2)


def f_2(t, A, B, alpha, C):
    return (A * t**alpha) / (1 + B * t**alpha) + C


bounds_2 = [
    [-np.inf, -np.inf, 1],
    [np.inf, np.inf, 2],
]


def cold_f_1(t, A, B):
    return f_1(t, A, B, cold_C)


def cold_f_2(t, A, B, alpha):
    return f_2(t, A, B, alpha, cold_C)


cold_approx_1, cold_cov_1 = curve_fit(cold_f_1, data["t"], data["Q_1"])
cold_approx_2, cold_cov_2 = curve_fit(cold_f_2, data["t"], data["Q_1"], bounds=bounds_2)

cold_uncertainties_1 = np.sqrt(np.diag(cold_cov_1))
cold_uncertainties_2 = np.sqrt(np.diag(cold_cov_2))

cold_A_1 = ufloat(cold_approx_1[0], cold_uncertainties_1[0])
cold_B_1 = ufloat(cold_approx_1[1], cold_uncertainties_1[1])


cold_A_2 = ufloat(cold_approx_2[0], cold_uncertainties_2[0])
cold_B_2 = ufloat(cold_approx_2[1], cold_uncertainties_2[1])
cold_alpha_2 = ufloat(cold_approx_2[2], cold_uncertainties_2[2])


print("Fit cold:")
print("\tfunction 1:")
print(f"\t\t A = {cold_A_1}")
print(f"\t\t B = {cold_B_1}")
print(f"\t\t C = {cold_C}")
print(f"\t\t dT({data['t'][2]})/dt = {d_d_1(data['Q_2'][2], cold_A_1, cold_B_1)}")
print(f"\t\t dT({data['t'][4]})/dt = {d_d_1(data['Q_2'][4], cold_A_1, cold_B_1)}")
print(f"\t\t dT({data['t'][6]})/dt = {d_d_1(data['Q_2'][6], cold_A_1, cold_B_1)}")
print(f"\t\t dT({data['t'][8]})/dt = {d_d_1(data['Q_2'][8], cold_A_1, cold_B_1)}")
print(
    f"\t\t gueteziffer({data['t'][2]}) = {gueteziffer_1(data['t'][2], cold_A_1, cold_B_1, data['A'][2])}"
)
print(
    f"\t\t gueteziffer_ideal({data['t'][2]}) = {gueteziffer_ideal(data['Q_1'][2], data['Q_2'][2])}"
)
print(
    f"\t\t gueteziffer({data['t'][4]}) = {gueteziffer_1(data['t'][4], cold_A_1, cold_B_1, data['A'][4])}"
)
print(
    f"\t\t gueteziffer_ideal({data['t'][4]}) = {gueteziffer_ideal(data['Q_1'][4], data['Q_2'][4])}"
)
print(
    f"\t\t gueteziffer({data['t'][6]}) = {gueteziffer_1(data['t'][6], cold_A_1, cold_B_1, data['A'][6])}"
)
print(
    f"\t\t gueteziffer_ideal({data['t'][4]}) = {gueteziffer_ideal(data['Q_1'][4], data['Q_2'][4])}"
)
print(
    f"\t\t gueteziffer({data['t'][8]}) = {gueteziffer_1(data['t'][8], cold_A_1, cold_B_1, data['A'][8])}"
)
print(
    f"\t\t gueteziffer_ideal({data['t'][4]}) = {gueteziffer_ideal(data['Q_1'][4], data['Q_2'][4])}"
)
print("")
print("\tfunction 2:")
print(f"\t\t A = {cold_A_2}")
print(f"\t\t B = {cold_B_2}")
print(f"\t\t alpha = {cold_alpha_2}")
print(f"\t\t C = {cold_C}")


def hot_f_1(t, A, B):
    return f_1(t, A, B, hot_C)


def hot_f_2(t, A, B, alpha):
    return f_2(t, A, B, alpha, hot_C)


R = 8.314


def L(m):
    return -m * R


def massendurchsatz(T, m):
    return d_d_1(T, hot_A_1, hot_B_1) / L(m)


hot_approx_1, hot_cov_1 = curve_fit(hot_f_1, data["t"], data["Q_2"])
hot_approx_2, hot_cov_2 = curve_fit(hot_f_2, data["t"], data["Q_2"], bounds=bounds_2)

hot_uncertainties_1 = np.sqrt(np.diag(hot_cov_1))
hot_uncertainties_2 = np.sqrt(np.diag(hot_cov_2))

hot_A_1 = ufloat(hot_approx_1[0], hot_uncertainties_1[0])
hot_B_1 = ufloat(hot_approx_1[1], hot_uncertainties_1[1])


hot_A_2 = ufloat(hot_approx_2[0], hot_uncertainties_2[0])
hot_B_2 = ufloat(hot_approx_2[1], hot_uncertainties_2[1])
hot_alpha_2 = ufloat(hot_approx_2[2], hot_uncertainties_2[2])


print("")
print("")
print("")
print("Fit hot:")
print("\tfunction 1:")
print(f"\t\t A = {hot_A_1}")
print(f"\t\t B = {hot_B_1}")
print(f"\t\t C = {hot_C}")
print(f"\t\t dT({data['t'][2]})/dt = {d_d_1(data['Q_1'][2], hot_A_1, hot_B_1)}")
print(f"\t\t dT({data['t'][4]})/dt = {d_d_1(data['Q_1'][4], hot_A_1, hot_B_1)}")
print(f"\t\t dT({data['t'][6]})/dt = {d_d_1(data['Q_1'][6], hot_A_1, hot_B_1)}")
print(f"\t\t dT({data['t'][8]})/dt = {d_d_1(data['Q_1'][8], hot_A_1, hot_B_1)}")
print(
    f"\t\t gueteziffer({data['t'][2]}) = {gueteziffer_1(data['t'][2], hot_A_1, hot_B_1, data['A'][2])}"
)
print(
    f"\t\t gueteziffer({data['t'][4]}) = {gueteziffer_1(data['t'][4], hot_A_1, hot_B_1, data['A'][4])}"
)
print(
    f"\t\t gueteziffer({data['t'][6]}) = {gueteziffer_1(data['t'][6], hot_A_1, hot_B_1, data['A'][6])}"
)
print(
    f"\t\t gueteziffer({data['t'][8]}) = {gueteziffer_1(data['t'][8], hot_A_1, hot_B_1, data['A'][8])}"
)

print("")
print("\tfunction 2:")
print(f"\t\t A = {hot_A_2}")
print(f"\t\t B = {hot_B_2}")
print(f"\t\t alpha = {hot_alpha_2}")
print(f"\t\t C = {hot_C}")


plt.plot(
    data["t"],
    cold_f_1(data["t"], cold_approx_1[0], cold_approx_1[1]),
    "-",
    label=r"Ausgleichsgerade 1 für Reservoir 1",
    # color='red'
)

plt.plot(
    data["t"],
    hot_f_1(data["t"], hot_approx_1[0], hot_approx_1[1]),
    "-",
    label=r"Ausgleichsgerade 1 für Reservoir 2",
    # color='lightblue'
)

plt.xlabel("t/s")
plt.ylabel("T/K")
plt.legend(loc="best")
plt.grid()
plt.savefig("build/plot_1.pdf")

plt.cla()

plt.plot(
    data["t"],
    data["Q_1"],
    "X",
    label=r"Reservoir 1",
)
plt.plot(
    data["t"],
    data["Q_2"],
    "X",
    label=r"Reservoir 2",
)

plt.plot(
    data["t"],
    cold_f_2(data["t"], cold_approx_2[0], cold_approx_2[1], cold_approx_2[2]),
    "-",
    label=r"Ausgleichsgerade 2 für Reservoir 1",
    # color='limegreen'
)

plt.plot(
    data["t"],
    hot_f_2(data["t"], hot_approx_2[0], hot_approx_2[1], hot_approx_2[2]),
    "-",
    label=r"Ausgleichsgerade 2 für Reservoir 2",
    # color='blue'
)

plt.xlabel("t/s")
plt.ylabel("T/K")
plt.legend(loc="best")
plt.grid()
plt.savefig("build/plot_2.pdf")

plt.cla()


# plt.plot(
#     data["t"],
#     data["p_a"],
#     "X",
#     label=r"Reservoir 1",
# )

# plt.plot(
#     data["t"],
#     data["p_b"],
#     "X",
#     label=r"Reservoir 1",
# )
# plt.cla()


def verdampfung_waerme(T, p, p_0):
    return R * T * np.log(p_0 / p)


def gerade(x, a, b):
    return a * x + b


plt.plot(
    1 / data["Q_2"],
    np.log(data["p_a"] / data["p_a"][0]),
    "X",
    label=r"Reservoir 1",
)

params, cov = curve_fit(gerade, 1 / data["Q_2"], np.log(data["p_a"] / data["p_a"][0]))

plt.plot(
    1 / data["Q_2"],
    gerade(1 / data["Q_2"], params[0], params[1]),
    "-",
    label=r"Reservoir 1",
)
print(params)
# plt.plot(
#     1 / data["Q_2"],
#     gerade( 1 / data["Q_2"], params[0], params[1]),
#     'X',
#     'Ausgleichsgerade'
# )

plt.xlabel("$\\frac{{1}}{{T}}$ in $K^{-1}$")
plt.ylabel("$\ln(\\frac{{p_a}}{{p_0}})$")
# plt.legend(loc="best")
plt.grid()
plt.savefig("build/plot_3.pdf")

print(
    data[["p_a", "Q_2"]].to_latex(
        header=["p_a", "Q_2"],
        index=False,
        formatters=["{:.2f}".format, "{:.2f}".format],
    )
)


print(
    f"\t\t massendurchsatz({data['t'][2]}) = {massendurchsatz(data['t'][2], params[0])}"
)
print(
    f"\t\t massendurchsatz({data['t'][4]}) = {massendurchsatz(data['t'][4], params[0])}"
)
print(
    f"\t\t massendurchsatz({data['t'][6]}) = {massendurchsatz(data['t'][6], params[0])}"
)
print(
    f"\t\t massendurchsatz({data['t'][8]}) = {massendurchsatz(data['t'][8], params[0])}"
)

roh_0 = 5.5
T_0 = 273.15


def dichte_gas(t):
    p_a = data["p_a"][t / 60]
    T_2 = data["Q_2"][t / 60]
    return (roh_0 * T_0 * p_a) / T_2


print(dichte_gas(120))
print(dichte_gas(240))
print(dichte_gas(360))
print(dichte_gas(480))


ce = 1.14


def kompressor_leistung(t):
    p_a = data["p_a"][t / 60]
    p_b = data["p_b"][t / 60]
    return (
        (1 / (ce - 1))
        * (p_b * np.sqrt(p_a / p_b) * 1 / dichte_gas(t))
        * massendurchsatz(data["t"][t / 60], params[0])
        * 120.91
    )


    
print(kompressor_leistung(120))
print(kompressor_leistung(240))
print(kompressor_leistung(360))
print(kompressor_leistung(480))
