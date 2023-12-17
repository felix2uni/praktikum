import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat

Elit = 1.2 * 10**11
L = 0.55  # länge in m
M = 1  # in kg, angehängte Masse
B = 0.005  # in m, radius des Stabes


def f(x, c, b):
    return c * x + b


rund_zweiseitig = pd.read_csv("data/rund_zweiseitig.csv", delimiter=";")

rund_zweiseitig["xe"] = rund_zweiseitig["xe"] * 10**-2  # von cm nach m

rund_zweiseitig["yle"] = rund_zweiseitig["yle"] * 10**-3  # mm nach m
rund_zweiseitig["yre"] = rund_zweiseitig["yre"] * 10**-3  # mm nach m

rund_zweiseitig["xle"] = (
    3 * (L**2) * rund_zweiseitig["xe"] - 4 * rund_zweiseitig["xe"] ** 3
)

rund_zweiseitig["xre"] = (
    4 * rund_zweiseitig["xe"] ** 3
    - 12 * L * rund_zweiseitig["xe"] ** 2
    + 9 * (L**2) * rund_zweiseitig["xe"]
    - L**3
)

params, pcov = curve_fit(f, rund_zweiseitig["xle"], rund_zweiseitig["yle"])
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (np.pi * B**4) / 4  # Trägheitsmoment
E = (M*9.81)/(48*I*a)  # Elastizitaetsmodul


print("Parameter fuer runder Stab bei zweiseitiger Einspannung links:")
print("a =", params[0], "±", errors[0])
print("b =", params[1], "±", errors[1])
print("Elastizitaetsmodul:")
print(E)
print(f"I = {I}")
print("Abweichung:")
print(str(((Elit - E) / E) * 100) + "%")


plt.plot(
    rund_zweiseitig["xle"] * 1e3,
    rund_zweiseitig["yle"] * 1e3,
    "kx",
    label="Messwerte",
)
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(params[1], unique=False, precision=3))
fitLable += r"$"
# plt.axline(xy1=(0, params[1]), slope=params[0], label=fitLable, color="red")
plt.plot(
    rund_zweiseitig["xle"] * 1e3,
    f(rund_zweiseitig["xle"], *params) * 1e3,
    "r-",
    label=fitLable,
)
plt.xlabel(r"$(3L^2x-4x^3) \:/\: (10^{-3} \: m^{3})$")
plt.ylabel(r"$D(x) \:/\: (10^{-3} \: m)$")
plt.legend(loc="best")

plt.savefig("build/rund_zweiseitig_links_plot.pdf")
plt.cla()


params, pcov = curve_fit(f, rund_zweiseitig["xre"], rund_zweiseitig["yre"])
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (np.pi * B**4) / 4   # Trägheitsmoment
E = (M*9.81)/(48*I*a)  # Elastizitaetsmodul

print('Parameter fuer runden Stab bei zweiseitiger Einspannung rechts:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print(f"I = {I}")
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

plt.plot(rund_zweiseitig["xre"]*1e3, rund_zweiseitig["yre"]*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
plt.plot(rund_zweiseitig["xre"]*1e3, f(rund_zweiseitig["xre"], *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(4x^3-12Lx^2+9L^2x-L^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/rund_zweiseitig_rechts_plot.pdf')