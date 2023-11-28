import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat

Elit = 1.2 * 10**11
L = 0.55  # länge in m
M = 1  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes


def f(x, c, b):
    return c * x + b


eckig_zweiseitig = pd.read_csv("data/eckig_zweiseitig.csv", delimiter=";")

eckig_zweiseitig["xe"] = eckig_zweiseitig["xe"] * 10**-2  # von cm nach m

eckig_zweiseitig["yle"] = eckig_zweiseitig["yle"] * 10**-3  # mm nach m
eckig_zweiseitig["yre"] = eckig_zweiseitig["yre"] * 10**-3  # mm nach m

eckig_zweiseitig["xle"] = (
    3 * (L**2) * eckig_zweiseitig["xe"] - 4 * eckig_zweiseitig["xe"] ** 3
)

eckig_zweiseitig["xre"] = (
    4 * eckig_zweiseitig["xe"] ** 3
    - 12 * L * eckig_zweiseitig["xe"] ** 2
    + 9 * (L**2) * eckig_zweiseitig["xe"]
    - L**3
)

params, pcov = curve_fit(f, eckig_zweiseitig["xle"], eckig_zweiseitig["yle"])
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (B**4)/12  # Trägheitsmoment
E = (M * 9.81) / (48 * I * a)  # Elastizitaetsmodul


print("Parameter fuer runder Stab bei zweiseitiger Einspannung links:")
print("a =", params[0], "±", errors[0])
print("b =", params[1], "±", errors[1])
print("Elastizitaetsmodul:")
print(E)
print(f'I = {I}')
print("Abweichung:")
print(str(((Elit - E) / E) * 100) + "%")


plt.plot(
    eckig_zweiseitig["xle"] * 1e3,
    eckig_zweiseitig["yle"] * 1e3,
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
    eckig_zweiseitig["xle"] * 1e3,
    f(eckig_zweiseitig["xle"], *params) * 1e3,
    "r-",
    label=fitLable,
)
plt.xlabel(r"$(3L^2x-4x^3) \:/\: (10^{-3} \: m^{3})$")
plt.ylabel(r"$D(x) \:/\: (10^{-3} \: m)$")
plt.legend(loc="best")

plt.savefig("build/eckig_zweiseitig_links_plot.pdf")
plt.cla()


params, pcov = curve_fit(f, eckig_zweiseitig["xre"], eckig_zweiseitig["yre"])
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (B**4)/12  # Trägheitsmoment
E = (M*9.81)/(48*I*a)  # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei zweiseitiger Einspannung rechts:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print(f'I = {I}')
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

plt.plot(eckig_zweiseitig["xre"]*1e3, eckig_zweiseitig["yre"]*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
plt.plot(eckig_zweiseitig["xre"]*1e3, f(eckig_zweiseitig["xre"], *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(4x^3-12Lx^2+9L^2x-L^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/eckig_zweiseitig_rechts_plot.pdf')