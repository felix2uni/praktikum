import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat

Elit = 1.2*10**11
L = 0.52 # länge in m
M = 0.5  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes




def f(x, c, b):
   return c * x + b


rund_einseitig = pd.read_csv("data/rund_einseitig.csv", delimiter=";")

rund_einseitig['xe'] = rund_einseitig['xe'] * 10**-2 # von cm nach m

rund_einseitig['ye'] = rund_einseitig['ye'] * 10**-3 # mm nach m

rund_einseitig['xe'] = L*rund_einseitig['xe']**2-rund_einseitig['xe']**3/3  # D(x)/y soll gegen diese Funktion aufgetragen werden


params, pcov = curve_fit(f, rund_einseitig['xe'], rund_einseitig['ye'])
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (np.pi/4)*(B/2)**4  # Trägheitsmoment
E = (M*9.81)/(2*I*a)  # Elastizitaetsmodul


print('Parameter fuer zylindrischen Stab bei einseitiger Einspannung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

 

plt.gcf().subplots_adjust(bottom=0.2)
plt.plot(rund_einseitig['xe']*1e3, rund_einseitig['ye']*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
# plt.axline(xy1=(0, params[1]), slope=params[0], label=fitLable, color="red")
plt.plot(rund_einseitig['xe']*1e3, f(rund_einseitig['xe'], *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(Lx^2-\frac{x^3}{3}) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/rund_einseitig_plot.pdf')

