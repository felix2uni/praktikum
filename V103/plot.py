import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat


xe, yqe, yre = np.genfromtxt('daten_1.txt', unpack=True, skip_header=1)


def f(x, a, b):
    return a*x + b


Elit = 1.2*10**11  # Elastizitaetsmodul Literaturwert

# zylindrischer Stab einseitige Einspannung
L = 0.535  # in m, Stablänge
M = 0.7523  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes
xe *= 10**-2  # von cm auf m
yre *= 10**-6  # von ym auf m

xe = L*xe**2-xe**3/3  # D(x)/y soll gegen diese Funktion aufgetragen werden

params, pcov = curve_fit(f, xe, yre)
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
plt.plot(xe*1e3, yre*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
plt.plot(xe*1e3, f(xe, *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(Lx^2-\frac{x^3}{3}) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/rundeinseitg.pdf')
plt.clf()

# quadratischer Stab einseitige Einspannung
L = 0.535  # in m, Stablänge
M = 0.7523  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes
yqe *= 1e-6  # von mm auf m

params, pcov = curve_fit(f, xe, yqe)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (B**4)/12  # Trägheitsmoment
E = (M*9.81)/(2*I*a)  # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei einseitiger Einspannung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

plt.gcf().subplots_adjust(bottom=0.2)
plt.plot(xe*1e3, yqe*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"

plt.plot(xe*1e3, f(xe, *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(Lx^2-\frac{x^3}{3}) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/quadeinseitg.pdf')
plt.clf()


xzr, yqzl, yqzr, _, _ = np.genfromtxt(
    'daten_2.txt', unpack=True, skip_header=1)

xzl = xzr


# quadratischer Stab zweiseitig Einspannung links
L = 0.559  # in m, Stablänge
M = 1.0517  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes
xzl *= 1e-2  # von mm auf m
yqzl *= 1e-6  # von mm auf m

# D(x)/y soll gegen diese Funktion aufgetragen werden
xzl = 3*(L**2)*xzl-4*xzl**3

params, pcov = curve_fit(f, xzl, yqzl)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (B**4)/12  # Trägheitsmoment
E = (M*9.81)/(48*I*a)  # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei zweiseitiger Einspannung links:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

plt.plot(xzl*1e3, yqzl*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
plt.plot(xzl*1e3, f(xzl, *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(3L^2x-4x^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/quadzweiseitg_links.pdf')
plt.clf()


# quadratischer Stab zweiseitig Einspannung rechts
L = 0.559  # in m, Stablänge
M = 1.0517  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes
xzr *= 1e-2  # von mm auf m
yqzr *= 1e-6  # von mm auf m

# D(x)/y soll gegen diese Funktion aufgetragen werden
xzr = 4*xzr**3-12*L*xzr**2+9*(L**2)*xzr-L**3

params, pcov = curve_fit(f, xzr, yqzr)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (B**4)/12  # Trägheitsmoment
E = (M*9.81)/(48*I*a)  # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei zweiseitiger Einspannung rechts:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

plt.plot(xzr*1e3, yqzr*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
plt.plot(xzr*1e3, f(xzr, *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(4x^3-12Lx^2+9L^2x-L^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/quadzweiseitg_rechts.pdf')
plt.clf()


xzr, _, _, yqzl, yqzr = np.genfromtxt(
    'daten_2.txt', unpack=True, skip_header=1)

xzl = xzr


# runder Stab zweiseitig Einspannung links
L = 0.559  # in m, Stablänge
M = 1.0517  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes
xzl *= 1e-2  # von mm auf m
yqzl *= 1e-6  # von mm auf m

# D(x)/y soll gegen diese Funktion aufgetragen werden
xzl = 3*(L**2)*xzl-4*xzl**3

params, pcov = curve_fit(f, xzl, yqzl)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (np.pi/4)*(B/2)**4  # Trägheitsmoment
E = (M*9.81)/(2*I*a)  # Elastizitaetsmodul

print('Parameter fuer runder Stab bei zweiseitiger Einspannung links:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

plt.plot(xzl*1e3, yqzl*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
plt.plot(xzl*1e3, f(xzl, *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(3L^2x-4x^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/rundzweiseitg_links.pdf')
plt.clf()


# runder Stab zweiseitig Einspannung rechts
L = 0.559  # in m, Stablänge
M = 1.0517  # in kg, angehängte Masse
B = 0.01  # in m, Breite des Stabes
xzr *= 1e-2  # von mm auf m
yqzr *= 1e-6  # von mm auf m

# D(x)/y soll gegen diese Funktion aufgetragen werden
xzr = 4*xzr**3-12*L*xzr**2+9*(L**2)*xzr-L**3

params, pcov = curve_fit(f, xzr, yqzr)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (np.pi/4)*(B/2)**4  # Trägheitsmoment
E = (M*9.81)/(2*I*a)  # Elastizitaetsmodul

print('Parameter fuer runden Stab bei zweiseitiger Einspannung rechts:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(str(((Elit - E) / E) * 100) + "%")

plt.plot(xzr*1e3, yqzr*1e3, 'kx', label='Messwerte')
fitLable = r"Fit: $f(x)="
fitLable += np.format_float_scientific(params[0], unique=False, precision=3)
fitLable += r"\times x+"
fitLable += str(np.format_float_scientific(
    params[1], unique=False, precision=3))
fitLable += r"$"
plt.plot(xzr*1e3, f(xzr, *params)*1e3, 'r-', label=fitLable)
plt.xlabel(r'$(4x^3-12Lx^2+9L^2x-L^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('build/rundzweiseitg_rechts.pdf')
plt.clf()
