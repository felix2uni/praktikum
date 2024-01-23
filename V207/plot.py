import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat


density_water = 998.2

kleine_kugel_device_konstant = 0.07640 * (10**-6)


def density_of_sphere(radius, mass):
    volume = (4 / 3) * np.pi * (radius**3)
    return mass / volume


kleine_kugel_diameter = 15.61 * 10**-3  # m
kleine_kugel_radius = kleine_kugel_diameter / 2  # m
kleine_kugel_mass = 4.4531 * 10**-3  ## g
kleine_kugel_density = density_of_sphere(kleine_kugel_radius, kleine_kugel_mass)


print(f"Dichte kleine Kugel: {kleine_kugel_density}")


up_char = "↑"
down_char = "↓"

kleine_kugel = pd.read_csv("./assets/kleine_kugel.csv")

kleine_kugel_up = kleine_kugel[kleine_kugel["d"] == "u"]["t"]
kleine_kugel_down = kleine_kugel[kleine_kugel["d"] == "d"]["t"]

kleine_kugel_mean = kleine_kugel["t"].mean()
kleine_kugel_std = kleine_kugel["t"].std()
kleine_kugel_up_mean = kleine_kugel_up.mean()
kleine_kugel_down_mean = kleine_kugel_down.mean()
kleine_kugel_viscosity = (
    (kleine_kugel_density - density_water)
    * kleine_kugel_device_konstant
    * kleine_kugel_mean
)

with open("build/kleine_kugel.tex", "w") as f:
    f.write(
        kleine_kugel.to_latex(
            header=["t/s", "Richtung"],
            index=False,
            formatters=["{:.2f}".format, lambda e: up_char if e == "u" else down_char],
        )
    )

print(f"Mean kleine Kugel: {kleine_kugel_mean}")
print(f"Std kleine Kugel: {kleine_kugel_std}")
print(f"Mean kleine Kugel up: {kleine_kugel_up_mean}")
print(f"Mean kleine Kugel down: {kleine_kugel_down_mean}")

print(f"Viskosität kleine Kugel: {kleine_kugel_viscosity}")


grosse_kugel_diameter = 15.78 * 10**-3  # m
grosse_kugel_radius = grosse_kugel_diameter / 2  # m
grosse_kugel_mass = 4.9528 * 10**-3  ## g
grosse_kugel_density = density_of_sphere(grosse_kugel_radius, grosse_kugel_mass)

print(
    f"Dichte grosse Kugel: {density_of_sphere(grosse_kugel_radius, grosse_kugel_mass)}"
)


grosse_kugel = pd.read_csv("./assets/grosse_kugel.csv")

grosse_kugel_up = grosse_kugel[grosse_kugel["d"] == "u"]["t"]
grosse_kugel_down = grosse_kugel[grosse_kugel["d"] == "d"]["t"]

grosse_kugel_mean = grosse_kugel["t"].mean()
grosse_kugel_std = grosse_kugel["t"].std()
grosse_kugel_up_mean = grosse_kugel_up.mean()
grosse_kugel_down_mean = grosse_kugel_down.mean()

grosse_kugel_way_length = 0.05

with open("build/grosse_kugel.tex", "w") as f:
    f.write(
        grosse_kugel.to_latex(
            header=["t/s", "Richtung"],
            index=False,
            formatters=["{:.2f}".format, lambda e: up_char if e == "u" else down_char],
        )
    )

print(f"Mean grosse Kugel: {grosse_kugel_mean}")
print(f"Std grosse Kugel: {grosse_kugel_std}")
print(f"Mean grosse Kugel up: {grosse_kugel_up_mean}")
print(f"Mean grosse Kugel up: {grosse_kugel_up_mean}")
# print(f"Viskosität grosse Kugel: {grosse_kugel_viscosity}")

grosse_kugel_velocity = grosse_kugel_way_length / grosse_kugel_mean

# grosse_kugel_reynold = (
#     grosse_kugel_velocity * grosse_kugel_diameter * density_water
# ) / grosse_kugel_density

# print(f"grosse kugel reynold: {grosse_kugel_reynold}")

grosse_kugel_temperature = pd.read_csv("assets/grosse_kugel_temperatur.csv")

grosse_kugel_temperature["T"] = grosse_kugel_temperature["T"] + 273.15

grosse_kugel_temperature["mean"] = grosse_kugel_temperature[
    ["t_u_1", "t_d_1", "t_u_2", "t_d_2"]
].mean(axis=1)


grosse_kugel_device_constant = 1.77 * 10**-8

grosse_kugel_viscosity = (
    (grosse_kugel_density - density_water)
    * grosse_kugel_device_constant
    * grosse_kugel_mean
)

grosse_kugel_temperature["eta"] = (
    (grosse_kugel_density - density_water)
    * grosse_kugel_device_constant
    * grosse_kugel_temperature["mean"]
)

grosse_kugel_temperature["eta"] = grosse_kugel_temperature["eta"] * 10**3

with open("build/grosse_kugel_temperatur.tex", "w") as f:
    f.write(
        grosse_kugel_temperature.to_latex(
            header=[
                "$T$ in $\\si{{\\kelvin}}$",
                up_char + " $t_{{1}}$ in $\\si{{\\second}}$",
                down_char + " $t_{{2}}$ in $\\si{{\\second}}$",
                up_char + " $t_{{3}}$ in $\\si{{\\second}}$",
                down_char + " $t_{{4}}$ in $\\si{{\\second}}$",
                "$t_{{m}}$ in $\si{{\second}}$",
                "$\eta$ in $\si{{\milli\pascal\second}}$",
            ],
            index=False,
            formatters=[
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
            ],
        )
    )


def f(T, A, B):
    return A * np.exp(B / T)


params, cov = curve_fit(
    f, grosse_kugel_temperature["T"], grosse_kugel_temperature["eta"]
)


print(f"Fit: A = {params[0]} = = {params[0] * 10**3}")
print(f"Fit: a = ln(A) = {np.log(params[0])}")
print(f"Fit: B = {params[1]}")

plt.plot(
    1 / grosse_kugel_temperature["T"],
    np.log(f(grosse_kugel_temperature["T"], *params)),
    "r-",
    label=r"Ausgleichsgerade",
)
plt.plot(
    1 / grosse_kugel_temperature["T"],
    np.log(grosse_kugel_temperature["eta"]),
    "x",
    label="Messwerte",
)
plt.xlabel("$\\frac{{1}}{{T}}/\\si{{\\kelvin}}^{{-1}}$")
plt.ylabel(r"$ln(\eta)ln(\si{\kilo\gram\per\meter\per\second})$")
plt.legend(loc="best")
plt.grid()
plt.savefig("build/grosse_kugel_temperature.pdf")


compare = pd.DataFrame()


def wasser_lit(temp):
    return 1 / (0.1 * (temp**2) - 34.335 * temp + 2472)


def abweichung(approx, exact):
    return (abs(approx - exact) / exact) * 100


compare["T"] = grosse_kugel_temperature["T"]
compare["eta1"] = grosse_kugel_temperature["eta"]
compare["eta2"] = wasser_lit(grosse_kugel_temperature["T"]) * 10**3
compare["difference"] = abweichung(compare["eta1"], compare["eta2"])

with open("build/compare.tex", "w") as f:
    f.write(
        compare.to_latex(
            header=[
                "$T/\\si{{\\kelvin}}$",
                "$\\eta_{{mess}}/\\si{{\\milli\\pascal\\second}}$",
                "$\\eta_{{lit}}/\\si{{\\milli\\pascal\\second}}$",
                "$\\Delta x/\\si{{\\percent}}$",
            ],
            index=False,
            formatters=[
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
            ],
        )
    )
print(compare)


def reynold(p, t, d, eta):
    v = grosse_kugel_way_length / t
    return (p * v * d) / eta


reynold_data_frame = pd.DataFrame()

reynold_data_frame["T"] = grosse_kugel_temperature["T"]
reynold_data_frame["t"] = grosse_kugel_temperature["mean"]
reynold_data_frame["v"] = (grosse_kugel_way_length / grosse_kugel_temperature["mean"]) * 10**3
reynold_data_frame["eta"] = grosse_kugel_temperature["eta"]
reynold_data_frame["R"] = reynold(
    density_water,
    grosse_kugel_temperature["mean"],
    grosse_kugel_diameter,
    reynold_data_frame["eta"] * 10**-3,
)

print(f"reynolds mean = {reynold_data_frame['R'].mean()}")


with open("build/reynolds.tex", "w") as f:
    f.write(
        reynold_data_frame.to_latex(
            header=[
                "$T$ in $\\si{{\\kelvin}}$",
                "$t$ in $\\si{{\\second}}$",
                "$v$ in $\\si{{\\milli\\meter\per\second}}$",
                "$\\eta_{{mess}}$ in $\\si{{\\milli\\pascal\\second}}$",
                # "$\\Delta x/\\si{{\\percent}}$",
                "$Re$",
            ],
            index=False,
            formatters=[
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
            ],
        )
    )
