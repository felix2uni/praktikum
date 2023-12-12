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
print(f"Mean grosse Kugel up: {grosse_kugel_up_mean}")
print(f"Mean grosse Kugel up: {grosse_kugel_up_mean}")
# print(f"Viskosität grosse Kugel: {grosse_kugel_viscosity}")

grosse_kugel_velocity = grosse_kugel_way_length / grosse_kugel_mean

grosse_kugel_reynold = (
    grosse_kugel_velocity * grosse_kugel_diameter * density_water
) / grosse_kugel_density

print(f"grosse kugelreynold: {grosse_kugel_reynold}")

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

with open("build/grosse_kugel_temperatur.tex", "w") as f:
    f.write(
        grosse_kugel_temperature.to_latex(
            header=[
                "$T/K$",
                f"{up_char} $t_1/s$",
                f"{down_char} $t_2/s$",
                f"{up_char} $t_3/s$",
                f"{down_char} $t_4/s$",
                "$t_m/s$",
                "$ \eta/\si{{\milli\pascal\second}} $",
            ],
            index=False,
            formatters=[
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2E}".format,
            ],
        )
    )


def f(T, A, B):
    return A * np.exp(B / T)


params, cov = curve_fit(
    f, grosse_kugel_temperature["T"], grosse_kugel_temperature["eta"]
)


print(params)

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
plt.xlabel(r"$(1/T)/\si{\kelvin}^{-1}$")
plt.ylabel(r"$ln(\eta/o)$")
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
compare["eta2"] = wasser_lit(grosse_kugel_temperature["T"])
compare["difference"] = abweichung(compare["eta1"], compare["eta2"])

with open("build/compare.tex", "w") as f:
    f.write(
        compare.to_latex(
            header=[
                """$T/K$""",
                """$\eta_\text{{mess}}/\si{{\milli\pascal\second}}$""",
                """$\eta_\text{{lit}}/\si{{\milli\pascal\second}}$""",
                """$\Delta x/\si{{\percent}}$""",
            ],
            index=False,
            formatters=[
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format
            ],
        )
    )
print(compare)
