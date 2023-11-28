import pandas as pd

rund_einseitig = pd.read_csv("data/rund_einseitig.csv", delimiter=";")
eckig_einseitig = pd.read_csv("data/eckig_einseitig.csv", delimiter=";")
rund_zweiseitig = pd.read_csv("data/rund_zweiseitig.csv", delimiter=";")
eckig_zweiseitig = pd.read_csv("data/eckig_zweiseitig.csv", delimiter=";")


with open("build/rund_einseitig_table.tex", "w") as f:
    f.write(
        rund_einseitig.to_latex(
            header=["x/cm", "$\Delta$D/cm"],
            index=False,
            formatters=["{:.1f}".format, "{:.2f}".format, "{:.2f}".format],
        )
    )


with open("build/eckig_einseitig_table.tex", "w") as f:
    f.write(
        eckig_einseitig.to_latex(
            header=["x/cm", "$\Delta$D/cm"],
            index=False,
            formatters=["{:.1f}".format, "{:.2f}".format, "{:.2f}".format],
        )
    )

with open("build/rund_zweiseitig_table.tex", "w") as f:
    f.write(
        rund_zweiseitig.to_latex(
            header=["x/cm", "$\Delta$D/cm links", "$\Delta$D/cm rechts"],
            index=False,
            formatters=[
                int,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
            ],
        )
    )


with open("build/eckig_zweiseitig_table.tex", "w") as f:
    f.write(
        eckig_zweiseitig.to_latex(
            header=["x/cm", "$\Delta$D/cm links", "$\Delta$D/cm rechts"],
            index=False,
            formatters=[
                int,
                "{:.2f}".format,
                "{:.2f}".format,
                "{:.2f}".format,
            ],
        )
    )
