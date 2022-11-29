import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def dBToRelational(value: float):
    return pow(10, ((1/20)*value))
    return 10**((1/20)*value)


class Base:
    dataX = np.array([])
    dataY = np.array([])

    plotX = np.array([])

    title = ""

    outputFile = ""

    def f(self, x, c, b):
        return c * x + b

    def plot(self):
        assert self.outputFile, "No output file path provided"

        
        dataXLog = np.log(self.dataX)
        dataYLog = np.log(dBToRelational(self.dataY))
        plotXLog = np.log(self.plotX)
        params, _ = curve_fit(
            self.f, dataXLog, dataYLog)
        plt.plot(plotXLog, self.f(plotXLog, *params), 'k-',
                 label='Anpassungsfunktion ' + r'$(y=x\times ' + str(round(params[0], 3)) + r'+' + str(round(params[1], 3)) + r')$', linewidth=0.5)
        plt.gcf().subplots_adjust(bottom=0.18)
        plt.plot(dataXLog, dataYLog,
                 'r.', label='Messwerte')
        if(self.title):
            plt.title(self.title)
        plt.legend()
        plt.grid()
        plt.xlabel(r'$\ln{(f)}$')
        plt.ylabel(r'$\ln { \left( \frac{U_n}{U_0} \right)}$')
        plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
        plt.savefig(self.outputFile)
        plt.clf()


class Square(Base):
    dataX = np.array([20,  60, 100,   140,   180,
                      220,   260,   300,   340,   380])
    dataY = np.array([-1.79, -11, -15, -17.8, -19.8, -
                      21.4, -23.4, -24.6, -26.2, -27.4])

    plotX = np.linspace(0, 400)

    outputFile = "build/square.pdf"


class Ramp1(Base):
    dataX = np.array([20,  40, 60,   80,   100,
                      120,   140,   160,   180,   200])
    dataY = np.array([-6.59, -13, -16.6, -19.8, -22.2,
                      -24.6, -23.8, -24.6, -26.2, -26.6])

    plotX = np.linspace(0, 300)

    outputFile = "build/ramp1.pdf"


class Ramp2(Ramp1):
    dataY = np.array([-6.59, -13, -16.6, -19.8, -22.2,
                      -24.6, -23.8, -24.6, -26.2, -26.6])
    outputFile = "build/ramp2.pdf"


Square().plot()
Ramp1().plot()
Ramp2().plot()
