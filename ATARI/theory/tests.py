from resonance_statistics import chisquare_PDF, wigner_PDF
import numpy as np
from scipy.integrate import trapezoid

x = np.linspace(0,100, 1000000)
y = chisquare_PDF(x, 1000, 60)
assert(np.isclose(trapezoid(y, x), 1.0))

xw =np.linspace(0, 50, 100000)
yw = wigner_PDF(xw, 8)
assert(np.isclose(trapezoid(yw, xw), 1.0))