from resonance_statistics import chisquare_PDF, wigner_PDF
import numpy as np
from scipy.integrate import trapezoid
from ATARI.ModelData.particle import Particle, Neutron
from ATARI.ModelData.particle_pair import Particle_Pair

x = np.linspace(0,100, 1000000)
y = chisquare_PDF(x, 1000, 60)
assert(np.isclose(trapezoid(y, x), 1.0))

xw =np.linspace(0, 50, 100000)
yw = wigner_PDF(xw, 8)
assert(np.isclose(trapezoid(yw, xw), 1.0))

Ta181 = Particle(Z=73, A=181, I=3.5, mass=180.94803, name='Ta181')
Ta_pair = Particle_Pair()

# test adding spin groups
Ta_pair.add_spin_group(Jpi='3.0',
                       J_ID=1,
                       D=8.79,
                       gn2_avg=46.5,
                       gn2_dof=1,
                       gg2_avg=64.0,
                       gg2_dof=1000)
# test float as Jpi
Ta_pair.add_spin_group(Jpi=4.0,
                       J_ID=2,
                       D=4.99,
                       gn2_avg=35.5,
                       gn2_dof=1,
                       gg2_avg=64.0,
                       gg2_dof=1000)

# test adding spin group with impossible Jpi
try:
    Ta_pair.add_spin_group(Jpi=4.1,
                       J_ID=2,
                       D=4.99,
                       gn2_avg=35.5,
                       gn2_dof=1,
                       gg2_avg=64.0,
                       gg2_dof=1000)
    raise ValueError()
except:
    pass

# test sampling resonance ladder
Ta_pair.sample_resonance_ladder()

# test clearing spin groups
Ta_pair.clear_spin_groups()
assert not Ta_pair.spin_groups 


