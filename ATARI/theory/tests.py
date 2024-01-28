
from ATARI.ModelData.particle import Particle, Neutron
from ATARI.ModelData.particle_pair import Particle_Pair

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


