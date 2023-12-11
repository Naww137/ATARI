from ATARI.models.spingroups import halfint

__doc__ = """
This file stores the "Particle" class. The "Particle" class contains all relevent information to a
specific particle. Objects of this class are used when defining a reaction's properties in
"Reaction" objects. The "Neutron" and "Proton" objects has already been defined for convenience.
"""

# =================================================================================================
#    Particle:
# =================================================================================================

mass_neutron = 1.00866491588 #  amu  (source: ENDF-6 manual Table H.2)
mass_proton  = 1.007276466621 # amu  (source: ENDF-6 manual Table H.2)

class Particle:
    """
    Particle is a class that contains information about a particular particle, including atomic
    number, atomic mass number, nuclei mass, nuclear radius, and the name of the particle.
    """

    def __init__(self, Z, A, I, mass, radius=None, name=None):
        """
        Initialize a Particle object.

        Attributes:
        ----------
        Z      :: int
            Atomic number
        A      :: int
            Atomic mass number
        I      :: halfint
            Particle spin
        mass   :: float
            Nuclei mass in atomic mass units (amu)
        AWRI   :: float
            Nuclei mass divided by neutron mass
        radius :: float
            Nuclear mean square radius in femtometers (fm)
        name   :: str
            Name of the particle
        """
        # Atomic Number:
        self.Z = int(Z)
        # Atomic Mass:
        self.A = int(A)
        # Isotope Spin:
        self.I = halfint(I)
        # Mass: (amu)
        self.mass = float(mass)
        # AWRI:
        if self.mass is not None:   self.AWRI = self.mass / mass_neutron
        else:                       self.AWRI = None
        
        # Nuclear Radius: (fm)
        if radius is not None:
            if   radius > 1e2:      print(Warning(f'The channel radius, {radius}, is quite high. Make sure it is in units of femtometers.'))
            elif radius < 1e-2:     print(Warning(f'The channel radius, {radius}, is quite low. Make sure it is in units of femtometers.'))
            self.radius = float(radius)
        elif self.A is not None:
            self.radius = 1.23 * self.A**(1/3)
        else:
            self.radius = None
        
        # Particle Name:
        if name is not None:
            self.name = str(name)
        elif (self.A is not None) and (self.Z is not None):
            self.name = str(self.Z*1000+self.A) # MCNP ID for the isotope.
        else:
            self.name = '???'

    def __repr__(self):
        txt  = f'Particle       = {self.name}\n'
        txt += f'Atomic Number  = {self.Z}\n'
        txt += f'Atomic Mass    = {self.A}\n'
        txt += f'Nuclear Spin   = {self.I}\n'
        txt += f'Mass           = {self.mass:.7f} (amu)\n'
        txt += f'Nuclear Radius = {self.radius:.7f} (fm)\n'
        return txt
    
    def __str__(self):
        return self.name
    
Neutron = Particle(Z=0 , A=1  , I=0.5, mass=mass_neutron, radius=0.8  , name='neutron')
Proton  = Particle(Z=1 , A=1  , I=0.5, mass=mass_proton , radius=0.833, name='proton')
Ta181   = Particle(Z=73, A=181, I=3.5, mass=180.94800   , radius=None , name='Ta181')