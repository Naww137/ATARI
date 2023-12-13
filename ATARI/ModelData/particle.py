
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

        Attributes
        ----------
        Z      : int
            Atomic number
        A      : int
            Atomic mass number
        I      : half-integer
            Particle spin
        mass   : float
            Nuclei mass in atomic mass units (amu)
        radius : float
            Nuclear mean square radius in femtometers (fm). Default is automatically approximated
            using `1.23 * A**(1/3)`.
        name   : str
            Name of the particle. Default is ZZAAA MCNP ID form.
        """
        # Atomic Number:
        self._Z = int(Z)
        # Atomic Mass:
        if A < Z:   print(Warning('Are you sure A < Z?'))
        self._A = int(A)
        # Isotope Spin:
        if I % 0.5 != 0.0:  raise ValueError(f'The isotope spin, {I}, must be a half-integer.')
        self._I = float(I)
        # Mass: (amu)
        self._mass = mass
        
        # Nuclear Radius: (fm)
        if radius is not None:
            if   radius > 1e2:      print(Warning(f'The channel radius, {radius} fm, is quite high. Make sure it is in units of femtometers.'))
            elif radius < 1e-2:     print(Warning(f'The channel radius, {radius} fm, is quite low. Make sure it is in units of femtometers.'))
            self._radius = float(radius)
        else:
            self._radius = 1.23 * self._A**(1/3)
        
        # Particle Name:
        if name is not None:    self._name = str(name)
        else:                   self._name = str(self._Z*1000+self._A) # MCNP ID for the isotope.

    @property
    def Z(self):
        'Atomic number'
        return self._Z
    @property
    def A(self):
        'Atomic mass number'
        return self._A
    @property
    def I(self):
        'Particle spin'
        return self._I
    @property
    def mass(self):
        'Nuclei mass in atomic mass units (amu)'
        return self._mass
    @property
    def radius(self):
        'Nuclear mean square radius in femtometers (fm)'
        return self._radius
    @property
    def name(self):
        'Name of the particle'
        return self._name
    @property
    def N(self):
        'Number of Neutrons'
        return self._A - self._Z
    @property
    def AWRI(self):
        'Nuclei mass divided by neutron mass'
        return self._mass / mass_neutron

    def __repr__(self):
        txt  = f'Particle       = {self._name}\n'
        txt += f'Atomic Number  = {self._Z}\n'
        txt += f'Atomic Mass    = {self._A}\n'
        txt += f'Nuclear Spin   = {self._I}\n'
        txt += f'Mass           = {self._mass:.7f} (amu)\n'
        txt += f'Nuclear Radius = {self._radius:.7f} (fm)\n'
        return txt
    
    def __str__(self):
        return self.name
    
Neutron = Particle(Z=0 , A=1  , I=0.5, mass=mass_neutron, radius=0.8  , name='neutron')
Proton  = Particle(Z=1 , A=1  , I=0.5, mass=mass_proton , radius=0.833, name='proton')
Ta181   = Particle(Z=73, A=181, I=3.5, mass=180.94803   , radius=None , name='Ta181')