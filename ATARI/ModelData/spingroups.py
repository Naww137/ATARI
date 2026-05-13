
__doc__ = """
This file defines all classes related to spingroups. "HalfInt" is a class for half-integers.
"Spingroup" is a class to define one spingroup (a pair of orbital angular momentum, total angular
momentum, and channel spin).
"""

# =================================================================================================
#   Half-Integers:
# =================================================================================================

class HalfInt:
    """
    Data type for half-integers to represent spins.
    """

    def __init__(self, value):
        if isinstance(value, HalfInt):
            self = value
        else:
            if value % 0.5 != 0.0:
                raise ValueError(f'The number, {value}, is not a half-integer.')
            self.__2x_value = int(2*value)
    
    @property
    def parity(self):   return '+' if self.__2x_value >= 0 else '-'
    @property
    def value(self):    return 0.5 * float(self.__2x_value)

    def __repr__(self):
        if self.__2x_value % 2 == 0:
            return f'{self.__2x_value//2}'
        else:
            return f'{self.__2x_value}/2'
    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(self.__2x_value)

    # Arithmetic:
    def __float__(self):
        return self.value
    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.value == other.value
        else:
            return self.value == other
    def __ne__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.value != other.value
        else:
            return self.value != other
    def __lt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.value < other.value
        else:
            return self.value < other
    def __le__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.value <= other.value
        else:
            return self.value <= other
    def __gt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.value > other.value
        else:
            return self.value > other
    def __ge__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.value >= other.value
        else:
            return self.value >= other
    def __add__(self, other):
        if   isinstance(other, self.__class__):
            return self.__class__(self.value + other.value)
        elif isinstance(other, int):
            return self.__class__(self.value + other)
        else:
            return self.value + other
    def __radd__(self, other):
        if   isinstance(other, int):
            return self.__class__(other + self.value)
        else:
            return other + self.value
    def __sub__(self, other):
        if   isinstance(other, self.__class__):
            return self.__class__(self.value - other.value)
        elif isinstance(other, int):
            return self.__class__(self.value - other)
        else:
            return self.value - other
    def __rsub__(self, other):
        if   isinstance(other, int):
            return self.__class__(other - self.value)
        else:
            return other - self.value
    def __mul__(self, other):
        if   isinstance(other, self.__class__):
            return self.value * other.value
        elif isinstance(other, int) and (other % 2 == 0):
            return self.__2x_value * (other // 2)
        else:
            return self.value * other
    def __rmul__(self, other):
        if isinstance(other, int) and (other % 2 == 0):
            return self.__2x_value * (other // 2)
        else:
            return self.value * other
        
# =================================================================================================
#   Spingroup Class:
# =================================================================================================

class Spingroup:
    """
    A class containing the orbital angular momentum, "L", and the total spin, "J", for the
    reaction. The quantum number, "S", can also be given optionally.

    Attributes
    ----------
    L : int
        Orbital angular momentum.
    J : HalfInt
        Total angular momentum.
    S : HalfInt
        Channel spin. Default = None.
    """

    def __init__(self, l:int, j:HalfInt, s:HalfInt=None):
        """
        Creates a Spingroup object based on the quantum numbers for the reaction.

        Parameters
        ----------
        l : int
            Orbital angular momentum.
        j : HalfInt
            Total angular momentum.
        s : HalfInt
            Channel spin. Default = None.
        """

        self.L = int(l)
        
        self.J = HalfInt(j)

        if s is not None:   self.S = HalfInt(s)
        else:               self.S = None

    def __format__(self, spec:str):
        if (spec is None) or (spec == 'jpi'):
            if self.L % 2:  return f'{self.J}-'
            else:           return f'{self.J}+'
        elif spec == 'lj':
            return f'({self.L},{self.J})'
        elif spec == 'ljs':
            return f'({self.L},{self.J},{self.S})'
        else:
            raise ValueError('Unknown format specifier.')

    def __repr__(self):
        return f'{self:ljs}'
    def __str__(self):
        return f'{self:jpi}'
    
    def __hash__(self):
        return hash((self.L, self.J, self.S))
    
    @property
    def Jpi(self):
        if self.L % 2:  return  float(self.J)
        else:           return -float(self.J)
    
    def g(self, spin_target:HalfInt, spin_proj:HalfInt):
        'Statistical spin factor'
        return (2*self.J+1) / ((2*spin_target+1) * (2*spin_proj+1))
    
    @classmethod
    def zip(cls, Ls, Js, Ss=None):
        """
        Generates spingroups from the provided "Ls", "Js" and "Ss" quantities.

        Parameters
        ----------
        Ls : list [int]
            The ordered list of orbital angular momentums numbers.
        Js : list [HalfInt]
            The ordered list of total angular momentum numbers.
        Ss : list [HalfInt]
            The ordered list of channel spin numbers.

        Returns
        -------
        spingroups : Spingroups
            The generated spingroups.
        """
        if Ss is None:
            if not (len(Ls) == len(Js)):
                raise ValueError('The number of "L" and "J" values for spin-groups are not equal.')
            spingroups = [cls(l, j) for l, j in zip(Ls, Js)]
        else:
            if not (len(Ls) == len(Js) == len(Ss)):
                raise ValueError('The number of "L", "J", and "S" values for spin-groups are not equal.')
            spingroups = [cls(l, j, s) for l, j, s in zip(Ls, Js, Ss)]
        return spingroups
    
    @classmethod
    def unzip(cls, spingroups:list):
        """
        Given a list of spingroups, unzip returns a tuple of lists of orbital angular momentums,
        total angular momentums, and channel spins.

        Parameter:
        ---------
        spingroups : list [Spingroup]
            The list of spingroup objects.

        Returns
        -------
        Ls : list [int]
            The ordered list of orbital angular momentums numbers.
        Js : list [HalfInt]
            The ordered list of total angular momentum numbers.
        Ss : list [HalfInt]
            The ordered list of channel spin numbers.
        """
        Ls = [];  Js = [];  Ss = []
        for spingroup in spingroups:
            Ls.append(spingroup.L)
            Js.append(spingroup.J)
            Ss.append(spingroup.S)
        return Ls, Js, Ss

    @classmethod
    def find(cls, spin_targ, spin_proj=1/2, l_max:int=1):
        """
        Finds all of the valid spingroups with "l" less than or equal to "l_max".

        Parameters
        ----------
        spin_target : HalfInt
            The quantum spin number for the target nuclei.
        spin_proj   : HalfInt
            The quantum spin number for the projectile nuclei.
        l_max       : int
            The maximum orbital angular momentum number generated.

        Returns
        -------
        spingroups  : Spingroups
            The generated spingroups.
        """
        l_max = int(l_max)
        spingroups = []
        for l in range(l_max+1):
            for s in range(abs(spin_targ-spin_proj), (spin_targ+spin_proj+1), 1):
                for j in range(abs(s-l), s+l+1, 1):
                    spingroups.append(cls(l, j, s))
        return spingroups
    
    @staticmethod
    def id(spingroup, spingroups:list):
        """
        Returns an integer index ID if provided a spingroup. If an integer id is provided, the id
        is passed.

        Parameters
        ----------
        spingroup  : Spingroup or int or 'false' or 'False'
            The Spingroup object or integer ID.
        spingroups : List [Spingroup]
            A list of all considered spingroups.
        
        Returns
        -------
        g          : int
            Integer ID for the spingroup, based on the list, spingroups.
        """

        if spingroup in ('false', 'False'):
            return len(spingroups)
        elif isinstance(spingroup, Spingroup):
            for g, candidate in enumerate(spingroups):
                if spingroup == candidate:
                    return g
            raise ValueError(f'The provided spingroup, {spingroup}, does not match any of the recorded spingroups.')
        elif isinstance(spingroup, int):
            num_sgs = len(spingroups)
            if (spingroup > num_sgs) or (spingroup < 0):
                raise ValueError(f'The provided spingroup id, {spingroup}, is above the number of spingroups, {num_sgs}.')
            g = spingroup
            return g
        else:
            raise TypeError(f'The provided spingroup, {spingroup}, is not an integer ID nor is it a "Spingroup" object.')