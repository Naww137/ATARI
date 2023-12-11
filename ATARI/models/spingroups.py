

class halfint:
    """
    Data type for half-integers to represent spins.
    """

    def __init__(self, value):
        if type(value) == halfint:
            self = value
        else:
            if value % 0.5 != 0.0:
                raise ValueError(f'The number, {value}, is not a half-integer.')
            self.__2x_value = int(2*value)

    @property
    def value(self):    return 0.5 * float(self.__2x_value)

    def __repr__(self):
        if self.__2x_value % 2 == 0:
            return f'{self.__2x_value//2}'
        else:
            return f'{self.__2x_value}/2'

    # Arithmetic:
    def __float__(self):
        return float(self.value)
    def __eq__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value == other.value
        else:
            return self.value == other
    def __ne__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value != other.value
        else:
            return self.value != other
    def __lt__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value < other.value
        else:
            return self.value < other
    def __le__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value <= other.value
        else:
            return self.value <= other
    def __gt__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value > other.value
        else:
            return self.value > other
    def __ge__(self, other) -> bool:
        if type(other) == self.__class__:
            return self.value >= other.value
        else:
            return self.value >= other
    def __add__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value + other.value)
        elif type(other) == int:
            return self.__class__(self.value + other)
        else:
            return self.value + other
    def __radd__(self, other):
        if type(other) == int:
            return self.__class__(other + self.value)
        else:
            return other + self.value
    def __sub__(self, other):
        if type(other) == self.__class__:
            return self.__class__(self.value - other.value)
        elif type(other) == int:
            return self.__class__(self.value - other)
        else:
            return self.value - other
    def __rsub__(self, other):
        if type(other) == int:
            return self.__class__(other - self.value)
        else:
            return other - self.value
    def __mul__(self, other):
        if type(other) == self.__class__:
            return self.value * other.value
        elif (type(other) == int) and (other % 2 == 0):
            return self.__2x_value * (other // 2)
        else:
            return self.value * other
    def __rmul__(self, other):
        if (type(other) == int) and (other % 2 == 0):
            return self.__2x_value * (other // 2)
        else:
            return self.value * other

