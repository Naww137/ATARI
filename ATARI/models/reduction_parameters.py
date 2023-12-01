


class reduction_parameter:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner) -> tuple:
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        # instance.__dict__[self._name] = date.fromisoformat(value)
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Tuple for reduction parameter must be (value, uncertainty)")
        else:
            raise ValueError("Must supply tuple for reduction parameter value and uncertainty")
        instance.__dict__[self._name] = value


class transmission_rpi_parameters:

    trigo = reduction_parameter()
    trigs = reduction_parameter()
    m1    = reduction_parameter()
    m2    = reduction_parameter()
    m3    = reduction_parameter()
    m4    = reduction_parameter()
    ks    = reduction_parameter()
    ko    = reduction_parameter()
    b0s   = reduction_parameter()
    b0o   = reduction_parameter()
    a_b   = reduction_parameter()

    def __init__(self, **kwargs):

        self.trigo  = (9758727,      0)
        self.trigs  = (18476117,     0)
        self.m1     = (1,            0.016)
        self.m2     = (1,            0.008)
        self.m3     = (1,            0.018)
        self.m4     = (1,            0.005)
        self.ks     = (0.563,        0.02402339737495515)
        self.ko     = (1.471,        0.05576763648617445)
        self.b0s    = (9.9,          0.1)
        self.b0o    = (13.4,         0.7)
        self.a_b    = ([582.7768594580712, 0.05149689096209191],
                        [[1.14395753e+03,  1.42659922e-1],
                         [1.42659922e-1,   2.19135003e-05]])

        for key, value in kwargs.items():
            setattr(self, key, value)

    
    


