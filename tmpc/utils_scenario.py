class Scenario():
    """Scenario
    basic information about problem setting:
        dimension of system
        dimension of input
        local controler gain
        system matrix
        input matrix
    """
    def __init__(self):
        self._x_0 = None
        self._dim_sys = None
        self._dim_input = None
