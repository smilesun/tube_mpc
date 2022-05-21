"""
Interface for stage constraint
"""


class ConstraintUbStage():
    """ConstraintStage.
    for a single time-step (stage) decision variable
    like z_0, z_T, x_T, x_t, z_t
    but not including dynamic equality constraint,
    since the dynamic equality constraint connect
    two time-step (stage) decision variables

    # Statement: single time-step constraint has only inequality:
    it looks like z_0 is the only variable that requires
    a single stage equality constraint, however, z_0=x-s_0,
    where s_0 is decomposed to be J(\alpha) number of single
    step disturbance $w^j$, so the conclusion is any equality
    constraint connect different time-step, i.e. different
    decision variables.
    """

    @property
    def ncol(self):
        """ncol.
        this reflect the dimension of the single time-step
        decision variable, e.g. dim(x_t) = dim_sys
        this need to be the same for both equality and
        inequality constraint
        """
        raise NotImplementedError

    @property
    def nrow_ub(self):
        """nrow.
        this reflect the number of constraint for a single
        time-step decision variable, e.g. Mx<=1
        this function will return nrow(M)
        """
        raise NotImplementedError
