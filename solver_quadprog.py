"""
ref:https://scaron.info/blog/quadratic-programming-in-python.html
"""
import numpy
import numpy as np
import quadprog


def quadprog_solve_qp(P, A_ub, b_ub, A_eq=None, b_eq=None, q=None):
    """quadprog_solve_qp.
    :param P:
    :param q: -a^T*x
    :param A_ub:
    :param b_ub:
    :param A_eq:
    :param b_eq:
    The code below converts the above parameters to the follwing problem:
    quadprog.solve_qp(mat_pos_def, qp_a, qp_C, qp_b, meq)[0]
    Minimize     1/2 x^T G x - a^T x
    Subject to   C.T x >= b
    <=>   -C.T x <= -b
        Parameters
        ----------
        suppose dim(x) = n
        G : array, shape=(n, n)
            matrix appearing in the quadratic function to be minimized
        a : array, shape=(n,)
            vector appearing in the quadratic function to be minimized
        C : array, shape=(n, m)
            matrix defining the constraints under which we want to minimize the
            quadratic function
        b : array, shape=(m), default=None
            vector defining the constraints
        meq : int, default=0
            the first meq constraints are treated as equality constraints,
            all further as inequality constraints (defaults to 0).
            """
    mat_pos_def = .5 * (P + P.T)   # make sure P is symmetric
    if A_eq is None:
        A_eq = np.zeros_like(A_ub)
    if b_eq is None:
        b_eq = np.zeros_like(b_ub)
    if q is None:
        q = np.zeros(P.shape[0])
    qp_a = -1.0 * q
    qp_C = -1.0 * numpy.vstack([A_eq, A_ub]).T
    qp_b = -1.0 * numpy.hstack([b_eq, b_ub])
    meq = A_eq.shape[0]
    return quadprog.solve_qp(mat_pos_def, qp_a, qp_C, qp_b, meq)[0]


def test_qp():
    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = np.dot(M.T, M)
    A_ub = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    b_ub = np.array([3., 2., -2.])
    quadprog_solve_qp(P=P, A_ub=A_ub, b_ub=b_ub)
    q = -np.dot(M.T, np.array([3., 2., 3.]))
    quadprog_solve_qp(P=P, A_ub=A_ub, b_ub=b_ub, q=q)
