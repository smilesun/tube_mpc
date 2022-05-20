import numpy as np


class ConstraintEqLdyn1T():
    """Equality Constraint for Linear Dynamic System
    - Decision variable: d = x_{0:T}, u_{0:T-1}
        In block matrix form including all decision variables:
        suppose T=3
        - Block matrix form of d = x_{1:T}, u_{0:T-1}, suppose T=3
        - Ax_0+Bu_0=x1:
        [A_{n}, -[I]_n,  [0]_n,   [0]_n,|[B]_{n*r}, [0]_n,         [0]_n] d =0
        - Ax_1+Bu_1=x2:
        [0_{n},  [A]_n,  -[I]_n,  [0]_n,|[0]_n,     [B]_{n*r},     [0]_n] d =0
        - Ax_2+Bu_2=x3=xN:
        [0_{n},  [0]_n,   [A]_n, -[I]_n,|[0]_n,     [0]_n,     [B]_{n*r}] d =0

    - All together: M*d=[0_{1, dim(d)}]^T
        ---------------------------------------------------------------------
        [A_{n}, -[I]_n,  [0]_n,  [0]_n,|[B]_{n*r}, [0]_n,         [0]_n] d =0
        [0_{n},  [A]_n, -[I]_n,  [0]_n,|[0]_n,     [B]_{n*r},     [0]_n] d =0
        [0_{n},  [0]_n,  [A]_n, -[I]_n,|[0]_n,     [0]_n,     [B]_{n*r}] d =0

    - Use matrix subspace decomposition
        M=block(A)+block(I)+block(B)
        block(A)=[kron(A_{n}, eye(T)), 0_{T*n, n}, | 0_{T*n, T*r}        ]
        block(B)=[0_{T*n,(T+1)*n}                  | kron(B_{n*r}, eye(T)]
        block(I)=[0_{T*n,n}, -1*kron(I_n, eye(T))  | 0_{T*n, T*r}        ]
        ------------------------------------------------------------
        [ A, -I, 00,00 |  B,  0,  0|00, 00]
        [00,  A, -I,00 |  0,  B,  0|00, 00]
        [00,  00, A,-I |  0,  0,  B|00, 00]

    """

    def __init__(self, mat_input, mat_sys):
        self.mat_sys = mat_sys
        self.mat_input = mat_input
        self.dim_sys = self.mat_sys.shape[0]
        self.dim_input = self.mat_input.shape[1]
        self.horizon = None
        self.mat_lhs = None
        # self.mat_rhs = None, always zero

    def __call__(self, horizon):
        """__call__.
        ---------------------------------------------------------------------
        [A_{n}, -[I]_n,  [0]_n,  [0]_n,|[B]_{n*r}, [0]_{n*r}, [0]_{n*r}] d =0
        [0_{n},  [A]_n, -[I]_n,  [0]_n,|[0]_{n*r}, [B]_{n*r}, [0]_{n*r}] d =0
        [0_{n},  [0]_n,  [A]_n, -[I]_n,|[0]_{n*r}, [0]_{n*r}, [B]_{n*r}] d =0
        :param x: current state
        """
        if self.horizon == horizon:
            return self.mat_lhs
        n = self.mat_sys.shape[0]
        r = self.dim_input
        assert n == self.dim_sys

        mat_block_a = self.build_block_A(horizon, n, r)
        mat_block_b = self.build_block_b(horizon, n)
        mat_block_eye = self.build_block_eye(horizon, n, r)
        self.mat_lhs = mat_block_a + mat_block_b + mat_block_eye
        #
        # self.mat_rhs = np.zeros((horizon*(n), 1)), no need to create here
        # equality constraint is w.r.t. x, not u, not n+r
        self.horizon = horizon
        return self.mat_lhs

    def build_block_eye(self, horizon, n, r):
        """
        - build subspace of constraint matrix M with only A involved:
        block(I)=[0_{T*n,n}, -1*kron(I_n, eye(T))  | 0_{T*n, T*r}        ]
        - let horizon change each time, so keep horizon as parameter
        [ A, -I, 00,00 |  B,  0,  0]
        [00,  A, -I,00 |  0,  B,  0]
        [00,  00, A,-I |  0,  0,  B]
        """
        block_zero_a = np.zeros((horizon*n, n))  # shift
        # NOTE: not block_eye_subspace = -1*np.kron(np.eye(n),
        # np.eye(horizon)),
        # although in this case two forms are equivalent
        block_eye_subspace = -1*np.kron(np.eye(horizon), np.eye(n))
        block_zero_b = np.zeros((horizon*n, horizon*r))  # shift
        block_eye_full_space = np.hstack(
            [block_zero_a, block_eye_subspace, block_zero_b])
        return block_eye_full_space

    def build_block_b(self, horizon, n):
        """
        - build subspace of constraint matrix M with only A involved:

        block(B)=[0_{T*n,(T+1)*n}| kron(B_{n*r}, eye(T)]
        - let horizon change each time, so keep horizon as parameter
        [ A, -I, 00,00 |  B,  0,  0]
        [00,  A, -I,00 |  0,  B,  0]
        [00,  00, A,-I |  0,  0,  B]
        """
        block_zero_a = np.zeros((horizon*n, (horizon+1)*n))
        # NOTE: not block_b_subspace = np.kron(self.mat_input, np.eye(horizon))
        block_b_subspace = np.kron(np.eye(horizon), self.mat_input)
        # NOTE: for kronecker product,
        # the first argument's element will be replaced
        block_b_full_space = np.hstack(
            [block_zero_a, block_b_subspace])
        return block_b_full_space

    def build_block_A(self, horizon, n, r):
        """
        - build subspace of constraint matrix M with only A involved:
        block(A)=[kron(A_{n}, eye(T)), 0_{T*n, n}, | 0_{T*n, T*r}]
        - let horizon change each time, so keep horizon as parameter
        """
        block_zero_b = np.zeros((horizon*n, horizon*r))
        block_zero_a = np.zeros((horizon*n, n))
        # NOTE: not block_a_subspace = np.kron(self.mat_sys, np.eye(horizon))
        block_a_subspace = np.kron(np.eye(horizon), self.mat_sys)
        block_a_fullspace = np.hstack(
            [block_a_subspace, block_zero_a, block_zero_b])
        return block_a_fullspace
