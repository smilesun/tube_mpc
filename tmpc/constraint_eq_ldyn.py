import numpy as np


class ConstraintEqLdyn():
    """Equality Constraint for Linear Dynamic System
    - Decision variable: d = x_{1:T}, u_{0:T-1}
        - First constraint: x_1 = Ax_0 + Bu_0 with dimension:  n*1
        In block matrix form including all decision variables:
        suppose T=3
        [-I_{n},[0]_n,[0]_n,  B_{n*r],[0]_{n*r],[0]_{n*r]] d = A_{d*d}[x]_{n*1}
        not in consistent form with other constraint

    - Solution: use more variable to have
        Introduce x_0 as decision variable and constraint x_0=x
        Decision Variable: x_{0:T}, u_{0:T-1}
        - First two constraint: x_0=x, x_1=Ax_0+Bu_0
        - Block matrix form of d = x_{1:T}, u_{0:T-1}, suppose T=3
        - x_0=x:
        [I_{n}, [0]_n,   [0]_n,   [0]_n,|[0]_n,     [0]_n,         [0]_n] d =x
        - Ax_0+Bu_0=x1:
        [A_{n}, -[I]_n,  [0]_n,   [0]_n,|[B]_{n*r}, [0]_n,         [0]_n] d =0
        - Ax_1+Bu_1=x2:
        [0_{n},  [A]_n,  -[I]_n,  [0]_n,|[0]_n,     [B]_{n*r},     [0]_n] d =0
        - Ax_2+Bu_2=x3=xN:
        [0_{n},  [0]_n,   [A]_n, -[I]_n,|[0]_n,     [0]_n,     [B]_{n*r}] d =0

    - All together: Md=[x, 0_{1, T*d}]^T
    Md= [I_{n},  [0]_n,  [0]_n,  [0]_n,|[0]_n,     [0]_n,         [0]_n] d =x
        ---------------------------------------------------------------------
        [A_{n}, -[I]_n,  [0]_n,  [0]_n,|[B]_{n*r}, [0]_n,         [0]_n] d =0
        [0_{n},  [A]_n, -[I]_n,  [0]_n,|[0]_n,     [B]_{n*r},     [0]_n] d =0
        [0_{n},  [0]_n,  [A]_n, -[I]_n,|[0]_n,     [0]_n,     [B]_{n*r}] d =0
    - Use matrix subspace decomposition
    M[2:T*n, :]=block(A)+block(I)+block(B)
    block(A)=[kron(A_{n}, eye(T)), 0_{T*n, n}, | 0_{T*n, T*r}        ]
    block(B)=[0_{T*n,(T+1)*n}                  | kron(B_{n*r}, eye(T)]
    block(I)=[0_{T*n,n}, -1*kron(I_n, eye(T))  | 0_{T*n, T*r}        ]
    - M=vstack(M[1,:], M[2:T*n,:])
    """

    def __init__(self, mat_input, mat_sys):
        self.mat_sys = mat_sys
        self.mat_input = mat_input
        self.dim_sys = self.mat_sys.shape[0]
        self.dim_input = self.mat_input.shape[1]

    def __call__(self, x, horizon):
        """__call__.
    Md= [I_{n},  [0]_n,  [0]_n,  [0]_n,|[0]_{n*r}, [0]_{n*r}, [0]_{n*r}] d =x
        ---------------------------------------------------------------------
        [A_{n}, -[I]_n,  [0]_n,  [0]_n,|[B]_{n*r}, [0]_{n*r}, [0]_{n*r}] d =0
        [0_{n},  [A]_n, -[I]_n,  [0]_n,|[0]_{n*r}, [B]_{n*r}, [0]_{n*r}] d =0
        [0_{n},  [0]_n,  [A]_n, -[I]_n,|[0]_{n*r}, [0]_{n*r}, [B]_{n*r}] d =0
        :param x: current state
        """
        n = x.shape[0]
        x = x.reshape(n, 1)   # FIXME: do we need this?
        r = self.dim_input
        assert n == self.dim_sys

        # Md= [I_{n},[0]_n,[0]_n,[0]_n,|[0]_{n*r},[0]_{n*r},[0]_{n*r}] d =x
        mat_init_block_zero = np.zeros((n, horizon*(n+r)))
        mat_init_block = np.hstack([np.eye(n), mat_init_block_zero])

        mat_block_a = self.build_block_A(horizon, n, r)
        mat_block_b = self.build_block_b(horizon, n)
        mat_block_eye = self.build_block_eye(horizon, n, r)
        mat_block = mat_block_a + mat_block_b + mat_block_eye
        #
        mat_lhs = np.vstack([mat_init_block, mat_block])
        mat_rhs = np.vstack([x, np.zeros((horizon*(n), 1))])  # equality constraint is w.r.t. x, not u, not n+r
        return mat_lhs, mat_rhs

    def build_block_eye(self, horizon, n, r):
        """
        - build subspace of constraint matrix M with only A involved:
        block(I)=[0_{T*n,n}, -1*kron(I_n, eye(T))  | 0_{T*n, T*r}        ]
        - let horizon change each time, so keep horizon as parameter
        """
        block_zero_a = np.zeros((horizon*n, n))  # shift
        block_eye_subspace = -1*np.kron(np.eye(n), np.eye(horizon))
        block_zero_b = np.zeros((horizon*n, horizon*r))  # shift
        block_eye_full_space = np.hstack(
            [block_zero_a, block_eye_subspace, block_zero_b])
        return block_eye_full_space

    def build_block_b(self, horizon, n):
        """
        - build subspace of constraint matrix M with only A involved:

        block(B)=[0_{T*n,(T+1)*n}| kron(B_{n*r}, eye(T)]
        - let horizon change each time, so keep horizon as parameter
        """
        block_zero_a = np.zeros((horizon*n, (horizon+1)*n))
        block_b_subspace = np.kron(self.mat_input, np.eye(horizon))
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
        block_a_subspace = np.kron(self.mat_sys, np.eye(horizon))
        block_a_fullspace = np.hstack(
            [block_a_subspace, block_zero_a, block_zero_b])
        return block_a_fullspace
