.
├── check_coverage.sh  # check coverage of this package
├── conftest.py
├── demos
│   ├── demo2_plot_poly_feasible.py
│   ├── demo_linprog.py
│   ├── demo_plot2d_poly_feasible_raw.py
│   └── demo_plot_poly_feasible.py
├── file_overview.md  (this file you are reading)
├── README.md
├── requirements.txt
├── tests
│   ├── __pycache__
│   │   ├── test_constraint.cpython-39-pytest-7.1.2.pyc
│   │   ├── test_exp2.cpython-39-pytest-7.1.2.pyc
│   │   ├── test_exp.cpython-39-pytest-7.1.2.pyc
│   │   ├── test_mpc.cpython-39-pytest-7.1.2.pyc
│   │   ├── test_pos_inva.cpython-39-pytest-7.1.2.pyc
│   │   ├── test_qp_mpc_tube.cpython-39-pytest-7.1.2.pyc
│   │   └── test_s_infinity_alpha_j.cpython-39-pytest-7.1.2.pyc
│   ├── test_constraint.py
│   ├── test_exp2.py
│   ├── test_exp.py
│   ├── test_mpc.py
│   ├── test_pos_inva.py
│   ├── test_qp_mpc_tube.py
│   └── test_s_infinity_alpha_j.py
└── tmpc
    ├── block_lqr_loss.py
    ├── constraint_block_horizon_stage_x_u.py
    ├── constraint_block_horizon_terminal.py
    ├── constraint_eq_ldyn_1_terminal.py
    ├── constraint_eq_ldyn.py
    ├── constraint_pos_inva_terminal.py
    ├── constraint_s_inf.py
    ├── constraint_stage_interface.py
    ├── constraint_tightening.py
    ├── constraint_tightening_z0w.py
    ├── constraint_tightening_z_terminal.py
    ├── constraint_x_u_couple.py
    ├── dyn_sys.py   # A, B, STATE feedback
    ├── __init__.py
    ├── loss_terminal.py   # Lyapunov
    ├── memo.py   # log trajectory
    ├── mpc_qp.py  # mpc without tube
    ├── mpc_qp_tube.py  # mpc with tube
    ├── riccati.py  # riccati equation to find K_w and K_z
    ├── simulate.py #  Exp class that combines simulator of dynamic system and controller
    ├── solver_quadprog.py  # QPsolver
    ├── support_decomp.py  # decomposition
    ├── support_fun.py  # support function of a set
    ├── support_set_inclusion.py  # check set inclusion
    ├── utils_case2.py  # examples on system A, B matrix
    ├── utils_case.py
    ├── utils_plot_constraint.py  # matplotlib for constraint visurualization
    └── utils_scenario.py  # Not used yet

6 directories, 78 files
