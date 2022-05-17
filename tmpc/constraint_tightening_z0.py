def is_in_minkowski_sum(mat_set_1, mat_set_2):
    """
    x \\in set_1 + set_2
    <=>
    exist x_1 + x_2 = x, s.t. x_1 \\in set_1, x_2\\in set_2
    decision variable x_1, x_2
    constraint:
        x_1 + x_2 = x   # equality constraint
        mat_set_1 x_1 <= 1  # upper bound
        mat_set_2 x_2 <= 1  # upper bound
    ###
    $z_0$ constraint:
    S_k= \minkowski_sum_{i=0:k-1}(A+BK^{s})^i*W
    to judge if s_k \\ in S_k
    (Note S_k is pre-stabalized, by choosing K^{s}, so A+BK^{s} is hurwitz)
    <=>
    there exist w_{0:k-1}, s.t.
    - s_k = \sum_{i=0:k-1}(A+BK^{s})^i*w_i, equality constraint
    - w_i \in W  or mat_disturbance * w_i <= 1

    # state tubes:
    X_0={z_0}+S, X_1={z_1}+S, ..., X_T={z_T}+S  # minkowski sum

    # control tubes:
    {v_0} + K^{s}*S,  {v_1} + K^{s}*S, ..., {v_T} + K^{s}*S,
    #
    #x_k = z_k + s_k \\in X_k \\ in X
    #so s_k = x_k - z_k where z_k is the decision variable
    #
    #s_k = x_k - z_k
    #s_{k} = (A+BK^{s})s_{k-1} + w_{k-1}
    """


