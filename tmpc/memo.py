"""
class responsible for taking the position and control signal during an episode
"""


class MemoTraj():
    """
    use redundant design to ensure the validity of the logged trajectory
    """
    def __init__(self, shape_state, shape_u,
                 name_x='x', name_u='u'):
        self.state_name = name_x
        self.control_name = name_u
        self.set_names = set([name_u, name_x])
        self.shape_state = shape_state
        self.shape_u = shape_u
        self.dict_list_trajectory = None
        self.list_trajectory_dict = None
        self.reset()

    def reset(self):
        """
        clean up
        """
        self.dict_list_trajectory = {self.state_name:[], self.control_name:[]}
        # {"x":[0.9, 1.1, ...], "u":[1.7, 2.5, 0.09, ...]}
        self.list_trajectory_dict = []
        # [{"x":[[0.9]], "u":[1.7, 3]}, {"x":[1.1], "u":2.5}]

    def takedown(self, dict_state):
        """
        @param dict_state: a dictionary
        """
        assert set(dict_state.keys()) == self.set_names
        self.list_trajectory_dict.append(dict_state)
        for key in dict_state.keys():
            self.dict_list_trajectory[key].append(dict_state[key])

    def log_x_u(self, arr_x, arr_u):
        """
        array representing state will be checked for correctness of its shape
        the same for control signal
        """
        len_x = max(arr_x.shape) if len(arr_x.shape)==2 else arr_x.shape[0]
        len_u = max(arr_u.shape) if len(arr_u.shape)==2 else arr_u.shape[0]
        assert len_x == self.shape_state
        assert len_u == self.shape_u
        dict_state = {self.state_name: arr_x, self.control_name:arr_u}
        self.takedown(dict_state)
