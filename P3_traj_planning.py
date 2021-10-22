import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *
# from HW1.utils import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch # Switch occurs at t_final - t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        # Hint: Both self.traj_controller and self.pose_controller have compute_control() functions. 
        #       When should each be called? Make use of self.t_before_switch and 
        #       self.traj_controller.traj_times.
        ########## Code starts here ##########
        if t < self.traj_controller.traj_times[-1] - self.t_before_switch:
            return self.traj_controller.compute_control(x, y, th, t)
        else:
            return self.pose_controller.compute_control(x, y, th, t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    path = np.array(path)
    N = len(path)
    vt = np.zeros(N)
    for i, p in enumerate(path):
        if i == 0:
            continue
        vt[i] = np.linalg.norm(path[i] - path[i-1]) / V_des + vt[i - 1]
    x_spl = splrep(vt, path[:, 0], s=alpha)
    y_spl = splrep(vt, path[:, 1], s=alpha)
    t_smoothed = np.arange(0, vt[-1], dt)
    x_d = splev(t_smoothed, x_spl)
    xd_d = splev(t_smoothed, x_spl, der=1)
    xdd_d = splev(t_smoothed, x_spl, der=2)
    y_d = splev(t_smoothed, y_spl)
    yd_d = splev(t_smoothed, y_spl, der=1)
    ydd_d = splev(t_smoothed, y_spl, der=2)
    theta_d = np.arctan2(yd_d, xd_d)
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    Hint: Take a close look at the code within compute_traj_with_limits() and interpolate_traj() 
          from P1_differential_flatness.py
    """
    ########## Code starts here ##########
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    s_f = State(traj[-1, 0], traj[-1, 1], traj[-1, 3], traj[-1, 2])
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    width = 100
    height = 100
    num_obs = 25
    min_size = 5
    max_size = 30
    np.random.seed(0)
    occupancy, x_init, x_goal = generate_planning_problem(width, height, num_obs, min_size, max_size)
    astar = AStar((0, 0), (width, height), x_init, x_goal, occupancy)
    if not astar.solve():
        print("No path found")
    V_des = 0.3  # Nominal velocity
    alpha = 0.1  # Smoothness parameter
    dt = 0.05
    traj_smoothed, t_smoothed = compute_smoothed_traj(astar.path, V_des, alpha, dt)
    fig = plt.figure()
    astar.plot_path(fig.number)


    def plot_traj_smoothed(traj_smoothed):
        plt.plot(traj_smoothed[:, 0], traj_smoothed[:, 1], color="red", linewidth=2, label="solution path", zorder=10)


    plot_traj_smoothed(traj_smoothed)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
    plt.show()
    # V_max = 0.5  # max speed
    # om_max = 1  # max rotational speed
    # kpx = 10
    # kpy = 10
    # kdx = 10
    # kdy = 10
    # t_new, V_smooth_scaled, om_smooth_scaled, traj_smooth_scaled = modify_traj_with_limits(traj_smoothed, t_smoothed,
    #                                                                                        V_max, om_max, dt)
    # noise_scale = 0.1
    # tf_actual = t_new[-1]
    # times_cl = np.arange(0, tf_actual, dt)
    # s_0 = State(x=x_init[0], y=x_init[1], V=V_max, th=traj_smooth_scaled[0, 2])
    # s_f = State(x=x_goal[0], y=x_goal[1], V=V_max, th=traj_smooth_scaled[-1, 2])
    #
    # actions_ol = np.stack([V_smooth_scaled, om_smooth_scaled], axis=-1)
    # states_ol, ctrl_ol = simulate_car_dyn(s_0.x, s_0.y, s_0.th, times_cl, actions=actions_ol, noise_scale=noise_scale)
    # # states_cl, ctrl_cl = simulate_car_dyn(s_0.x, s_0.y, s_0.th, times_cl, controller=traj_controller,
    # #                                       noise_scale=noise_scale)
    # #
    # fig = plt.figure()
    # astar.plot_path(fig.number)
    # # plot_traj_smoothed(traj_smoothed)
    #
    #
    # def plot_traj_ol(states_ol):
    #     plt.plot(states_ol[:, 0], states_ol[:, 1], color="orange", linewidth=1, label="open-loop path", zorder=10)
    #
    #
    # def plot_traj_cl(states_cl):
    #     plt.plot(states_cl[:, 0], states_cl[:, 1], color="purple", linewidth=1, label="TrajController closed-loop path",
    #              zorder=10)
    #
    #
    # plot_traj_ol(states_ol)
    # # plot_traj_cl(states_cl)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=4)
    # plt.show()