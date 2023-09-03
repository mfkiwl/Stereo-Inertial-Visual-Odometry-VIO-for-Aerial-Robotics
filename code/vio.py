#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()

    R = Rotation.as_matrix(q)
    new_p = p + v * dt + 0.5 * (R @ (a_m - a_b) + g) * dt ** 2
    new_v = v + (R @ (a_m - a_b) + g) * dt
    q_new = Rotation.from_rotvec(((w_m - w_b) * dt).flatten())
    new_q = q * q_new

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    F_x = np.zeros((18, 18))
    Q_i = np.zeros((12, 12))
    F_i = np.zeros((18, 12))
    R = Rotation.as_matrix(q)

    F_x[0:3, 0:3] = np.identity(3)
    F_x[0:3, 3:6] = np.identity(3) * dt
    F_x[3:6, 3:6] = np.identity(3)
    a_diff_vec = a_m - a_b
    a_skew = np.array([[0, -a_diff_vec[2], a_diff_vec[1]],
                       [a_diff_vec[2], 0, -a_diff_vec[0]],
                       [-a_diff_vec[1], a_diff_vec[0], 0]], dtype=object)

    F_x[3:6, 6:9] = - (R @ a_skew) * dt
    F_x[3:6, 9:12] = -R * dt
    F_x[3:6, 15:] = np.identity(3) * dt
    rot_f_vec = Rotation.from_rotvec(((w_m - w_b) * dt).flatten())
    F_x[6:9, 6:9] = rot_f_vec.as_matrix().T
    F_x[6:9, 12:15] = - np.identity(3) * dt
    F_x[9:12, 9:12] = np.identity(3)
    F_x[12:15, 12:15] = np.identity(3)
    F_x[15:, 15:] = np.identity(3)

    F_i[3:6, 0:3] = np.identity(3)
    F_i[6:9, 3:6] = np.identity(3)
    F_i[9:12, 6:9] = np.identity(3)
    F_i[12:15, 9:] = np.identity(3)

    v_i = (accelerometer_noise_density ** 2) * (dt ** 2) * np.identity(3)
    theta_i = (gyroscope_noise_density ** 2) * (dt ** 2) * np.identity(3)
    A_i = (accelerometer_random_walk ** 2) * dt * np.identity(3)
    omega_i = (gyroscope_random_walk ** 2) * dt * np.identity(3)
    Q_i[0:3, 0:3] = v_i
    Q_i[3:6, 3:6] = theta_i
    Q_i[6:9, 6:9] = A_i
    Q_i[9:, 9:] = omega_i

    P = (F_x @ error_state_covariance @ F_x.T) + (F_i @ Q_i @ F_i.T)
    return P

    # return an 18x18 covariance matrix
    # return np.identity(18)


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    innovation = np.zeros((2, 1))
    ni = 0
    R = Rotation.as_matrix(q)
    Pc = R.T @ (Pw - p)
    innovation = uv - (Pc[0:2] / Pc[2]).reshape(-1, 1)
    if norm(innovation) < error_threshold:
        z_t = Pc[0:2] / Pc[2] + ni

        d_z_c = 1 / Pc[2] * np.array([[1, 0, -float(z_t[0])],
                                      [0, 1, -float(z_t[1])]], dtype=float)

        P_c_0 = R.T @ (Pw - p)
        d_P_theta = np.array([[0, -float(P_c_0[2]), float(P_c_0[1])],
                              [float(P_c_0[2]), 0, -float(P_c_0[0])],
                              [float(-P_c_0[1]), float(P_c_0[0]), 0]], dtype=float)
        d_P_p = - R.T

        d_z_theta = d_z_c @ d_P_theta
        d_z_p = d_z_c @ d_P_p

        H = np.zeros((2, 18))
        H[0:2, 0:3] = d_z_p
        H[0:2, 3:6] = np.zeros((2, 3))
        H[0:2, 6:9] = d_z_theta
        H[0:2, 9:18] = np.zeros((2, 9))
        # print()
        K = error_state_covariance @ H.T @ \
            np.linalg.inv((H @ error_state_covariance @ H.T) + Q)
        error_state_covariance = (np.identity(18) - K @ H) @ error_state_covariance @ (np.identity(18) - K @ H).T \
                                 + (K @ Q @ K.T)
        delta_x = K @ innovation
        delta_p = delta_x[0:3]
        delta_v = delta_x[3:6]
        delta_theta = delta_x[6:9]
        delta_a_b = delta_x[9:12]
        delta_w_b = delta_x[12:15]
        delta_g = delta_x[15:18]

        p = p + delta_p
        v = v + delta_v
        q = Rotation.from_matrix(R @ (Rotation.from_rotvec(delta_theta.flatten()).as_matrix()))
        a_b = a_b + delta_a_b
        w_b = w_b + delta_w_b
        g = g + delta_g
    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
