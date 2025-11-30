import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
import sys
import os

# ==========================================
# 1. Helper Functions (標準數學函式 - 完全沒動)
# ==========================================
def ECE569_NearZero(z): return abs(z) < 1e-9
def ECE569_VecTose3(V):
    V = np.array(V, dtype=float).flatten()
    if V.size < 6: return np.zeros((4,4))
    omega = V[0:3]; v = V[3:6]
    so3mat = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    return np.vstack((np.column_stack((so3mat, v)), [0, 0, 0, 0]))
def ECE569_se3ToVec(se3mat):
    return np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0], se3mat[0, 3], se3mat[1, 3], se3mat[2, 3]])
def ECE569_TransInv(T):
    R = T[0:3, 0:3]; p = T[0:3, 3]
    return np.vstack((np.column_stack((R.T, -np.dot(R.T, p))), [0, 0, 0, 1]))
def ECE569_Adjoint(T):
    R = T[0:3, 0:3]; p = T[0:3, 3]
    p_skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    return np.vstack((np.column_stack((R, np.zeros((3, 3)))), np.column_stack((np.dot(p_skew, R), R))))
def ECE569_MatrixLog6(T):
    R = T[0:3, 0:3]; p = T[0:3, 3]
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1: return np.vstack((np.column_stack((np.zeros((3, 3)), p)), [0, 0, 0, 0]))
    elif acosinput <= -1:
        if not ECE569_NearZero(1 + R[2, 2]): omg = (1.0 / np.sqrt(2 * (1 + R[2, 2]))) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
        elif not ECE569_NearZero(1 + R[1, 1]): omg = (1.0 / np.sqrt(2 * (1 + R[1, 1]))) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
        else: omg = (1.0 / np.sqrt(2 * (1 + R[0, 0]))) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
        so3mat = np.array([[0, -omg[2], omg[1]], [omg[2], 0, -omg[0]], [-omg[1], omg[0], 0]]) * np.pi
        return np.vstack((np.column_stack((so3mat, p)), [0, 0, 0, 0])) 
    else:
        theta = np.arccos(acosinput)
        so3mat = theta / (2 * np.sin(theta)) * (R - R.T)
        omgmat = so3mat / theta
        G_inv = np.eye(3)/theta - 0.5*omgmat + (1/theta - 0.5/np.tan(theta/2))*np.dot(omgmat,omgmat)
        v = np.dot(G_inv, p)
        return np.vstack((np.column_stack((so3mat, v*theta)), [0, 0, 0, 0]))
def ECE569_MatrixExp6(se3mat):
    se3mat = np.array(se3mat)
    omgtheta = np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]])
    theta = np.linalg.norm(omgtheta)
    if ECE569_NearZero(theta): return np.eye(4) + se3mat
    else:
        omgmat = se3mat[0:3, 0:3] / theta
        I = np.eye(3)
        R = I + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)
        v_vec = se3mat[0:3, 3] / theta
        p = np.dot((I*theta + (1-np.cos(theta))*omgmat + (theta-np.sin(theta))*np.dot(omgmat,omgmat)), v_vec)
        return np.vstack((np.column_stack((R, p)), [0, 0, 0, 1]))
def ECE569_FKinBody(M, Blist, thetalist):
    Blist = np.array(Blist, dtype=float).reshape(6, 6)
    T = np.array(M)
    thetalist = np.array(thetalist).flatten()
    for i in range(len(thetalist)):
        vec = Blist[:, i] * thetalist[i]
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(vec)))
    return T
def ECE569_FKinSpace(M, Slist, thetalist):
    T = np.array(M)
    thetalist = np.array(thetalist).flatten()
    for i in range(len(thetalist) - 1, -1, -1):
        vec = Slist[:, i] * thetalist[i]
        T = np.dot(ECE569_MatrixExp6(ECE569_VecTose3(vec)), T)
    return T
def ECE569_JacobianBody(Blist, thetalist):
    Blist = np.array(Blist, dtype=float)
    Jb = Blist.copy()
    T = np.eye(4)
    thetalist = np.array(thetalist).flatten()
    for i in range(len(thetalist) - 2, -1, -1):
        vec = Blist[:, i + 1] * -thetalist[i + 1]
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(vec)))
        Jb[:, i] = np.dot(ECE569_Adjoint(T), Blist[:, i])
    return Jb
def ECE569_IKinBody(body_screw_axes, M, T, thetalist0, eomg, ev):
    Bmat = np.array(body_screw_axes, dtype=float)
    thetalist = np.array(thetalist0).copy().astype(float).flatten()
    i = 0
    maxiterations = 100 
    Tsb = ECE569_FKinBody(M, Bmat, thetalist)
    T_diff = np.dot(ECE569_TransInv(Tsb), T)
    Vb = ECE569_se3ToVec(ECE569_MatrixLog6(T_diff))
    err = np.linalg.norm(Vb[0:3]) > eomg or np.linalg.norm(Vb[3:6]) > ev
    while err and i < maxiterations:
        Jb = ECE569_JacobianBody(Bmat, thetalist)
        delta_theta = np.dot(np.linalg.pinv(Jb), Vb)
        thetalist = thetalist + delta_theta
        thetalist = thetalist.flatten()
        i += 1
        Tsb = ECE569_FKinBody(M, Bmat, thetalist)
        T_diff = np.dot(ECE569_TransInv(Tsb), T)
        Vb = ECE569_se3ToVec(ECE569_MatrixLog6(T_diff))
        err = np.linalg.norm(Vb[0:3]) > eomg or np.linalg.norm(Vb[3:6]) > ev
    return (thetalist, not err)
def RpToTrans(R, p):
    return np.vstack((np.column_stack((R, p)), [0, 0, 0, 1]))

# ==========================================
# 2. Trajectory Generator (完全沒動)
# ==========================================
def generate_smooth_dino(tf, dt):
    points_raw = np.array([
        [-0.14585, -0.13513],[-0.16000, -0.10117],
        [-0.15503, -0.07007],[-0.12941, -0.05190],
        [-0.09845, -0.05333],[-0.09118, 0.04855],
        [-0.12559, 0.10164],[-0.09271, 0.12843],
        [-0.05486, 0.08825],[-0.06404, 0.14087],
        [-0.05677, 0.15570],[-0.01663, 0.15617],
        [-0.00593, 0.10260],[0.01969, 0.11025],
        [0.00325, 0.14422],[0.01013, 0.15809],
        [0.04301, 0.16000],[0.06022, 0.13417],
        [0.06710, 0.15187],[0.08583, 0.15139],
        [0.10724, 0.12556],[0.09271, 0.09303],
        [0.13477, 0.07438],[0.15771, 0.04233],
        [0.16000, -0.00694],[0.12789, 0.02033],
        [0.09959, 0.01937],[0.09577, -0.01411],
        [0.07665, -0.01794],[0.07627, -0.05477],
        [0.04645, -0.04712],[0.03307, -0.08251],
        [0.00554, -0.05525],[-0.01816, -0.07103],
        [-0.04033, -0.03229],[-0.04836, -0.12126],
        [-0.07551, -0.15904],[-0.11374, -0.16000],
        [-0.125, -0.155], 
        [-0.138, -0.148]
    ])
    
    points_raw[:, 1] = -points_raw[:, 1] 
    points_shifted = np.roll(points_raw, -30, axis=0) 
    points_closed = np.vstack([points_shifted, points_shifted[0]]) 

    center = np.mean(points_closed[:-1], axis=0)
    scale = 1.2
    points = (points_closed - center) * scale

    all_x = []
    all_y = []
    
    for i in range(len(points) - 1):
        p_start = points[i]; p_end = points[i+1]
        dist = np.linalg.norm(p_end - p_start)
        steps = int(max(dist * 1000, 30)) 
        xs = np.linspace(p_start[0], p_end[0], steps)
        ys = np.linspace(p_start[1], p_end[1], steps)
        if i < len(points) - 2:
            all_x.extend(xs[:-1]); all_y.extend(ys[:-1])
        else:
            all_x.extend(xs); all_y.extend(ys)
            
    x_traj = np.array(all_x)
    y_traj = np.array(all_y)
    
    x_traj = gaussian_filter1d(x_traj, sigma=5)
    y_traj = gaussian_filter1d(y_traj, sigma=5)

    total_steps = len(x_traj)
    t = np.linspace(0, tf, total_steps)
    led = np.ones_like(t)
    
    return t, x_traj, y_traj, led

# ==========================================
# 3. Main Script
# ==========================================
def main():
    print("Initializing...")
    M = np.array([[-1.0, 0, 0, 0.4567], [0, 0, 1.0, 0.2232], [0, 1.0, 0, 0.0665], [0, 0, 0, 1.0]])
    S = np.array([[0, 0, 0, 0, 0, 0], [0, 1.0, 1.0, 1.0, 0, 1.0], [1.0, 0, 0, 0, -1.0, 0], 
                  [0, -0.1519, -0.1519, -0.1519, -0.1311, -0.0665], [0, 0, 0, 0, 0.4567, 0], [0, 0, 0.2435, 0.4567, 0, 0.4567]])
    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696], dtype=float)
    M_inv = np.linalg.inv(M)
    Ad_Minv = ECE569_Adjoint(M_inv)
    B = np.zeros((6, 6), dtype=float) 
    for i in range(6): B[:, i] = np.dot(Ad_Minv, S[:, i])
    T0 = ECE569_FKinSpace(M, S, theta0)
    
    draw_center_pos = np.array([0.25, 0.0, 0.25]) 
    R_fixed = T0[0:3, 0:3] 

    dt = 1/500.0
    tf = 7.0 
    print("Generating Smoothed Dino...")
    t, x_rel, y_rel, led = generate_smooth_dino(tf, dt)

    print("Running IK...")
    theta_trajectory = np.zeros((6, len(t)))
    current_guess = theta0.copy()
    eomg = 1e-4; ev = 1e-4
    
    for k in range(len(t)):
        p_absolute = draw_center_pos + np.array([0, x_rel[k], y_rel[k]]) 
        T_target = RpToTrans(R_fixed, p_absolute)
        theta_sol, success = ECE569_IKinBody(B, M, T_target, current_guess, eomg, ev)
        
        if k > 0:
            diff = theta_sol - theta_trajectory[:, k-1]
            for j in range(6):
                if diff[j] > np.pi: theta_sol[j] -= 2*np.pi
                elif diff[j] < -np.pi: theta_sol[j] += 2*np.pi
        
        if not success: theta_trajectory[:, k] = current_guess
        else: theta_trajectory[:, k] = theta_sol; current_guess = theta_sol

    theta_trajectory[:, -1] = theta_trajectory[:, 0] 

    # Plot 1: Difference (強制改 Y 軸顯示範圍)
    print("Plotting Difference...")
    diff_theta = np.diff(theta_trajectory, axis=1)
    plt.figure(figsize=(10, 6))
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE']
    labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    for i in range(6):
        plt.plot(t[:-1], diff_theta[i, :], color=colors[i], label=labels[i], linewidth=1.0)
    
    plt.title('First Order Difference', fontweight='bold')
    plt.xlabel('Time (s)'); plt.ylabel('Difference (rad)')
    plt.ylim([-0.005, 0.005]) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.grid(True, alpha=0.6); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()

    # Plot 2: 3D
    actual_p_trace = np.zeros((3, len(t)))
    for k in range(len(t)):
        T_actual = ECE569_FKinSpace(M, S, theta_trajectory[:, k])
        actual_p_trace[:, k] = T_actual[0:3, 3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(actual_p_trace[0, :], actual_p_trace[1, :], actual_p_trace[2, :], 'b-', linewidth=2, label='Smoothed Dino')
    ax.scatter(actual_p_trace[0, 0], actual_p_trace[1, 0], actual_p_trace[2, 0], c='g', s=60, label='Start')
    ax.set_title('Bonus: Smoothed Dinosaur Trajectory', fontsize=14)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    X = actual_p_trace[0, :]; Y = actual_p_trace[1, :]; Z = actual_p_trace[2, :]
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    ax.view_init(elev=10, azim=0) 
    ax.legend()
    plt.show()

    # Plot 3: Manipulability
    det_Jb = np.zeros(len(t))
    for k in range(len(t)):
        Jb_curr = ECE569_JacobianBody(B, theta_trajectory[:, k])
        det_Jb[k] = np.linalg.det(Jb_curr)
    plt.figure(figsize=(8, 4)); plt.plot(t, det_Jb, color='purple')
    plt.title('Manipulability'); plt.grid(True); plt.show()

    # --- Save CSV ---
    filename = "lo156_bonus.csv" 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    data_to_save = np.column_stack((t, theta_trajectory.T, led))
    np.savetxt(full_path, data_to_save, delimiter=',', fmt='%.6f')
    print(f"CSV 已儲存至: {full_path}")

if __name__ == "__main__":
    main()