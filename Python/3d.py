import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. Helper Functions (數學核心)
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
    maxiterations = 20
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

# ==========================================
# 2. Main Script
# ==========================================
def main():
    print("Initializing...")
    # 參數設定
    M = np.array([[-1.0, 0, 0, 0.4567], [0, 0, 1.0, 0.2232], [0, 1.0, 0, 0.0665], [0, 0, 0, 1.0]])
    S = np.array([[0, 0, 0, 0, 0, 0], [0, 1.0, 1.0, 1.0, 0, 1.0], [1.0, 0, 0, 0, -1.0, 0], 
                  [0, -0.1519, -0.1519, -0.1519, -0.1311, -0.0665], [0, 0, 0, 0, 0.4567, 0], [0, 0, 0.2435, 0.4567, 0, 0.4567]])
    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696], dtype=float)

    # 計算 B
    M_inv = np.linalg.inv(M)
    Ad_Minv = ECE569_Adjoint(M_inv)
    B = np.zeros((6, 6), dtype=float) 
    for i in range(6): B[:, i] = np.dot(Ad_Minv, S[:, i])

    # 計算 T0
    T0 = ECE569_FKinSpace(M, S, theta0)

    # 產生軌跡
    Amp_x = 0.15; Amp_y = 0.08; a = 1; b = 2
    tf = 5.0; dt = 1/500.0
    t = np.arange(0, tf + dt, dt)
    alpha = np.linspace(0, 2*np.pi, len(t))
    x_traj = Amp_x * np.sin(a * alpha)
    y_traj = Amp_y * np.sin(b * alpha)

    # 執行 IK
    print(f"Running IK for {len(t)} steps...")
    theta_trajectory = np.zeros((6, len(t)))
    current_guess = theta0.copy()
    eomg = 1e-6; ev = 1e-6

    for k in range(len(t)):
        p_d = np.array([x_traj[k], y_traj[k], 0])
        T_d = np.eye(4); T_d[0:3, 3] = p_d
        T_target = np.dot(T0, T_d)
        
        theta_sol, success = ECE569_IKinBody(B, M, T_target, current_guess, eomg, ev)
        theta_trajectory[:, k] = theta_sol
        current_guess = theta_sol

    # 執行 FK 驗證
    print("Verifying Trajectory using FK...")
    actual_p_trace = np.zeros((3, len(t)))
    for k in range(len(t)):
        T_actual = ECE569_FKinSpace(M, S, theta_trajectory[:, k])
        actual_p_trace[:, k] = T_actual[0:3, 3]

    # --- 繪圖 (Example Solution Style) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 畫軌跡 (Blue line)
    ax.plot(actual_p_trace[0, :], actual_p_trace[1, :], actual_p_trace[2, :], 'b-', linewidth=2, label='p(t)')
    
    # 標記起點 (Green dot)
    ax.scatter(actual_p_trace[0, 0], actual_p_trace[1, 0], actual_p_trace[2, 0], c='g', marker='o', s=50, label='start', depthshade=False)
    
    # 標記終點 (Red x)
    ax.scatter(actual_p_trace[0, -1], actual_p_trace[1, -1], actual_p_trace[2, -1], c='r', marker='x', s=50, label='end', depthshade=False)

    ax.set_title('Verified Trajectory in s frame', fontsize=14, y=1.02)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    # 【關鍵：強制設定 Box Aspect Ratio (1:1:1)】
    # 這會讓圖形呈現真實的物理比例 (扁平的 8 字形)，而不會被拉成正方體
    X = actual_p_trace[0, :]; Y = actual_p_trace[1, :]; Z = actual_p_trace[2, :]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 強制扁平
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))

    # 設定視角 (根據 Example Solution 調整)
    # elev=20, azim=-45 是最接近截圖的標準視角
    ax.view_init(elev=20, azim=-45)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()