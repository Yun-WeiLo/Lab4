import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. Helper Functions (工具函式)
# ==========================================

def ECE569_VecTose3(V):
    omega = V[0:3]; v = V[3:6]
    so3mat = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    return np.vstack((np.column_stack((so3mat, v)), [0, 0, 0, 0]))

def ECE569_MatrixExp6(se3mat):
    se3mat = np.array(se3mat)
    omgtheta = np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]])
    theta = np.linalg.norm(omgtheta)
    if theta < 1e-6:
        return np.eye(4) + se3mat
    else:
        omgmat = se3mat[0:3, 0:3] / theta
        I = np.eye(3)
        R = I + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)
        v_vec = se3mat[0:3, 3] / theta
        term1 = I * theta
        term2 = (1 - np.cos(theta)) * omgmat
        term3 = (theta - np.sin(theta)) * np.dot(omgmat, omgmat)
        p = np.dot((term1 + term2 + term3), v_vec)
        return np.vstack((np.column_stack((R, p)), [0, 0, 0, 1]))

def ECE569_FKinSpace(M, Slist, thetalist):
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        se3 = ECE569_VecTose3(Slist[:, i] * thetalist[i])
        T = np.dot(ECE569_MatrixExp6(se3), T)
    return T

# ==========================================
# 2. Main Script for Problem 2(d)
# ==========================================

def main():
    # --- A. 準備機器人參數 ---
    M = np.array([[-1.0, 0, 0, 0.4567], [0, 0, 1.0, 0.2232], [0, 1.0, 0, 0.0665], [0, 0, 0, 1.0]])
    S = np.array([[0, 0, 0, 0, 0, 0], [0, 1.0, 1.0, 1.0, 0, 1.0], [1.0, 0, 0, 0, -1.0, 0], 
                  [0, -0.1519, -0.1519, -0.1519, -0.1311, -0.0665], [0, 0, 0, 0, 0.4567, 0], [0, 0, 0.2435, 0.4567, 0, 0.4567]])
    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])

    # 計算初始位置 T0 (Base Frame)
    T0 = ECE569_FKinSpace(M, S, theta0)
    print("Calculated T0:\n", np.round(T0, 4))

    # --- B. 產生 Task 1 的軌跡資料 ---
    # 參數設定 (與 Task 1 一致)
    A = 0.15; B = 0.12; a = 1; b = 2
    tf = 10.0; dt = 1/500.0
    t = np.arange(0, tf + dt, dt)
    
    # 為了簡化，我們直接用線性插值或簡單的 S-curve 產生 alpha (確保形狀正確即可)
    # alpha 從 0 走到 2*pi
    alpha = np.linspace(0, 2*np.pi, len(t))
    
    # 計算 Lissajous 軌跡 x(t), y(t)
    x_traj = A * np.sin(a * alpha)
    y_traj = B * np.sin(b * alpha)

    # --- C. 計算每一個時間點的 T_sb(t) 與 p(t) ---
    # 儲存路徑點 p(t) = [x, y, z]
    p_trace = np.zeros((3, len(t)))

    for i in range(len(t)):
        # 1. 建構 T_d(t)
        # pd(t) = [x(t), y(t), 0]
        # Rd(t) = Identity (題目說可以選 I)
        pd = np.array([x_traj[i], y_traj[i], 0])
        Td = np.eye(4)
        Td[0:3, 3] = pd
        
        # 2. 計算 T_sb(t) = T0 * Td(t)
        Tsb = np.dot(T0, Td)
        
        # 3. 提取位置 p(t)
        p_trace[:, i] = Tsb[0:3, 3]

    # --- D. 3D 繪圖 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 畫出軌跡 (藍色線)
    ax.plot(p_trace[0, :], p_trace[1, :], p_trace[2, :], 'b-', linewidth=2, label='Trajectory p(t)')

    # 標記起點 (綠色圓圈) - Start
    ax.scatter(p_trace[0, 0], p_trace[1, 0], p_trace[2, 0], c='g', marker='o', s=100, label='Start', depthshade=False)

    # 標記終點 (紅色X) - End
    ax.scatter(p_trace[0, -1], p_trace[1, -1], p_trace[2, -1], c='r', marker='x', s=100, label='End', depthshade=False)

    # 設定標籤與標題
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Problem 2(d): End-Effector Trajectory in Base Frame {s}')
    ax.legend()
    
    # 設定相同的比例尺 (Equal Aspect Ratio) 讓 8 字形不會變形
    # Python 3D plot 的 equal aspect ratio 比較麻煩，這是簡易解法：
    max_range = np.array([p_trace[0].max()-p_trace[0].min(), p_trace[1].max()-p_trace[1].min(), p_trace[2].max()-p_trace[2].min()]).max() / 2.0
    mid_x = (p_trace[0].max()+p_trace[0].min()) * 0.5
    mid_y = (p_trace[1].max()+p_trace[1].min()) * 0.5
    mid_z = (p_trace[2].max()+p_trace[2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()