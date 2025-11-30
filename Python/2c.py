import numpy as np

# ==========================================
# 1. Helper Functions (工具函式 - 數學部分不變)
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

def ECE569_Adjoint(T):
    R = T[0:3, 0:3]; p = T[0:3, 3]
    p_skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    return np.vstack((np.column_stack((R, np.zeros((3, 3)))), np.column_stack((np.dot(p_skew, R), R))))

def ECE569_FKinSpace(M, Slist, thetalist):
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        se3 = ECE569_VecTose3(Slist[:, i] * thetalist[i])
        T = np.dot(ECE569_MatrixExp6(se3), T)
    return T

def ECE569_FKinBody(M, Blist, thetalist):
    T = np.array(M)
    for i in range(len(thetalist)):
        se3 = ECE569_VecTose3(Blist[:, i] * thetalist[i])
        T = np.dot(T, ECE569_MatrixExp6(se3))
    return T

# ==========================================
# 2. MATLAB 風格輸出函式 (視覺整形)
# ==========================================
def print_matlab_style(name, matrix, scale_factor=None):
    print(f"\n{name} = 4x4")
    
    # 如果有指定縮放因子 (例如 1.0e-15)
    if scale_factor:
        print(f"   {scale_factor:.1e} *")
        display_matrix = matrix / scale_factor
    else:
        display_matrix = matrix

    rows, cols = display_matrix.shape
    for i in range(rows):
        row_str = "   "
        for j in range(cols):
            # 模仿 MATLAB 的對齊和精度
            val = display_matrix[i, j]
            if scale_factor:
                # 差異矩陣顯示格式 (小數點後4位)
                row_str += f"{val:10.4f} "
            else:
                # 一般矩陣顯示格式 (T0)
                if abs(val) < 1e-9: val = 0.0 # 清理 -0.0000
                row_str += f"{val:10.4f} "
        print(row_str)

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # 參數設定
    M = np.array([[-1.0000, 0, 0, 0.4567], [0, 0, 1.0000, 0.2232], [0, 1.0000, 0, 0.0665], [0, 0, 0, 1.0000]])
    S = np.array([[0, 0, 0, 0, 0, 0], [0, 1.0000, 1.0000, 1.0000, 0, 1.0000], [1.0000, 0, 0, 0, -1.0000, 0], 
                  [0, -0.1519, -0.1519, -0.1519, -0.1311, -0.0665], [0, 0, 0, 0, 0.4567, 0], [0, 0, 0.2435, 0.4567, 0, 0.4567]])
    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])

    # 計算 B
    M_inv = np.linalg.inv(M)
    Ad_Minv = ECE569_Adjoint(M_inv)
    B = np.zeros_like(S)
    for i in range(6): B[:, i] = np.dot(Ad_Minv, S[:, i])

    # 計算 FK
    T0_Space = ECE569_FKinSpace(M, S, theta0)
    T0_Body = ECE569_FKinBody(M, B, theta0)
    Difference = T0_Space - T0_Body

    # === 輸出部分 (模仿截圖) ===
    
    # 印出 T0_Space
    print_matlab_style("T0_space", T0_Space)
    
    # 印出 T0_Body
    print_matlab_style("T0_body", T0_Body)

    # 印出 Difference Matrix (模仿 MATLAB 的 1.0e-15 * 格式)
    # 強制提取 1e-15 作為係數，讓你看起來像 MATLAB
    print_matlab_style("ans", Difference, scale_factor=1e-15)

if __name__ == "__main__":
    main()