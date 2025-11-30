import matplotlib.pyplot as plt
import numpy as np
import math

'''
*** BASIC HELPER FUNCTIONS ***
'''

def ECE569_NearZero(z):
    return abs(z) < 1e-6

def ECE569_Normalize(V):
    return V / np.linalg.norm(V)

'''
*** CHAPTER 3: RIGID-BODY MOTIONS ***
'''

def ECE569_RotInv(R):
    return np.array(R).T

def ECE569_VecToso3(omg):
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def ECE569_so3ToVec(so3mat):
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def ECE569_AxisAng3(expc3):
    return (ECE569_Normalize(expc3), np.linalg.norm(expc3))

def ECE569_MatrixExp3(so3mat):
    omgtheta = ECE569_so3ToVec(so3mat)
    if ECE569_NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = ECE569_AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def ECE569_MatrixLog3(R):
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not ECE569_NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not ECE569_NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return ECE569_VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def ECE569_RpToTrans(R, p):
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def ECE569_TransToRp(T):
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def ECE569_TransInv(T):
    """Inverts a homogeneous transformation matrix"""
    R, p = ECE569_TransToRp(T)
    Rt = np.array(R).T
    # Formula: [R^T, -R^T*p; 0, 1]
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def ECE569_VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3"""
    # V = [omega, v]
    return np.r_[np.c_[ECE569_VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]], np.zeros((1, 4))]

def ECE569_se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector"""
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]], [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

def ECE569_Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation matrix"""
    R, p = ECE569_TransToRp(T)
    # Formula: [R, 0; [p]R, R]
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(ECE569_VecToso3(p), R), R]]

def ECE569_MatrixExp6(se3mat):
    """Computes the matrix exponential of an se3 representation"""
    se3mat = np.array(se3mat)
    omgtheta = ECE569_so3ToVec(se3mat[0: 3, 0: 3])
    
    if ECE569_NearZero(np.linalg.norm(omgtheta)):
        return np.eye(4) + se3mat
    else:
        theta = ECE569_AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        # Calculate G(theta)v term
        I = np.eye(3)
        term1 = I * theta
        term2 = (1 - np.cos(theta)) * omgmat
        term3 = (theta - np.sin(theta)) * np.dot(omgmat, omgmat)
        
        # Upper left is Rodrigues formula for rotation
        R = np.eye(3) + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)
        
        # Top right p = (term1 + term2 + term3) * v / theta
        v_vec = se3mat[0:3, 3] / theta
        p = np.dot((term1 + term2 + term3), v_vec)
        
        return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def ECE569_MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix"""
    R, p = ECE569_TransToRp(T)
    omgmat = ECE569_MatrixLog3(R)
    
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)), [T[0][3], T[1][3], T[2][3]]], [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        omgmat = omgmat / theta
        I = np.eye(3)
        # G_inv(theta) calculation
        G_inv = I / theta - 0.5 * omgmat + (1 / theta - 0.5 / np.tan(theta / 2)) * np.dot(omgmat, omgmat)
        v = np.dot(G_inv, p)
        # Construct [S]theta
        return np.r_[np.c_[omgmat * theta, v * theta], [[0, 0, 0, 0]]]


'''
*** CHAPTER 4: FORWARD KINEMATICS ***
'''

def ECE569_FKinBody(M, Blist, thetalist):
    """Computes forward kinematics in the body frame"""
    T = np.array(M)
    for i in range(len(thetalist)):
        # Formula: M * e^[B1]theta1 * ... * e^[Bn]thetan
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(np.array(Blist)[:, i] * thetalist[i])))
    return T

def ECE569_FKinSpace(M, Slist, thetalist):
    """Computes forward kinematics in the space frame"""
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        # Formula: e^[S1]theta1 * ... * e^[Sn]thetan * M
        T = np.dot(ECE569_MatrixExp6(ECE569_VecTose3(np.array(Slist)[:, i] * thetalist[i])), T)
    return T

'''
*** CHAPTER 5: VELOCITY KINEMATICS AND STATICS***
'''

def ECE569_JacobianBody(Blist, thetalist):
    """Computes the body Jacobian for an open chain robot"""
    Jb = np.array(Blist).copy().astype(float)
    T = np.eye(4)
    # Filling from n-1 down to 0 (right to left columns)
    # J_b_i = Ad_{e^-[B_n]theta_n ... e^-[B_i+1]theta_i+1} (B_i)
    # But usually simpler to iterate:
    for i in range(len(thetalist) - 2, -1, -1):
        # Update T with the transformation of the NEXT joint (i+1)
        # We need T = e^-[B_n]theta_n * ... * e^-[B_i+1]theta_i+1
        # The loop starts at i = n-2 (second to last col).
        # We apply the transformation of joint i+1 to T.
        # Note: Blist needs to be negated because we are moving "inwards" towards tool frame logic
        # T_incremental = e^(-[B_{i+1}] * theta_{i+1})
        vec = np.array(Blist)[:, i + 1] * -thetalist[i + 1]
        T = np.dot(T, ECE569_MatrixExp6(ECE569_VecTose3(vec)))
        
        # Transform the current B_i using the accumulated Adjoint T
        Jb[:, i] = np.dot(ECE569_Adjoint(T), np.array(Blist)[:, i])
    return Jb

'''
*** CHAPTER 6: INVERSE KINEMATICS ***
'''

def ECE569_IKinBody(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame"""
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    
    # Calculate current configuration Tsb
    Tsb = ECE569_FKinBody(M, Blist, thetalist)
    # Calculate twist Vb = log(Tsb^-1 * Tsd)
    Vb = ECE569_se3ToVec(ECE569_MatrixLog6(np.dot(ECE569_TransInv(Tsb), T)))
    
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
          
    while err and i < maxiterations:
        # Calculate Jacobian
        Jb = ECE569_JacobianBody(Blist, thetalist)
        # Update theta: theta = theta + pinv(Jb) * Vb
        thetalist = thetalist + np.dot(np.linalg.pinv(Jb), Vb)
        
        i += 1
        # Recalculate Error
        Tsb = ECE569_FKinBody(M, Blist, thetalist)
        Vb = ECE569_se3ToVec(ECE569_MatrixLog6(np.dot(ECE569_TransInv(Tsb), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
              
    return (thetalist, not err)

# the ECE569_normalized trapezoid function
def g(t, T, ta):
    if t < 0 or t > T:
        return 0
    
    if t < ta:
        return (T/(T-ta))* t/ta
    elif t > T - ta:
        return (T/(T-ta))*(T - t)/ta
    else:
        return (T/(T-ta))
    
def trapezoid(t, T, ta):
    return g(t, T, ta)

def main():

    ### Step 1: Trajectory Generation

    # Lissaous curve definition
    T_period = 2*np.pi
    t_dummy = np.linspace(0, T_period, 1000)
    # Using the suggested Lissajous parameters
    xd = 0.16*np.sin(t_dummy)
    yd = 0.08*np.sin(2*t_dummy)

    # calculate the arc length
    d = 0
    for i in range(1, len(t_dummy)):
        dist = np.sqrt((xd[i] - xd[i-1])**2 + (yd[i] - yd[i-1])**2)
        d += dist
    
    print(f"Total path length: {d:.4f} m")

    # tfinal: Choose a reasonable time. Example: 5 seconds
    tfinal = 5.0
    
    # calculate average velocity
    c = d/tfinal
    print(f"Average velocity: {c:.4f} m/s")

    # forward euler to calculate alpha (path parameter mapping)
    dt = 0.01 # Smaller step for smoother simulation
    t = np.arange(0, tfinal, dt)
    alpha = np.zeros(len(t))
    
    # We need the derivative of the geometric path wrt the parameter t_dummy (let's call it gamma)
    # x = 0.16 sin(gamma), x' = 0.16 cos(gamma)
    # y = 0.08 sin(2gamma), y' = 0.16 cos(2gamma)
    
    # Initial state
    current_gamma = 0
    alpha[0] = 0 # Gamma starts at 0

    for i in range(1, len(t)):
        # Calculate geometric speed at current gamma: || dr/dgamma ||
        dx_dgamma = 0.16 * np.cos(current_gamma)
        dy_dgamma = 0.16 * np.cos(2 * current_gamma)
        geom_speed = np.sqrt(dx_dgamma**2 + dy_dgamma**2)
        
        # Desired velocity v(t) from trapezoid profile
        v_des = c * trapezoid(t[i], tfinal, tfinal/5.0) # Using ta = T/5
        
        # Equation: d(gamma)/dt = v(t) / || dr/dgamma ||
        # Euler integration: gamma_new = gamma_old + (dgamma/dt) * dt
        dgamma = (v_des / geom_speed) * dt
        current_gamma += dgamma
        alpha[i] = current_gamma

    # plot alpha vs t
    plt.plot(t, alpha,'b-',label='alpha (path param)')
    # Note: Alpha reaches 2*pi at the end ideally
    plt.plot(t, np.ones(len(t))*T_period, 'k--',label='2pi')
    plt.xlabel('t')
    plt.ylabel('alpha')
    plt.title('alpha vs t')
    plt.legend()
    plt.grid()
    plt.show()

    # Rescale our trajectory with alpha
    x = 0.16*np.sin(alpha)
    y = 0.08*np.sin(2*alpha)

    # calculate velocity
    xdot = np.diff(x)/dt
    ydot = np.diff(y)/dt
    v = np.sqrt(xdot**2 + ydot**2)

    # plot velocity vs t
    plt.plot(t[1:], v, 'b-',label='velocity')
    plt.plot(t[1:], np.ones(len(t[1:]))*c, 'k--',label='average velocity')
    # Limit line just for viz
    plt.plot(t[1:], np.ones(len(t[1:]))*0.25, 'r--',label='velocity limit') 
    plt.xlabel('t')
    plt.ylabel('velocity')
    plt.title('velocity vs t')
    plt.legend()
    plt.grid()
    plt.show()

    ### Step 2: Forward Kinematics
    L1 = 0.2435
    L2 = 0.2132
    W1 = 0.1311
    W2 = 0.0921
    H1 = 0.1519
    H2 = 0.0854

    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]])
    
    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -H1, 0, 0])
    S3 = np.array([0, 1, 0, -H1, 0, L1])
    S4 = np.array([0, 1, 0, -H1, 0, L1 + L2])
    S5 = np.array([0, 0, -1, -W1, L1+L2, 0])
    S6 = np.array([0, 1, 0, H2-H1, 0, L1+L2])
    S = np.array([S1, S2, S3, S4, S5, S6]).T
    
    B1 = np.linalg.inv(ECE569_Adjoint(M))@S1
    B2 = np.linalg.inv(ECE569_Adjoint(M))@S2
    B3 = np.linalg.inv(ECE569_Adjoint(M))@S3
    B4 = np.linalg.inv(ECE569_Adjoint(M))@S4
    B5 = np.linalg.inv(ECE569_Adjoint(M))@S5
    B6 = np.linalg.inv(ECE569_Adjoint(M))@S6
    B = np.array([B1, B2, B3, B4, B5, B6]).T

    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])
    
    # perform forward kinematics using ECE569_FKinSpace and ECE569_FKinBody
    T0_space = ECE569_FKinSpace(M, S, theta0)
    print(f'T0_space:\n{T0_space}')
    T0_body = ECE569_FKinBody(M, B, theta0)
    print(f'T0_body:\n{T0_body}')
    T0_diff = T0_space - T0_body
    print(f'T0_diff norm: {np.linalg.norm(T0_diff)}')
    T0 = T0_body

    # calculate Tsd for each time step
    # We assume the Lissajous curve is a translation offset relative to the STARTING pose T0
    # or relative to the base frame but starting at T0's position.
    # Given the small magnitude (0.16m), it's likely an offset added to T0's position.
    
    Tsd = np.zeros((4, 4, len(t)))
    # Starting position
    p0 = T0[0:3, 3]
    
    for i in range(len(t)):
        # Create a copy of the starting pose
        T_target = T0.copy()
        
        # Update the position (adding the Lissajous offset)
        # Assuming the drawing is in the X-Y plane of the SPACE frame
        # T_target[0, 3] = p0[0] + x[i]
        # T_target[1, 3] = p0[1] + y[i]
        
        # ALTERNATIVE: If the drawing is in the Body frame (End Effector Frame)
        # T_target = T0 @ Trans(x, y, 0)
        # Given standard lab tasks, it's usually in the base frame, but relative to start.
        # Let's assume absolute Space Frame offsets relative to start:
        
        T_target[0, 3] = p0[0] + x[i] # Add offset to X
        T_target[1, 3] = p0[1] + y[i] # Add offset to Y
        
        Tsd[:, :, i] = T_target
        
    # plot p(t) vs t in the {s} frame
    xs = Tsd[0, 3, :]
    ys = Tsd[1, 3, :]
    zs = Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_title('Trajectory in s frame')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    plt.show()

    ### Step 3: Inverse Kinematics

    # when i=0
    thetaAll = np.zeros((6, len(t)))

    initialguess = theta0
    eomg = 1e-4 # Relaxed tolerance slightly for stability
    ev = 1e-4

    thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,0], initialguess, eomg, ev)
    if not success:
        print(f"Warning: Convergence failed at index 0. Error: {success}")
        # Proceeding with best guess anyway for visualization
    
    thetaAll[:, 0] = thetaSol

    # when i=1...,N-1
    for i in range(1, len(t)):
        # use previous solution as current guess
        initialguess = thetaAll[:, i-1]

        # calculate thetaSol for Tsd[:,:,i] with initial guess
        thetaSol, success = ECE569_IKinBody(B, M, Tsd[:,:,i], initialguess, eomg, ev)
        
        if not success:
            print(f'Failed to find a solution at index {i}')
            # Keep previous valid as fail-safe or break
            thetaSol = thetaAll[:, i-1]
            
        thetaAll[:, i] = thetaSol

    # verify that the joint angles don't change much
    dj = np.diff(thetaAll, axis=1)
    plt.plot(t[1:], dj[0], 'b-',label='joint 1')
    plt.plot(t[1:], dj[1], 'g-',label='joint 2')
    plt.plot(t[1:], dj[2], 'r-',label='joint 3')
    plt.plot(t[1:], dj[3], 'c-',label='joint 4')
    plt.plot(t[1:], dj[4], 'm-',label='joint 5')
    plt.plot(t[1:], dj[5], 'y-',label='joint 6')
    plt.xlabel('t (seconds)')
    plt.ylabel('first order difference')
    plt.title('Joint angles first order difference')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # verify that the joint angles will trace out our trajectory
    actual_Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        # use forward kinematics to calculate Tsd from our thetaAll
        actual_Tsd[:,:,i] = ECE569_FKinSpace(M, S, thetaAll[:, i])
    
    xs = actual_Tsd[0, 3, :]
    ys = actual_Tsd[1, 3, :]
    zs = actual_Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Verified Trajectory in s frame')
    ax.legend()
    plt.show()
    
    # (3e) verify the robot does not enter kinematic singularity
    # by plotting the determinant of the body jacobian
    body_dets = np.zeros(len(t))
    for i in range(len(t)):
        # Calculate Jacobian at current configuration
        Jb_curr = ECE569_JacobianBody(B, thetaAll[:, i])
        # Determinant (for square J) or sqrt(det(J*J.T))
        # Since Jb is 6x6, we can take det directly
        body_dets[i] = np.linalg.det(Jb_curr)
        
    plt.plot(t, body_dets, '-')
    plt.xlabel('t (seconds)')
    plt.ylabel('det of J_B')
    plt.title('Manipulability')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # save to csv file
    led = np.ones_like(t)
    data = np.column_stack((t, thetaAll.T, led))
    np.savetxt('ldihel.csv', data, delimiter=',')
    print("Data saved to ldihel.csv")


if __name__ == "__main__":
    main()