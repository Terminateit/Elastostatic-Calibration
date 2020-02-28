import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.patches as mpatches

# np.set_printoptions(precision=2, suppress=True, linewidth=200)

def trotx(q):
    Tx = np.array([[1,         0,          0, 0],
                   [0, np.cos(q), -np.sin(q), 0],
                   [0, np.sin(q),  np.cos(q), 0],
                   [0,         0,          0, 1]], dtype = float)
    return Tx

def troty(q):
    Ty = np.array([[ np.cos(q), 0, np.sin(q),  0],
                   [         0, 1,         0,  0],
                   [-np.sin(q), 0, np.cos(q),  0],
                   [         0, 0,         0,  1]], dtype = float)
    return Ty

def trotz(q):
    Tz = np.array([[np.cos(q), -np.sin(q),   0, 0],
                   [np.sin(q),  np.cos(q),   0, 0],
                   [        0,            0, 1, 0],
                   [        0,            0, 0, 1]], dtype = float)
    return Tz

def translx(q):
    Tx = np.array([[1, 0, 0, q],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype = float)
    return Tx

def transly(q):
    Ty = np.array([[1, 0, 0, 0],
                   [0, 1, 0, q],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype = float)
    return Ty

def translz(q):
    Tz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, q],
                   [0, 0, 0, 1]], dtype = float)
    return Tz

def dtrotx(q):
    dTx = np.array([[0,         0,          0,  0],
                   [ 0, -np.sin(q), -np.cos(q), 0],
                   [ 0,  np.cos(q), -np.sin(q), 0],
                   [ 0,         0,          0,  0]], dtype = float)
    return dTx

def dtroty(q):
    dTy = np.array([[-np.sin(q), 0, np.cos(q), 0],
                   [          0, 0,         0, 0],
                   [ -np.cos(q), 0,-np.sin(q), 0],
                   [          0, 0,         0, 0]], dtype = float)
    return dTy

def dtrotz(q):
    dTz = np.array([[-np.sin(q), -np.cos(q), 0, 0],
                   [ np.cos(q),  -np.sin(q), 0, 0],
                   [        0,            0, 0, 0],
                   [        0,            0, 0, 0]], dtype = float)
    return dTz

def dtranslx(q):
    dTx = np.array([[0, 0, 0, 1],
                   [ 0, 0, 0, 0],
                   [ 0, 0, 0, 0],
                   [ 0, 0, 0, 0]], dtype = float)
    return dTx

def dtransly(q):
    dTy = np.array([[0, 0, 0, 0],
                   [ 0, 0, 0, 1],
                   [ 0, 0, 0, 0],
                   [ 0, 0, 0, 0]], dtype = float)
    return dTy

def dtranslz(q):
    dTz = np.array([[0, 0, 0, 0],
                   [ 0, 0, 0, 0],
                   [ 0, 0, 0, 1],
                   [ 0, 0, 0, 0]], dtype = float)
    return dTz
    
def ik(tool_pose, links_length):
    q1 = np.arctan2(tool_pose[1],tool_pose[0])
    q2 = tool_pose[2] - links_length[0]
    q3 = np.sqrt(tool_pose[0]**2+tool_pose[1]**2) - links_length[1]

    return np.array([q1,q2,q3], dtype = float)

def fk(q, theta, links_length):
   T01 = np.linalg.multi_dot([trotz(q[0]), trotz(theta[0]), translz(links_length[0])])
   T02 = np.linalg.multi_dot([T01, translz(theta[1]), translz(q[1]), translx(links_length[1])])
   T03 = np.linalg.multi_dot([T02, translx(q[2]), translx(theta[2])])
   return T03

def k_theta(stiff_coeffs):
    Ktheta = np.array([[stiff_coeffs[0], 0, 0],
                        [0, stiff_coeffs[1],0],
                        [0, 0, stiff_coeffs[2]]], dtype =float)
    return Ktheta

def J_t(links_length, theta, q):
    T = fk(q, theta, links_length)
    T[0:3,3] = np.array([0,0,0], dtype = float)
    T = np.transpose(T)

    Td = np.linalg.multi_dot([trotz(q[0]), dtrotz(theta[0]), translz(links_length[0]), translz(theta[1]), translz(q[1]), translx(links_length[1]), translx(q[2]), translx(theta[2]), T])

    Jt1 = np.vstack([Td[0,3], Td[1,3], Td[2,3], Td[2,1], Td[0,2], Td[1,0]])

    Td = np.linalg.multi_dot([trotz(q[0]), trotz(theta[0]), translz(links_length[0]), dtranslz(theta[1]), translz(q[1]), translx(links_length[1]), translx(q[2]), translx(theta[2]), T])

    Jt2 = np.vstack([Td[0,3], Td[1,3], Td[2,3], Td[2,1], Td[0,2], Td[1,0]])

    Td = np.linalg.multi_dot([trotz(q[0]), trotz(theta[0]), translz(links_length[0]), translz(theta[1]), translz(q[1]), translx(links_length[1]), translx(q[2]), dtranslx(theta[2]), T])

    Jt3 = np.vstack([Td[0,3], Td[1,3], Td[2,3], Td[2,1], Td[0,2], Td[1,0]])
    
    Jt = np.hstack([Jt1, Jt2, Jt3])
    return Jt, Jt1, Jt2, Jt3



if __name__ == "__main__":
    links_length = np.array([2, 1], dtype = float)

    stiff_coeffs = np.array([1*10**6,2*10**6,0.5*10**6], dtype = float)

    # tool_pose = np.array([10, 10, 10], dtype = float)
    q = np.array([np.pi/3,0.5,0.5], dtype = float)
    theta = np.array([0,0,0], dtype = float)


    Ktheta = k_theta(stiff_coeffs)
    A_first = np.zeros((3,3), dtype = float)
    A_second = np.zeros((3,1), dtype = float)

    n = 30 # number of experiments
    for i in range(n):
        # w = np.array([500,500,1000,0,0,0], dtype = float)
        w = np.random.uniform(low = -1000, high = 1000, size = (6,1))
        w = np.reshape(w, (6,1))
        q_r = np.random.uniform(low = -np.pi, high = np.pi, size = (1,1))
        q_t = np.random.uniform(low = 0, high = 1, size = (1,2))
        q = np.array([q_r[0], q_t[0,0], q_t[0,1]])
        Jt, Jt1, Jt2, Jt3 = J_t(links_length, theta, q)
        eps = np.random.normal(loc = 0.0, scale = 1e-5, size = (6,1))
        dt = np.linalg.multi_dot([Jt, np.linalg.inv(Ktheta), np.transpose(Jt), w]) + eps
        wt = np.reshape(w[0:3], (3,1))

        A1 = np.linalg.multi_dot([Jt1[0:3],np.transpose(Jt1[0:3]),wt])
        A2 = np.linalg.multi_dot([Jt2[0:3],np.transpose(Jt2[0:3]),wt])
        A3 = np.linalg.multi_dot([Jt3[0:3],np.transpose(Jt3[0:3]),wt])
        # Observation matrix
        A = np.hstack([A1, A2, A3])

        dt = dt[0:3]
        A_first = A_first + np.linalg.multi_dot([np.transpose(A),A])
        A_second = A_second + np.linalg.multi_dot([np.transpose(A), dt])

    ks = np.linalg.multi_dot([np.linalg.inv(A_first), A_second])
    stiffness = np.divide(1, ks)

    print('stiffness = \n', stiffness,'\n')

    force = np.array([-440, -1370, -1635, 0, 0, 0], dtype= float)
    
    r = 0.2
    xc = 1.1
    yc = 0.0
    zc = 2.3

    n = 30

    alpha = np.linspace(0, 2*np.pi, n, dtype = float)
    space_vector_list = np.zeros((3, n))

    xx = xc + r*np.cos(alpha)
    yy = yc + r*np.sin(alpha)
    zz = zc*np.ones(n, dtype = float)
    for i in range(n):
        space_vector_list[:,i] = ik([xx[i], yy[i], zz[i]], links_length)

    trag_desired = np.stack([xx,yy,zz])

    # Uncalibrated Trajectory

    Ktheta = k_theta(stiffness)

    circle2 = np.zeros(trag_desired.shape, dtype = float)

    for i in range(n):
        q = space_vector_list[:,i]
        Jt, Jt1, Jt2, Jt3 = J_t(links_length, theta, q)
        eps = np.random.normal(loc = 0.0, scale = 1e-5, size = 6)
        dt = np.linalg.multi_dot([Jt, np.linalg.inv(Ktheta), np.transpose(Jt), force]) + eps
        circle2[:,i] = dt[0:3] + trag_desired[:,i]

    diff = trag_desired - circle2 # deflection

    upd = trag_desired + diff # new 'initial' trajectory
    for i in range(n):
        space_vector_list[:,i] = ik([upd[0,i], upd[1,i], upd[2,i]], links_length)
    Ktheta = k_theta(stiff_coeffs) # return old trajectory

    # Calibrated trajectory
    circle3 = np.zeros(trag_desired.shape, dtype = float)

    for i in range(n):
        q = space_vector_list[:,i]
        Jt, Jt1, Jt2, Jt3 = J_t(links_length, theta, q)
        eps = np.random.normal(loc = 0.0, scale = 1e-5, size = 6)
        dt = np.linalg.multi_dot([Jt, np.linalg.inv(Ktheta), np.transpose(Jt), force]) + eps
        circle3[:,i] = dt[0:3] + upd[:,i]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot3D(trag_desired[0], trag_desired[1], trag_desired[2], c='green', linewidth=5)

    ax.scatter3D(circle2[0], circle2[1], circle2[2], c='red', linewidth=5)

    ax.scatter3D(circle3[0], circle3[1], circle3[2], c='blue', linewidth=5)

    red_patch = mpatches.Patch(color='red', label='Uncalibrated Trajectory')

    blue_patch = mpatches.Patch(color='blue', label='Calibrated Trajectory')

    green_patch = mpatches.Patch(color='green', label='Desired Trajectory')

    F_patch = mpatches.Patch(color='black', label='F =' + str(force))

    plt.legend(handles=[red_patch, blue_patch, green_patch, F_patch])
    plt.show()

