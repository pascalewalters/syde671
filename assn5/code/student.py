# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu, 
# Grady Williams, and James Hays for CSCI 1430 @ Brown and 
# CS 4495/6476 @ Georgia Tech

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color
from mpl_toolkits.mplot3d import Axes3D

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    #[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the '\' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 

    assert Points_2D.shape[0] == Points_3D.shape[0]

    A = np.zeros(((Points_2D.shape[0] * 2), 11))
    c = np.zeros(((Points_2D.shape[0] * 2), 1))

    for i in range(Points_2D.shape[0]):
        a = 2 * i
        b = (2 * i) + 1
        
        A[a, 0] = Points_3D[i, 0]
        A[a, 1] = Points_3D[i, 1]
        A[a, 2] = Points_3D[i, 2]
        A[a, 3] = 1
        A[a, 8] = -Points_2D[i, 0] * Points_3D[i, 0] 
        A[a, 9] = -Points_2D[i, 0] * Points_3D[i, 1] 
        A[a, 10] = -Points_2D[i, 0] * Points_3D[i, 2] 


        A[b, 4] = Points_3D[i, 0]
        A[b, 5] = Points_3D[i, 1]
        A[b, 6] = Points_3D[i, 2]
        A[b, 7] = 1
        A[b, 8] = -Points_2D[i, 1] * Points_3D[i, 0] 
        A[b, 9] = -Points_2D[i, 1] * Points_3D[i, 1] 
        A[b, 10] = -Points_2D[i, 1] * Points_3D[i, 2]

        c[a] = Points_2D[i, 0]
        c[b] = Points_2D[i, 1]

    sol = np.linalg.lstsq(A, c, rcond = None)[0]
    M = np.reshape(np.append(sol, [1]), (3, -1))

    return M

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    m_4 = M[:, 3]
    Q = M[:, 0:3]

    Center = np.matmul(-1 * np.linalg.inv(Q), m_4)

    return Center

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix
def estimate_fundamental_matrix(Points_a, Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project

    assert Points_a.shape[0] == Points_b.shape[0]

    A = np.zeros((Points_a.shape[0], 9))
    c = np.zeros((Points_a.shape[0], 1))

    for i in range(Points_a.shape[0]):
        A[i, 0] = Points_a[i, 0] * Points_b[i, 0]
        A[i, 1] = Points_a[i, 1] * Points_b[i, 0]
        A[i, 2] = Points_b[i, 0]
        A[i, 3] = Points_a[i, 0] * Points_b[i, 1]
        A[i, 4] = Points_a[i, 1] * Points_b[i, 1]
        A[i, 5] = Points_b[i, 1]
        A[i, 6] = Points_a[i, 0]
        A[i, 7] = Points_a[i, 1]
        A[i, 8] = 1

    U, S, Vh = np.linalg.svd(A, full_matrices = False)
    F_matrix = np.reshape(Vh[-1, :], (3,3))

    return F_matrix

# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.

    # Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.
    n = 9
    thresh = 0.05
    normalize = True

    if normalize:
        # Perform coordinate normalization
        matches_a_norm = np.zeros_like(matches_a)
        matches_b_norm = np.zeros_like(matches_b)

        mean_a = np.mean(matches_a, axis = 0)
        mean_b = np.mean(matches_b, axis = 0)

        std_a = np.std(matches_a - mean_a, axis = 0)
        std_b = np.std(matches_b - mean_b, axis = 0)

        S_a = np.eye(3)
        S_a[0, 0] = 1.0 / std_a[0]
        S_a[1, 1] = 1.0 / std_a[1]
        S_b = np.eye(3)
        S_b[0, 0] = 1.0 / std_b[0]
        S_b[1, 1] = 1.0 / std_b[1]

        O_a = np.eye(3)
        O_a[0, 2] = -1 * mean_a[0]
        O_a[1, 2] = -1 * mean_a[1]
        O_b = np.eye(3)
        O_b[0, 2] = -1 * mean_b[0]
        O_b[1, 2] = -1 * mean_b[1]

        T_a = np.matmul(S_a, O_a)
        T_b = np.matmul(S_b, O_b)

        for i in range(matches_a.shape[0]):
            normalized_a = np.matmul(T_a, 
                    np.reshape(np.append(matches_a[i], [1]), (3, -1)))[0:2, 0]
            matches_a_norm[i, 0] = normalized_a[0]
            matches_a_norm[i, 1] = normalized_a[1]

            normalized_b = np.matmul(T_b, 
                    np.reshape(np.append(matches_b[i], [1]), (3, -1)))[0:2, 0]
            matches_b_norm[i, 0] = normalized_b[0]
            matches_b_norm[i, 1] = normalized_b[1]
    else:
        matches_a_norm = np.copy(matches_a)
        matches_b_norm = np.copy(matches_b)

        T_a = np.eye(3)
        T_b = np.eye(3)


    # Perform RANSAC
    while True:
        idx = np.random.choice(matches_b.shape[0], n, replace=False)

        points_a = matches_a[idx, :]
        points_b = matches_b[idx, :]

        points_a_norm = matches_a_norm[idx, :]
        points_b_norm = matches_b_norm[idx, :]

        fun_matrix = estimate_fundamental_matrix(points_a_norm, points_b_norm)
        fun_matrix = np.matmul(np.matmul(T_b.T, fun_matrix), T_a)

        inliers = []

        for j in range(matches_a.shape[0]):
            a = np.reshape(np.matmul(np.append(matches_b[j], [1]), fun_matrix), (1, 3))
            b = np.reshape(np.append(matches_a[j], [1]), (3, 1))
            error = np.abs(np.matmul(a, b)[0, 0])

            if error < thresh:
                inliers.append(j)

        if len(inliers) >= int(0.2 * matches_a.shape[0]):
            break

    Best_Fmatrix = estimate_fundamental_matrix(matches_a_norm[inliers], matches_b_norm[inliers])
    Best_Fmatrix = np.matmul(np.matmul(T_b.T, Best_Fmatrix), T_a)
    print(Best_Fmatrix)

    inliers_a = []
    inliers_b = []

    for j in range(matches_a.shape[0]):
        a = np.reshape(np.matmul(np.append(matches_b[j], [1]), Best_Fmatrix), (1, 3))
        b = np.reshape(np.append(matches_a[j], [1]), (3, 1))
        error = np.abs(np.matmul(a, b)[0, 0])

        if error < thresh:
            inliers_a.append(matches_a[j])
            inliers_b.append(matches_b[j])

    return Best_Fmatrix, np.array(inliers_a)[0:30, :], np.array(inliers_b)[0:30, :]
