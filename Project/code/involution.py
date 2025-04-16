import numpy as np
from scipy import stats
import scipy.linalg as la
import matplotlib.pyplot as plt
import jax
from jax import random
import jax.numpy as jnp
from jax import grad,value_and_grad
import math




## setup
eps = 0.001
nmax = 100
d = 2
a = 3
b = 2
# manifold eqn
def q(input):
    x = input[0]
    y = input[1]

    # # 1-D circle
    # return x**2 + y**2 - 1

    # 1-D ellipse
    return x**2/a**2 + y**2/b**2 - 1


Q = grad(q)



## Newton solver
## v is shift
## q is equation constraint
## Q_x is matrix rep of normal span at x, whose row vector as gradient
def solve(v, q, Q_x):
    num_comp = len(Q_x)
    a = jnp.zeros(num_comp)
    i = 0
    flag = 1
    temp = Q_x.T @ a
    while jnp.linalg.norm(q(v + temp)) > eps:
        ## solve (grad(q)(v+Q*a)^T * Q) * delta_a = -q(v+Q*a) for delta_a
        g = grad(q)
        A = jnp.array([g(v+temp) @ (Q_x.T)])
        b = jnp.array([-q(v+temp)])
        delta_a = jnp.linalg.solve(A, b)
        a = a + delta_a
        i = i + 1
        # exceding max times, stop
        if i > nmax:
            flag = 0
            return [a, flag]
        temp = Q_x.T @ a
    return [a, flag]

## helper for vector projection onto space
## M's row vector is the orthonormal basis vector(usually column vector)
def project(v, M):
    dim = len(M[0])
    proj = jnp.zeros(dim)
    for i in range(len(M)):
        e_i = M[i]
        ## inner product to generate coeff: c_i = <v, e_i>
        coeff = v @ e_i
        ## projection = sum(c_i*e_i)
        proj = proj + coeff * e_i
    return proj


def null_space(A, eps=1e-15):
    u, s, vh = jnp.linalg.svd(A)
    nnz = (s >= eps).sum()
    ns = vh[nnz:].conj().T
    return ns


# ###=========================================================
# ## phi(x, v_x) = (y, v_y)
# ## This gives Jacobian = 0

# ## x, v_x of jnp.array type
# def pre_inv(input):
#     x = input[0:d]
#     v_x = input[d:]
#     ## this input (x, v_x) guarantees the solution of solver
#     Q_x = jnp.array([Q(x)])
#     res = solve(x+v_x, q, Q_x)
#     a = res[0]
#     w_x = Q_x.T @ a
#     y = x + v_x + w_x

#     # find gradient: Q_y
#     # use QR decomposition to have bases of T_y and N_y
#     ## tangent space at y
#     Q_y = jnp.array([Q(y)])

#     z = jnp.linalg.norm(Q_y[0])
#     T_y = jnp.array([[-Q_y[0][1]/z, Q_y[0][0]/z]])
#     v_y = project(x-y, T_y)
    
#     # T_y = jnp.array(la.null_space(Q_y))
#     # T_y = null_space(Q_y)
#     # v_y = project(x-y, T_y.T)
#     return jnp.concatenate((y, v_y))


# def involution(input):
#     x = input[0:d]
#     v_x = input[d:]
#     Q_x = jnp.array([Q(x)])
#     res = solve(x+v_x, q, Q_x)
#     flag = res[1]
#     ## if fail to project y on manifold, set (x', v_x') = (x, v_x)
#     ## else, pass (x, vx) to pre_inv
#     ## this creates a subset: S \subset V
#     if flag == 0:
#         return input
#     else:
#         img = pre_inv(input)
#         if jnp.linalg.norm(pre_inv(img) - input) > eps:
#             return input
#         else:
#             return img


###========================================================
## T0(x, v_x) (y, v_y)
## \hat{T0} = \psi(T0(\phi^-1))
## \hat{T0}(theta_x, t_x) = (theta_y, t_y)
## This should give non-zero Jacobian

def pre_inv(input):
    x = input[0:d]
    v_x = input[d:]
    ## this input (x, v_x) guarantees the solution of solver
    Q_x = jnp.array([Q(x)])
    res = solve(x+v_x, q, Q_x)
    a = res[0]
    w_x = Q_x.T @ a
    y = x + v_x + w_x

    # find gradient: Q_y
    # use QR decomposition to have bases of T_y and N_y
    ## tangent space at y
    Q_y = jnp.array([Q(y)])
    z = jnp.linalg.norm(Q_y[0])
    T_y = jnp.array([[-Q_y[0][1]/z, Q_y[0][0]/z]])

    # # 1-D ellipse
    # T_y = jnp.array([[-Q_y[0][1]/z, Q_y[0][0]/z]])

    v_y = project(x-y, T_y)
    
    # T_y = jnp.array(la.null_space(Q_y))
    # T_y = null_space(Q_y)
    # v_y = project(x-y, T_y.T)
    return jnp.concatenate((y, v_y))


def involution(input):
    x = input[0:d]
    v_x = input[d:]
    Q_x = jnp.array([Q(x)])
    
    res = solve(x+v_x, q, Q_x)
    flag = res[1]
    ## if fail to project y on manifold, set (x', v_x') = (x, v_x)
    ## else, pass (x, vx) to pre_inv
    ## this creates a subset: S \subset V
    if flag == 0:
        return input
    else:
        img = pre_inv(input)
        if jnp.linalg.norm(pre_inv(img) - input) > eps:
            return input
        else:
            return img
        

# input the local coordinate of original input: \phi(x, v_x)
# assume the original input satisfies solvablity and reversibilty

# input (theta_x, t_x) are local coord of point and tangent vector respectively. But I am not sure whether t_x should be the coord wrt "natural" basis {d/dx^i} or ONB {E_i} of tangent space
def pre_inv_local(input):
    # theta_x is the local coordinate of the point on manifold
    theta_x = input[0]
    # t_x is the local coordinate of the tangent vector
    t_x = input[1]

    # # 1-D circle
    # x = jnp.array([jnp.cos(theta_x), jnp.sin(theta_x)])
    # Q_x = jnp.array([Q(x)])
    # z_x = jnp.linalg.norm(Q_x[0])
    # T_x = jnp.array([-Q_x[0][1]/z_x, Q_x[0][0]/z_x])
    # v_x = t_x * T_x
    # img = pre_inv(jnp.concatenate((x, v_x)))
    # y = img[0:d]
    # v_y = img[d:]
    # theta_y = jnp.arccos(y[0])
    # t_y = jnp.linalg.norm(v_y)


    # 1-D ellipse
    x = jnp.array([a*jnp.cos(theta_x), b*jnp.sin(theta_x)])
    Q_x = jnp.array([Q(x)])
    z_x = jnp.linalg.norm(Q_x[0])
    T_x = jnp.array([-Q_x[0][1]/z_x, Q_x[0][0]/z_x])
    # Riemann metric at x
    g_x = a**2 * (jnp.sin(theta_x))**2 + b**2 * (jnp.cos(theta_x))**2
    ## if we want t_x to be coord wrt natural basis {d/dx^i}
    # norm_vx = t_x * jnp.sqrt(g_x)
    # if we want t_x to be coord wrt orthonormal basis {E_i}
    norm_vx = t_x
    v_x = norm_vx * T_x
    img = pre_inv(jnp.concatenate((x, v_x)))
    y = img[0:d]
    v_y = img[d:]
    theta_y = jnp.arccos(y[0]/a)
    norm_vy = jnp.linalg.norm(v_y)
    # Riemann metric at y
    g_y = a**2 * (jnp.sin(theta_y))**2 + b**2 * (jnp.cos(theta_y))**2
    # t_y = -norm_vy / jnp.sqrt(g_y)
    t_y = -norm_vy



    return jnp.array([theta_y, t_y])



###==========================================================  
## test setup
        
# 1-D test case

#(1) using ambient coord
# u = np.random.uniform(0,1)
# x = jnp.array([u, math.sqrt(1-u**2)])
# Q_x = jnp.array([Q(x)])
# T_x = la.null_space(Q_x)
# v = np.random.uniform(0,1)
# v_x = v * (T_x.T)[0]
# input = jnp.concatenate((x, v_x))



#(2) using local coord

# # 1-D circle
# u = np.random.uniform(0,1)
# x = jnp.array([u, math.sqrt(1-u**2)])
# Q_x = jnp.array([Q(x)])
# T_x = la.null_space(Q_x)
# v = np.random.uniform(0,1)
# v_x = v * (T_x.T)[0]

# input = jnp.concatenate((x, v_x))

# print("input of T: ", input)
# res = involution(involution(input))
# # check the double-mappings are consistent with the original value (up to small margin)
# print("output of T: ", res)
# print("error: ", res - input)


# # transform input (x, v_x) into local coord (theta_x, t_x)
# theta_x = jnp.arccos(x[0])
# t_x = jnp.linalg.norm(v_x)
# local_coord = jnp.array([theta_x, t_x])
# print("output of T0hat: ", pre_inv_local(local_coord))


# 1-D ellipse
u = np.random.uniform(0,a)
x = jnp.array([u, math.sqrt(b**2-(b**2/a**2)*u**2)])
Q_x = jnp.array([Q(x)])
T_x = la.null_space(Q_x)
v = np.random.uniform(0,5)
# print(T_x)
# print(jnp.linalg.norm((T_x.T)[0]))
v_x = v * (T_x.T)[0]


input = jnp.concatenate((x, v_x))

print("input of T: ", input)
res = involution(involution(input))
# check the double-mappings are consistent with the original value (up to small margin)
print("output of ToT: ", res)
print("error: ", res - input)

if (jnp.linalg.norm(res - input) != 0):
    # transform input (x, v_x) into local coord (theta_x, t_x)
    theta_x = jnp.arccos(x[0]/a)
    # Riemann metric g at x
    g_x = a**2 * (jnp.sin(theta_x))**2 + b**2 * (jnp.cos(theta_x))**2
    # t_x = jnp.linalg.norm(v_x) / jnp.sqrt(g_x)
    t_x = jnp.linalg.norm(v_x)
    local_coord = jnp.array([theta_x, t_x])
    print("input of T0hat: ", local_coord)
    print("output of T0hat: ", pre_inv_local(local_coord))
    Jac_fn = jax.jacfwd(pre_inv_local)
    Jac_mtx = Jac_fn(local_coord)
    print("Jacobian matrix of T0hat: \n", Jac_mtx)
    print("Determinant of J: ", jax.numpy.linalg.det(Jac_mtx))
    print()
else:
    Jac_fn = jax.jacfwd(involution)
    Jac_mtx = Jac_fn(input)
    print("Jacobian matrix of T0hat: \n", Jac_mtx)
    print("Determinant of J: ", jax.numpy.linalg.det(Jac_mtx))







# 2-D test case
# u = np.random.uniform(0,1)
# x = jnp.array([u, math.sqrt(1-u**2)])
# sd = 0.1
# v_x = np.random.multivariate_normal([0,0], [[sd,0],[0,sd]])
