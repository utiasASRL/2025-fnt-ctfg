import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import inv
from sksparse.cholmod import cholesky
import matplotlib.pyplot as plt


# Build the transition function
def gp_trans(t2, t1):
    return np.block([[1, t2-t1], [0, 1]])

# Covariance block
def gp_cov(Qc, t2, t1):
    delt = t2 - t1 
    return np.block([[delt**3/3, delt**2/2], [delt**2/2, delt]]) * Qc

# Inverse covariance block
def gp_cov_inv(Qc, t2, t1):
    delt = t2 - t1   
    return np.array([[12*delt**-3, -6*delt**-2], [-6*delt**-2, 4/delt]]) / Qc

# Ground truth trajectory
def ground_truth(t):
    return np.sin(t).reshape(-1, 1), np.cos(t).reshape(-1, 1)

##################################################################

# Define a sequence of times
K = 20
tmax = 5*np.pi
t = np.linspace(0, tmax, K+1)

# Noises
np.random.seed(0)  # For reproducibility
R = 0.03
Qc = 2e-1

#### Ground truth trajecotry
t_gt = np.linspace(0, tmax, 100*(K+1))
pos_gt, vel_gt = ground_truth(t_gt)

#### Prior

# Compute the prior error
v = [np.array([[0],[0]]) for _ in range(K+1)]
v = np.vstack(v)

# Build the lifted transition matrix block by block
Ainv = [[None for _ in range(K+1)] for _ in range(K+1)]
Ainv[0][0] = np.eye(2)
for k in range(1,K+1):
    Ainv[k][k] = np.eye(2)
    Ainv[k][k-1] = -gp_trans(t[k], t[k-1])
Ainv = bmat(Ainv, format="csr")

# Build the lifted covariance matrix block by block
Qinv = [[None for _ in range(K+1)] for _ in range(K+1)]
Qinv[0][0] = 1*np.linalg.inv(np.eye(2))
for k in range(1,K+1):
    Qinv[k][k] = gp_cov_inv(Qc, t[k], t[k-1])
Qinv = bmat(Qinv, format="csr")

# Build LHS and RHS
LHS = Ainv.T @ Qinv @ Ainv
RHS = Ainv.T @ Qinv @ v

#### Final Pose Measurement

# Measurements
y = [np.array([np.sin(t[k]) + np.sqrt(R)*np.random.randn()]) for k in range(K+1)]
y = np.vstack(y)

# # Compute the measurement Jacobian
C = [[None for _ in range(K+1)] for _ in range(K+1)]
for k in range(K+1):
    C[k][k] = np.array([[1, 0]])
C = bmat(C, format="csr")

# Measurement covariances
Rinv = [[None for _ in range(K+1)] for _ in range(K+1)]
for k in range(K+1):
    Rinv[k][k] = np.array([1.0/R]) 
Rinv = bmat(Rinv, format="csr")

# Update the LHS and RHS
LHS += C.T @ Rinv @ C
RHS += C.T @ Rinv @ y

#### Solve and update the trajectory

# Solve the linear system using Cholesky factorization
L = cholesky(LHS)
x = L.solve_A(RHS)

# Convert x back into a list of elements for position and velocity
pos = x[:2*(K+1):2]  # Extract positions
vel = x[1:2*(K+1):2]  # Extract velocities

# Covariance of the posterior - replace with partial calculation later
Sigma = inv(LHS)

# Extract the position and velocity covariances
pos_Sigma = np.array([Sigma[2*k, 2*k] for k in range(K+1)]).reshape(K+1, 1)
vel_Sigma = np.array([Sigma[2*k+1, 2*k+1] for k in range(K+1)]).reshape(K+1, 1)


#### Interpolate for the mean and covariance
J = 100  # Number of interpolation points per segment
t_interp_final = None
pos_interp_final = None
vel_interp_final = None
pos_Sigma_interp_final = None
vel_Sigma_interp_final = None

for k in range(K):
    # Interpolate this segment
    t_interp = np.linspace(t[k], t[k+1], J)
    pos_interp = np.zeros((J, 1))
    vel_interp = np.zeros((J, 1))
    pos_Sigma_interp = np.zeros((J, 1))
    vel_Sigma_interp = np.zeros((J, 1))
    
    # Interpolate the mean/cov using the GP interpolation scheme
    for j in range(J):
        if j == 0:
            # Use the first point directly
            pos_interp[j] = pos[k]
            vel_interp[j] = vel[k]
            pos_Sigma_interp[j] = pos_Sigma[k]
            vel_Sigma_interp[j] = vel_Sigma[k]
        elif j == J-1:
            # Use the last point directly
            pos_interp[j] = pos[k+1]
            vel_interp[j] = vel[k+1]
            pos_Sigma_interp[j] = pos_Sigma[k+1]
            vel_Sigma_interp[j] = vel_Sigma[k+1]
        else:
            # Setup
            Sigma_cond = np.linalg.inv(gp_cov_inv(Qc, t_interp[j], t[k]) + gp_trans(t[k+1], t_interp[j]).T @ gp_cov_inv(Qc, t[k+1], t_interp[j]) @ gp_trans(t[k+1], t_interp[j]))
            Psi = Sigma_cond @ gp_trans(t[k+1], t_interp[j]).T @ gp_cov_inv(Qc, t[k+1], t_interp[j])
            Lambda = Sigma_cond @ gp_cov_inv(Qc, t_interp[j], t[k]) @ gp_trans(t_interp[j], t[k])
            
            # Mean
            x1 = np.array([[pos[k]], [vel[k]]]).reshape(2, 1)
            x2 = np.array([[pos[k+1]], [vel[k+1]]]).reshape(2, 1)
            x = Lambda @ x1 + Psi @ x2
            pos_interp[j] = x[0, 0]
            vel_interp[j] = x[1, 0]

            # Covariance
            Sigma_k = Sigma[2*k:2*k+4, 2*k:2*k+4]
            B = np.block([Lambda, Psi])
            Sigma_j = Sigma_cond + B @ Sigma_k @ B.T
            pos_Sigma_interp[j] = Sigma_j[0, 0]
            vel_Sigma_interp[j] = Sigma_j[1, 1]

    # Append the interpolated values
    if t_interp_final is None:
        t_interp_final = t_interp
        pos_interp_final = pos_interp
        vel_interp_final = vel_interp
        pos_Sigma_interp_final = pos_Sigma_interp
        vel_Sigma_interp_final = vel_Sigma_interp
    else:
        t_interp_final = np.concatenate((t_interp_final, t_interp))
        pos_interp_final = np.concatenate((pos_interp_final, pos_interp))
        vel_interp_final = np.concatenate((vel_interp_final, vel_interp))
        pos_Sigma_interp_final = np.concatenate((pos_Sigma_interp_final, pos_Sigma_interp))
        vel_Sigma_interp_final = np.concatenate((vel_Sigma_interp_final, vel_Sigma_interp))


# Plot the results
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(t_gt, pos_gt, 'g-', label='ground truth', alpha=0.3)
plt.plot(t_interp_final, pos_interp_final, 'b-', label='mean')
plt.plot(t_interp_final, pos_interp_final + 3*np.sqrt(pos_Sigma_interp_final), 'b-', label='covariance', alpha=0.2)
plt.plot(t_interp_final, pos_interp_final - 3*np.sqrt(pos_Sigma_interp_final), 'b-', label=None, alpha=0.2)
plt.plot(t, pos, 'b.', label=None)
plt.plot(t, pos + 3 * np.sqrt(pos_Sigma), 'b.', label=None, alpha=0.2)
plt.plot(t, pos - 3 * np.sqrt(pos_Sigma), 'b.', label=None, alpha=0.2)
plt.plot(t, y, 'r.', label='measurements')
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$p$ [m]')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_gt, vel_gt, 'g-', label='ground truth', alpha=0.3)
plt.plot(t_interp_final, vel_interp_final, 'b-', label='mean')
plt.plot(t_interp_final, vel_interp_final + 3*np.sqrt(vel_Sigma_interp_final), 'b-', label='covariance', alpha=0.2)
plt.plot(t_interp_final, vel_interp_final - 3*np.sqrt(vel_Sigma_interp_final), 'b-', label=None, alpha=0.2)
plt.plot(t, vel, 'b.', label=None)
plt.plot(t, vel + 3 * np.sqrt(vel_Sigma), 'b.', label=None, alpha=0.2)
plt.plot(t, vel - 3 * np.sqrt(vel_Sigma), 'b.', label=None, alpha=0.2)
plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$\dot{p}$ [m/s]')
plt.legend()

plt.tight_layout()
plt.show(block=True)

# Export the plot to a PDF file
plt.savefig('linear_WNOA_example.pdf', format='pdf', bbox_inches='tight')




