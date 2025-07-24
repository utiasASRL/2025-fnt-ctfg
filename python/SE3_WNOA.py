import numpy as np
import numpy.linalg as npla
from math import cos, sin, factorial
from scipy.sparse import bmat
from scipy.sparse.linalg import inv
from sksparse.cholmod import cholesky
import pyvista as pv
from pylgmath import se3op
import time
import matplotlib.pyplot as plt

# Plot the SE(3) transform
def plot_se3(plotter, T, scale=0.05, color=None, opacity=0.3):
    """
    Plot the SE(3) transform T using PyVista Arrows.
    """
    # Translation
    origin = T[:3, 3]
    
    # Axes directions (scaled)
    x_axis = T[:3, 0] * scale
    y_axis = T[:3, 1] * scale
    z_axis = T[:3, 2] * scale
    
    # Create and add arrow meshes
    arrow_x = pv.Arrow(start=origin, direction=x_axis, scale=scale, shaft_radius=0.2*scale)
    arrow_y = pv.Arrow(start=origin, direction=y_axis, scale=scale, shaft_radius=0.2*scale)
    arrow_z = pv.Arrow(start=origin, direction=z_axis, scale=scale, shaft_radius=0.2*scale)

    if color is None:
        plotter.add_mesh(arrow_x, color='red', opacity=opacity)
        plotter.add_mesh(arrow_y, color='green', opacity=opacity)
        plotter.add_mesh(arrow_z, color='blue', opacity=opacity)
    else:
        plotter.add_mesh(arrow_x, color=color, opacity=opacity)
        plotter.add_mesh(arrow_y, color=color, opacity=opacity)
        plotter.add_mesh(arrow_z, color=color, opacity=opacity)

# Plot the covariance
def plot_covariance(plotter, T, P, nstd=1, color='lightgray', opacity=0.2):
    """
    Plot the covariance ellipsoid at the given SE(3) transform T.
    """
    # Extract the rotation and translation from T
    R = T[:3, :3]
    t = T[:3, 3]

    # Do an eigen decomposition of the covariance matrix
    D, V = npla.eig(R @ P @ R.T)
    W = nstd * V @ np.diag(np.sqrt(D))
    
    # Create an ellipsoid mesh
    ellipsoid = pv.Sphere(radius=1, theta_resolution=50, phi_resolution=50)
    
    # Scale the ellipsoid by the covariance matrix and translate it into place
    ellipsoid.points = (W @ ellipsoid.points.T).T + t
    
    plotter.add_mesh(ellipsoid, color=color, opacity=opacity)

# Make the axes equal
def axes_equal(ax):
    # Ensure equal scaling for all axes
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Find the maximum range
    max_range = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0])
    ) / 2.0

    # Calculate the midpoints
    x_mid = (x_limits[0] + x_limits[1]) / 2.0
    y_mid = (y_limits[0] + y_limits[1]) / 2.0
    z_mid = (z_limits[0] + z_limits[1]) / 2.0

    # Set the new limits
    ax.set_xlim([x_mid - max_range, x_mid + max_range])
    ax.set_ylim([y_mid - max_range, y_mid + max_range])
    ax.set_zlim([z_mid - max_range, z_mid + max_range])

# Define a function to save the image to high-res PNG
# and crop it to the bounding box of the non-transparent part
def save_image():
    # Export as before
    plotter.export_gltf('output.gltf', True, True, False)
    
    # Take screenshot with transparent background
    temp_file = 'output_temp.png'
    plotter.screenshot(temp_file, transparent_background=True, window_size=[15360, 8640])
    
    # Now crop the image using PIL
    from PIL import Image

    # Disable decompression bomb warning
    Image.MAX_IMAGE_PIXELS = None
    
    # Open the image
    img = Image.open(temp_file)
    
    # Get the alpha channel
    alpha = img.getchannel('A')
    
    # Get the bounding box of the non-transparent part
    bbox = alpha.getbbox()
    
    if bbox:
        # Crop to the bounding box
        cropped = img.crop(bbox)
        
        # Add a small margin (optional)
        margin = 20  # pixels
        width, height = cropped.size
        new_bbox = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(img.width, bbox[2] + margin),
            min(img.height, bbox[3] + margin)
        )
        cropped_with_margin = img.crop(new_bbox)
        
        # Save the result
        cropped_with_margin.save('output.png')
        
        print(f"Image cropped from {img.size} to {cropped_with_margin.size}")
    else:
        # If for some reason no content was found, save the original
        img.save('output.png')
        print("No content found for cropping")
    
    # Remove the temporary file
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)


##################################################################

class GPPrior:
    def __init__(self, 
                 K=11,
                 times=np.linspace(0,1,11),
                 Qc = np.eye(6),
                 P_0_inv = 1e1*np.eye(12), 
                 T_0=np.eye(4), 
                 varpi_0=np.array([-1, 0, 0, 0, 0, 0]).reshape(6,1),
                 order=1, 
                 error_type=1,
                 mean_type=0,
                 lock_dof_k=(0,0,0,0,0,0,0,0,0,0,0,0)              
                 ):
        
        # Initialize
        self.K=K
        self.times=times
        self.Qc = Qc
        self.P_0_inv = P_0_inv
        self.T_0 = T_0
        self.varpi_0 = varpi_0
        self.order = order
        self.error_type = error_type
        self.mean_type = mean_type
        self.lock_dof_k = lock_dof_k
                
    # Build the GP prior
    def build_lin_prior(self, T, varpi):

        # Compute the prior error
        e_prior = [None for _ in range(self.K+1)]
        e_prior[0] = np.block([[-se3op.tran2vec(T[0] @ np.linalg.inv(self.T_0))], [self.varpi_0 - varpi[0]]])
        e_prior[1:] = [ self.err(T[k], T[k-1], varpi[k], varpi[k-1], self.times[k] - self.times[k-1]) for k in range(1, self.K+1)]
        e_prior = np.vstack(e_prior)

        # Build the lifted transition matrix block by block
        Finv = [[None for _ in range(self.K+1)] for _ in range(self.K+1)]
        Finv[0][0] = self.E0(T[0], self.T_0)
        for k in range(1,self.K+1):
            Finv[k][k] = self.E(T[k], T[k-1], varpi[k], varpi[k-1], self.times[k]-self.times[k-1])
            Finv[k][k-1] = -self.F(T[k], T[k-1], varpi[k], varpi[k-1], self.times[k]-self.times[k-1])
        Finv = bmat(Finv, format="csr")

        # Build the lifted covariance matrix block by block
        Qinv = [[None for _ in range(self.K+1)] for _ in range(self.K+1)]
        Qinv[0][0] = self.P_0_inv
        for k in range(1,self.K+1):
            Qinv[k][k] = self.cov_inv(self.times[k] - self.times[k-1])
        Qinv = bmat(Qinv, format="csr")

        # Build LHS and RHS
        LHS = Finv.T @ Qinv @ Finv
        RHS = Finv.T @ Qinv @ e_prior

        # Return these components
        return LHS, RHS

    # Transition function
    def trans(self, delt):
        return np.kron(np.array([[1, delt], [0, 1]]), np.eye(6))

    # Covariance block
    def cov(self, delt):  
        return np.kron(np.array([[(delt**3)/3, (delt**2)/2], [(delt**2)/2, delt]]), self.Qc)

    # Inverse covariance block
    def cov_inv(self, delt): 
        return np.kron(np.array([[12*delt**(-3), -6*delt**(-2)], [-6*delt**(-2), 4/delt]]), np.linalg.inv(self.Qc))

    def err(self, T2, T1, varpi2, varpi1, delt):
        if self.error_type == 1:
            return np.block([[delt*varpi1 - se3op.tran2vec(T2 @ np.linalg.inv(T1))], \
                            [varpi1 - se3op.vec2jacinv(se3op.tran2vec(T2 @ np.linalg.inv(T1))) @ varpi2]])
        elif self.error_type == 2:
            return np.block([[delt*varpi1 - se3op.tran2vec(T2 @ np.linalg.inv(T1))], \
                            [varpi1 - varpi2]])
        else:
            e = np.block([[(varpi1 + varpi2)*delt/2 - se3op.tran2vec(T2 @ np.linalg.inv(T1))], [varpi1 - varpi2]])
            if self.order >= 2:
                e[:6] += (delt**2)/12 * se3op.curlyhat(varpi2)@varpi1
            if self.order >= 3:
                e[:6] += (delt**3)/240 * (se3op.curlyhat(varpi2)@se3op.curlyhat(varpi2)@varpi1 + se3op.curlyhat(varpi1)@se3op.curlyhat(varpi1)@varpi2)
            return e

    # "E0" Jacobian matrix - SER2 p. 460
    def E0(self, T2, T1):
        delT = T2 @ np.linalg.inv(T1)
        Jinv = se3op.vec2jacinv(se3op.tran2vec(delT))
        return np.block([[Jinv, np.zeros((6,6))], [np.zeros((6,6)), np.eye(6)]])

    # "E" Jacobian matrix - SER2 p. 460
    def E(self, T2, T1, varpi2, varpi1, delt):
        delT = T2 @ np.linalg.inv(T1)
        Jinv = se3op.vec2jacinv(se3op.tran2vec(delT))
        if self.error_type == 1:
            return np.block([[Jinv, np.zeros((6,6))], [0.5*se3op.curlyhat(varpi1) @ Jinv, Jinv]])
        elif self.error_type == 2:
            # return np.block([[Jinv, np.zeros((6,6))], [np.zeros((6,6)), Jinv]])
            return np.block([[Jinv, np.zeros((6,6))], [np.zeros((6,6)), np.eye(6)]])
        else:
            delT = T2 @ np.linalg.inv(T1)
            Jinv = se3op.vec2jacinv(se3op.tran2vec(delT))
            Eout = np.block([[Jinv, -(delt/2)*np.eye(6)], [np.zeros((6,6)), np.eye(6)]])
            if self.order >= 2:
                Eout[:6, 6:] += ((delt**2)/12)*se3op.curlyhat(varpi1)
            if self.order >= 3:
                Eout[:6, 6:] += -((delt**3)/240)*(se3op.curlyhat(varpi1)@se3op.curlyhat(varpi1) - 2*se3op.curlyhat(varpi2)@se3op.curlyhat(varpi1) + se3op.curlyhat(varpi1)@se3op.curlyhat(varpi2))
            return Eout

    # "F" Jacobian matrix - SER2 p. 460
    def F(self, T2, T1, varpi2, varpi1, delt):
        delT = T2 @ np.linalg.inv(T1)
        Jinv = se3op.vec2jacinv(se3op.tran2vec(delT))
        AdT = se3op.tranAd(delT)
        if self.error_type == 1:
            return np.block([[Jinv @ AdT, (delt) * np.eye(6)], [0.5*se3op.curlyhat(varpi1) @ Jinv @ AdT, np.eye(6)]])
        elif self.error_type == 2:
            return np.block([[Jinv @ AdT, (delt) * np.eye(6)], [np.zeros((6,6)), np.eye(6)]])
        else:
            delT = T2 @ np.linalg.inv(T1)
            Jinv = se3op.vec2jacinv(se3op.tran2vec(delT))
            AdT = se3op.tranAd(delT)
            Fout =  np.block([[Jinv @ AdT, (delt/2)*np.eye(6)], [np.zeros((6,6)), np.eye(6)]])
            if self.order >= 2:
                Fout[:6, 6:] += ((delt**2)/12)*se3op.curlyhat(varpi2)
            if self.order >= 3:
                Fout[:6, 6:] += ((delt**3)/240)*(se3op.curlyhat(varpi2)@se3op.curlyhat(varpi2) - 2*se3op.curlyhat(varpi1)@se3op.curlyhat(varpi2) + se3op.curlyhat(varpi2)@se3op.curlyhat(varpi1))
            return Fout

    def Lambda(self, t1, tau, t2):
        alpha = (tau - t1) / (t2 - t1)
        delt = t2 - t1
        return np.kron(np.array([[1-3*alpha**2+2*alpha**3, delt*(alpha-2*alpha**2+alpha**3)], [6*(-alpha+alpha**2)/delt, 1-4*alpha+3*alpha**2]]), np.eye(6))

    def Psi(self, t1, tau, t2):
        alpha = (tau - t1) / (t2 - t1)
        delt = t2 - t1
        return np.kron(np.array([[3*alpha**2-2*alpha**3, delt*(-alpha**2+alpha**3)], [6*(alpha-alpha**2)/delt, -2*alpha+3*alpha**2]]), np.eye(6))

    def interp(self, k, tau, T, varpi, P, interp_mean=True, interp_cov=True):

        T_tau, varpi_tau, P_tau = None, None, None

        # Verify tau is in the interpolation range
        if tau < self.times[k] or self.times[k+1] < tau:
            print('GPPrior.interp: invalid interpolation time')
            return None, None, None

        # If we are right at one of the boundaries, return the boundary values
        if np.abs(tau-self.times[k]) < 1e-6:
            return T[k], varpi[k], P[k][k].toarray()
        if np.abs(tau-self.times[k+1]) < 1e-6:
            return T[k+1], varpi[k+1], P[k+1][k+1].toarray()

        if interp_mean:

            if self.mean_type == 1 or self.mean_type == 2:

                # Compute the points at the two ends of the interval
                gamma_k = np.block([[np.zeros((6,1))], [varpi[k]]])
                xi_k1_k = se3op.tran2vec(T[k+1] @ np.linalg.inv(T[k]))
                gamma_k1 = np.block([[xi_k1_k], [se3op.vec2jacinv(xi_k1_k) @ varpi[k+1]]])

                # Interpolate using the old GP way
                if self.mean_type == 1:
                    # Mean interpolation setup
                    Q_tau_k = self.cov(tau - self.times[k])
                    Phi_k1_tau = self.trans(self.times[k+1] - tau)
                    Q_k1_k_inv = self.cov_inv(self.times[k+1] - self.times[k])
                    Psi = Q_tau_k @ Phi_k1_tau.T @ Q_k1_k_inv
                    Phi_tau_k = self.trans(tau - self.times[k])
                    Phi_k1_k = self.trans(self.times[k+1] - self.times[k])
                    Lambda = Phi_tau_k - Psi @ Phi_k1_k

                    # Mean interpolation
                    gamma_tau = Lambda @ gamma_k + Psi @ gamma_k1

                # Shortcut way that uses the Cubic Hermite polynomials directly
                elif self.mean_type == 2:
                    
                    gamma_tau = self.Lambda(self.times[k], tau, self.times[k+1]) @ gamma_k + self.Psi(self.times[k], tau, self.times[k+1]) @ gamma_k1

                T_tau = se3op.vec2tran(gamma_tau[:6]) @ T[k]
                varpi_tau_tmp = se3op.vec2jac(gamma_tau[:6]) @ gamma_tau[6:]

                # Handle locking the degrees of freedom that we want to lock
                varpi_tau = np.zeros((6,1))
                for i in range(6):
                    if self.lock_dof_k[i+6] == 0:
                        varpi_tau[i] = varpi_tau_tmp[i]
                    else:
                        varpi_tau[i] = varpi[k][i]

            # Interpolate the new way using a small optimization problem
            elif self.mean_type == 3:
                 # Initialize the mean interpolation using SLERP/linear interpolation
                alpha = (tau - self.times[k]) / (self.times[k+1] - self.times[k])
                T_tau = se3op.vec2tran(alpha * se3op.tran2vec(T[k+1] @ np.linalg.inv(T[k]))) @ T[k]
                varpi_tau = (1 - alpha) * varpi[k] + alpha * varpi[k+1]

                # General interpolation setup
                Q_tau_k_inv = self.cov_inv(tau - self.times[k])
                Q_k1_tau_inv = self.cov_inv(self.times[k+1] - tau)

                # Compute the projection matrix for locking some of the degrees of freedom
                Proj_k = np.eye(12)[np.array(self.lock_dof_k) == 0]

                # Iterate to improve the initial guess
                delx = 1
                iter = 0
                while np.linalg.norm(delx) > 1e-8 and iter < 100:
                    # Setup
                    E_tau_k = self.E(T_tau, T[k], varpi_tau, varpi[k], tau - self.times[k])
                    F_k1_tau = self.F(T[k+1], T_tau, varpi[k+1], varpi_tau, self.times[k+1] - tau)
                    Delta = E_tau_k.T @ Q_tau_k_inv @ E_tau_k + F_k1_tau.T @ Q_k1_tau_inv @ F_k1_tau
                    e_tau_k = self.err(T_tau, T[k], varpi_tau, varpi[k], tau - self.times[k])
                    e_k1_tau = self.err(T[k+1], T_tau, varpi[k+1], varpi_tau, self.times[k+1] - tau)
                    
                    # Solve and update (with some degrees of freedom locked)
                    ep_tau = Proj_k.T @ np.linalg.solve(Proj_k @ Delta @ Proj_k.T, Proj_k @ (E_tau_k.T @ Q_tau_k_inv @ e_tau_k - F_k1_tau.T @ Q_k1_tau_inv @ e_k1_tau) )
                    T_tau = se3op.vec2tran(ep_tau[:6]) @ T_tau
                    varpi_tau = varpi_tau + ep_tau[6:]

                    # Convergence checks
                    delx =  np.linalg.norm(ep_tau)
                    iter += 1

                    # cost = e_tau_k.T @ Q_tau_k_inv @ e_tau_k + e_k1_tau.T @ Q_k1_tau_inv @ e_k1_tau
                    # print("Cost", k, j, iter, "delx:", delx, "cost:", cost)
            
            else:
                print('GPPrior.interp: invalid mean_type')
                return None, None, None

            if interp_cov:

                # Covariance interpolation setup
                E_tau_k = self.E(T_tau, T[k], varpi_tau, varpi[k], tau - self.times[k])
                F_tau_k = self.F(T_tau, T[k], varpi_tau, varpi[k], tau - self.times[k])
                E_k1_tau = self.E(T[k+1], T_tau, varpi[k+1], varpi_tau, self.times[k+1] - tau)
                F_k1_tau = self.F(T[k+1], T_tau, varpi[k+1], varpi_tau, self.times[k+1] - tau)
                Q_tau_k_inv = self.cov_inv(tau - self.times[k])
                Q_k1_tau_inv = self.cov_inv(self.times[k+1] - tau)

                # Covariance interpolation
                Sigma = np.linalg.inv(E_tau_k.T @ Q_tau_k_inv @ E_tau_k + F_k1_tau.T @ Q_k1_tau_inv @ F_k1_tau)
                D =  np.block([E_tau_k.T @ Q_tau_k_inv @ F_tau_k, F_k1_tau.T @ Q_k1_tau_inv @ E_k1_tau])
                P_k_k = P[k][k].toarray()
                P_k_k1 = P[k][k+1].toarray()
                P_k1_k1 = P[k+1][k+1].toarray()
                P_kk_k1k1 = np.block([[P_k_k, P_k_k1], [P_k_k1.T, P_k1_k1]])
                P_tau = Sigma + Sigma @ D @ P_kk_k1k1 @ D.T @ Sigma

        # Return the interpolated mean and cov
        return T_tau, varpi_tau, P_tau

##################################################################

# Set the random seed for reproducibility
np.random.seed(42)

# Flags related to plotting
plotting = True  # Set to False to disable plotting
interp_mean = True
interp_cov = True
plotting_covariance = True
nstd = 3.762  # for chi-squared with d=3 99.7% of data lies within this number of standard deviations

# Control locking of some degrees of freedom at first time and later times
# lock_dof_0 = (1,1,1,1,1,1,1,1,1,0,0,0)  
lock_dof_0 = (0,0,0,0,0,0,0,1,1,0,0,0)  
Proj_0 = np.eye(12)[np.array(lock_dof_0) == 0]
lock_dof_k = (0,0,0,0,0,0,0,1,1,0,0,0)
Proj_k = np.eye(12)[np.array(lock_dof_k) == 0]
lock_dof_K = (0,0,0,0,0,0,0,1,1,0,0,0)  
Proj_K = np.eye(12)[np.array(lock_dof_K) == 0]

# Define a sequence of times
K_sample = 10
K = 7
tmax = 10
times = np.linspace(0, tmax, K+1)

# Define the ground-truth trajectory
K_interp = K_sample*K
times_interp = np.linspace(0, tmax, K_interp+1)
varpi_gt_interp = [ np.array([-1, 0, 0, 0, 0, -0.3*np.sin(2*np.pi*times_interp[k]/tmax)]).reshape(6,1) for k in range(K_interp+1)]
T_gt_interp = [ _ for _ in range(K_interp+1) ]
T_gt_interp[0] = np.eye(4)
for k in range(1, K_interp+1):
    T_gt_interp[k] = se3op.vec2tran(varpi_gt_interp[k]*(times_interp[k]-times_interp[k-1])) @ T_gt_interp[k-1]

T_gt = [ _ for _ in range(K+1) ]
varpi_gt = [ _ for _ in range(K+1) ]
T_gt[0] = np.eye(4)
varpi_gt[0] = np.array([-1, 0, 0, 0, 0, 0]).reshape(6,1)
for k in range(1, K+1):
    T_gt[k] = T_gt_interp[k*K_sample]
    varpi_gt[k] = varpi_gt_interp[k*K_sample]

#### Solve at the estimation times

# Start a clock for timing code
start_time = time.time()

#### Set up the GP prior

# Power spectral density matrix for motion prior
Qc = 0.008*np.diag([1, 1, 1, 0.1, 0.1, 0.1])

# Define the initial twist
varpi_0 = np.array([-1.0, 0, 0, 0, 0, 0]).reshape(6,1)

# Roll out the prior trajectory mean
T_0 = [np.eye(4)]
for k in range(1, K+1):
    T_0.append(se3op.vec2tran(varpi_0*(times[k]-times[k-1])) @ T_0[-1])

# Initialize a GP prior
gp = GPPrior(K=K, 
             times=times, 
             Qc=Qc, 
             varpi_0=varpi_0, 
             P_0_inv = 0*np.eye(12), 
             error_type=1, 
             mean_type=1,
             order=3,
             lock_dof_k=lock_dof_k
             )

##### Set up pose measurements

# Measurement covariance
R_meas = 1e-2*np.diag([1, 1, 1, 0.1, 0.1, 0.1])
Rinv = [[None for _ in range(K+1)] for _ in range(K+1)]
for k in range(K+1):
    Rinv[k][k] = np.linalg.inv(R_meas)
    
# Rinv = bmat(Rinv, format="csr")
Rinv = bmat(Rinv)

# Measurements
T_meas = [ _ for _ in range(K+1) ]
for k in range(K+1):
    T_meas[k] = se3op.vec2tran( np.sqrt(R_meas) @ np.random.randn(6,1) ) @ T_gt[k]

# Initial guess for the trajectory
varpi = [np.copy(varpi_0) for _ in range(K+1)]
T = [np.eye(4)]
for k in range(1, K+1):
    T.append(T_0[k])

#### Solver
delx = 1
iter = 0
while np.linalg.norm(delx) > 1e-6 and iter < 100:

    iter += 1

    #### GP Prior
    LHS, RHS = gp.build_lin_prior(T, varpi)
 
    #### Pose Measurements

    # Setup
    e_meas = [ None for _ in range(K+1)]
    C = [[None for _ in range(K+1)] for _ in range(K+1)]

    for k in range(K+1):
        # Compute the measurement error
        e_meas[k] = se3op.tran2vec(T[k] @ np.linalg.inv(T_meas[k]))

        # Compute the measurement Jacobian
        C[k][k] = np.block([-se3op.vec2jacinv(e_meas[k]), np.zeros((6,6))]).reshape(6,12)

    # Stack the measurement errors
    e_meas = np.vstack(e_meas)
    C = bmat(C, format="csr")

    # Update the LHS and RHS
    LHS += C.T @ Rinv @ C
    RHS += C.T @ Rinv @ e_meas

    # Build a big projection matrix to limit which degrees of freedom are allowed to move
    Proj = [[None for _ in range(K+1)] for _ in range(K+1)]
    for k in range(K+1):
        if k==0:
            Proj[k][k] = Proj_0
        elif k==K:
            Proj[k][k] = Proj_K
        else:
            Proj[k][k] = Proj_k
    Proj = bmat(Proj, format='csr')

    LHS = Proj @ LHS @ Proj.T
    RHS = Proj @ RHS

    #### Solve and update the trajectory

    # Solve the linear system using Cholesky factorization
    L = cholesky(LHS)
    delx = L.solve_A(RHS)

    delx = Proj.T @ delx

    # Convert x back into a list of elements size (6,1)
    delxi = [delx[k:k+6].reshape(6,1) for k in range(0, len(delx), 12)]
    delvarpi = [ np.block([delx[k+6:k+12]]).reshape(6,1) for k in range(0, len(delx), 12)]


    # Update the trajectory
    for k in range(K+1):
        varpi[k] += delvarpi[k]
        T[k] = se3op.vec2tran(delxi[k]) @ T[k]

    # Print the length of the change
    print("Iteration", iter, "delx:", np.linalg.norm(delx))


# Covariance of the posterior - replace with partial calculation later
LHS_inv = Proj.T @ inv(LHS) @ Proj

# Convert P into blocks
N = 12 # Proj_small.shape[0]
P = [[None for _ in range(K+1)] for _ in range(K+1)]
for k in range(K+1):
    for j in range(K+1):
        P[k][j] = LHS_inv[k*N:k*N+N, j*N:j*N+N]

# Report time for main solve
end_time = time.time()
print("Main solve time:", end_time - start_time)

if plotting:
    # Create a 3D plotter
    plotter = pv.Plotter()

    # Plot the posterior trajectory and covariances
    for k, Tk in enumerate(T):
        Tk_inv = np.linalg.inv(Tk)
        plot_se3(plotter, Tk_inv, scale=0.15, color='blue', opacity=1.0)
        Pk = P[k][k][:3,:3].toarray() # + 1e-3*np.eye(3)
        if plotting_covariance:
            plot_covariance(plotter, Tk_inv, Pk, nstd=nstd, color='blue')

    # Plot the velocities also using matplotlib
    plt.figure(figsize=(10, 5))
    ylabel = [r'$v_x$ [m/s]', r'$v_y$ [m/s]', r'$v_z$ [m/s]', r'$\omega_x$ [rad/s]', r'$\omega_y$ [rad/s]', r'$\omega_z$ [rad/s]']
    for i in range(6):
        plt.subplot(2, 3, i+1)
        v = [varpi[k][i] for k in range(K+1)]
        v_cov_plus = [varpi[k][i] + 3*np.sqrt(P[k][k][i+6,i+6]) for k in range(K+1)]
        v_cov_minus = [varpi[k][i] - 3*np.sqrt(P[k][k][i+6,i+6]) for k in range(K+1)]
        plt.plot(times, v, 'b.')
        if lock_dof_k[i+6] == 0:
            plt.plot(times, v_cov_plus, 'b.', alpha=0.2)
            plt.plot(times, v_cov_minus, 'b.', alpha=0.2)
        plt.xlabel('t [s]')
        plt.ylabel(ylabel[i])
    plt.tight_layout()


# Time the interpolation
start_time = time.time()

# Interpolate and plot
if interp_mean:
    for k in range(K):
        J = K_sample + 1

        # Save the interpolated velocity and covariance
        tau_save = np.zeros((J,))
        varpi_tau_save = np.zeros((J, 6))
        varpi_tau_std_save = np.zeros((J, 6))

        for j in range(J):
            # Get the interpolated time
            tau = times[k] + j / (J-1) * (times[k+1] - times[k]) 
            
            # Interpolate
            T_tau, varpi_tau, P_tau = gp.interp(k, tau, T, varpi, P, interp_mean=interp_mean, interp_cov=interp_cov)

            # Plot the interpolated pose        
            if plotting and j > 0 and j < J-1:
                T_tau_inv = np.linalg.inv(T_tau)
                plot_se3(plotter, T_tau_inv, scale=0.1, color='blue', opacity=0.3)

                # Plot the interpolated covariance ellipsoid
                if interp_cov and plotting_covariance:    
                    plot_covariance(plotter, T_tau_inv, P_tau[:3,:3], nstd=nstd, color='blue', opacity=0.05)

            # Save the interpolated velocity and covariance
            tau_save[j] = tau
            varpi_tau_save[j, :] = varpi_tau.flatten()
            if interp_cov:
                varpi_tau_std_save[j, :] = np.sqrt(np.diag(P_tau[6:, 6:]))

        if interp_mean and plotting:
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.plot(tau_save, varpi_tau_save[:, i], 'b-')
                if lock_dof_k[i+6] == 0:
                    plt.plot(tau_save, varpi_tau_save[:, i] + 3*varpi_tau_std_save[:, i], 'b-', alpha=0.2)
                    plt.plot(tau_save, varpi_tau_save[:, i] - 3*varpi_tau_std_save[:, i], 'b-', alpha=0.2)

# Report time to interpolate
end_time = time.time()
print("Interpolation time:", end_time - start_time)

if plotting:

    # Plot the ground truth poses
    for k, Tk in enumerate(T_gt_interp):
        Tk_inv = np.linalg.inv(Tk)
        plot_se3(plotter, Tk_inv, scale=0.1, opacity=0.2, color='green')
        
    # Plot the ground truth velocities
    for i in range(6):
        plt.subplot(2, 3, i+1)
        v = [varpi_gt_interp[k][i] for k in range(K_interp+1)]
        plt.plot(times_interp, v, 'g-', alpha=0.3)

    # Plot the measurements 
    for k, Tk in enumerate(T_meas):
        Tk_inv = np.linalg.inv(Tk)
        plot_se3(plotter, Tk_inv, scale=0.15, opacity= 1.0, color='red')

    # Set the plotter view to a specific angle
    plotter.view_xy()

    # Add key events
    plotter.add_key_event("1", plotter.view_yz)
    plotter.add_key_event("2", plotter.view_xz)
    plotter.add_key_event("3", plotter.view_xy)
    plotter.add_key_event("4", plotter.view_isometric)
    plotter.add_key_event("q", plotter.close)
    plotter.add_key_event("s", save_image)

    # Show the matplotlib plots without blocking
    for i in range(3, 6):
        plt.subplot(2, 3, i+1)
        plt.ylim(-0.4, 0.4)
    plt.show(block=False)
    
    # Export the plot to a PDF file
    plt.savefig('se3_velocity_example.pdf', format='pdf', bbox_inches='tight')  

    # Show the 3D plotter
    plotter.show()

 




