import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.1  # time step
T = 20  # total time
N = int(T / dt)  # number of steps
r = 5  # radius of the circle
omega = 2 * np.pi / T  # angular velocity

# True trajectory
t = np.linspace(0, T, N)
x_true = r * np.cos(omega * t)
y_true = r * np.sin(omega * t)
trajectory_true = np.vstack((x_true, y_true))

# IMU and GPS noise parameters
imu_noise_std = 0.1
gps_noise_std = 0.5
imu_bias = np.array([0.01, 0.01])
gps_bias = np.array([0.2, 0.2])

# Simulate IMU (accelerometer) measurements
imu_measurements = np.vstack((np.diff(np.diff(x_true) / dt) / dt, np.diff(np.diff(y_true) / dt) / dt)).T + imu_noise_std * np.random.randn(N-2, 2) + imu_bias

# Simulate GPS measurements
gps_measurements = trajectory_true.T + gps_noise_std * np.random.randn(N, 2) + gps_bias

# Kalman Filter
class KalmanFilter:
    def __init__(self, dt, std_acc, std_meas):
        self.dt = dt
        self.std_acc = std_acc
        self.std_meas = std_meas
        
        self.A = np.eye(4) + np.array([[0, 0, dt, 0],
                                       [0, 0, 0, dt],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]])
        
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.Q = std_acc**2 * np.array([[dt**4/4, 0, dt**3/2, 0],
                                        [0, dt**4/4, 0, dt**3/2],
                                        [dt**3/2, 0, dt**2, 0],
                                        [0, dt**3/2, 0, dt**2]])
        
        self.R = std_meas**2 * np.eye(2)
        self.P = np.eye(4)
        self.x = np.zeros(4)
        
    def predict(self, u):
        B = np.array([[dt**2 / 2, 0],
                      [0, dt**2 / 2],
                      [dt, 0],
                      [0, dt]])
        self.x = self.A @ self.x + B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# Extended Kalman Filter
class EKF(KalmanFilter):
    def __init__(self, dt, std_acc, std_meas):
        super().__init__(dt, std_acc, std_meas)
    
    def predict(self, u):
        # Non-linear state transition
        self.x[0] += self.x[2] * self.dt + 0.5 * u[0] * self.dt**2
        self.x[1] += self.x[3] * self.dt + 0.5 * u[1] * self.dt**2
        self.x[2] += u[0] * self.dt
        self.x[3] += u[1] * self.dt
        # Jacobian of the state transition
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Q

# Initialize filters
kf = KalmanFilter(dt, imu_noise_std, gps_noise_std)
ekf = EKF(dt, imu_noise_std, gps_noise_std)

kf_estimates = np.zeros((N, 2))
ekf_estimates = np.zeros((N, 2))

# Run the filters
for i in range(1, N-1):
    u = imu_measurements[i-1]
    kf.predict(u)
    ekf.predict(u)
    
    kf.update(gps_measurements[i])
    ekf.update(gps_measurements[i])
    
    kf_estimates[i] = kf.x[:2]
    ekf_estimates[i] = ekf.x[:2]

# Calculate RMSE
kf_rmse = np.sqrt(np.mean((kf_estimates - trajectory_true.T)**2))
ekf_rmse = np.sqrt(np.mean((ekf_estimates - trajectory_true.T)**2))

# Print RMSE
print(f'Kalman Filter RMSE: {kf_rmse}')
print(f'Extended Kalman Filter RMSE: {ekf_rmse}')

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(x_true, y_true, 'g-', label='True trajectory')
plt.plot(gps_measurements[:, 0], gps_measurements[:, 1], 'r.', label='GPS measurements')
plt.plot(kf_estimates[:, 0], kf_estimates[:, 1], 'b-', label='Kalman Filter estimates')
plt.plot(ekf_estimates[:, 0], ekf_estimates[:, 1], 'c--', label='EKF estimates')
plt.legend()
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Robot trajectory estimation with Kalman Filter and EKF')
plt.show()
