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
x_true = r * np.cos(omega * t) - r
y_true = r * np.sin(omega * t)
trajectory_true = np.vstack((x_true, y_true))

# IMU and GPS noise parameters
imu_acc_noise_std = 0.1
imu_gyro_noise_std = 0.01
gps_noise_std = 0.5
imu_acc_bias = np.array([0.01, 0.01])
imu_gyro_bias = 0.001
gps_bias = np.array([0.2, 0.2])

# Simulate IMU (accelerometer and gyroscope) measurements
acc_x_true = -r * omega**2 * np.cos(omega * t)
acc_y_true = -r * omega**2 * np.sin(omega * t)
gyro_true = np.ones(N) * omega

imu_acc_measurements = np.vstack((acc_x_true, acc_y_true)).T + imu_acc_noise_std * np.random.randn(N, 2) + imu_acc_bias
imu_gyro_measurements = gyro_true + imu_gyro_noise_std * np.random.randn(N) + imu_gyro_bias

# Simulate GPS measurements
gps_measurements = trajectory_true.T + gps_noise_std * np.random.randn(N, 2) + gps_bias

# Kalman Filter
class KalmanFilter:
    def __init__(self, dt, std_acc, std_gyro, std_meas, initial_state):
        self.dt = dt
        self.std_acc = std_acc
        self.std_gyro = std_gyro
        self.std_meas = std_meas
        
        self.A = np.eye(5) + np.array([[0, 0, dt, 0, 0],
                                       [0, 0, 0, dt, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]])
        
        self.H = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])
        
        self.Q = np.diag([std_acc**2, std_acc**2, std_gyro**2, std_gyro**2, std_gyro**2])
        self.R = std_meas**2 * np.eye(2)
        self.P = np.eye(5)
        self.x = np.array(initial_state)
        
    def predict(self, acc, gyro):
        B = np.array([[0.5 * self.dt**2, 0.5 * self.dt**2, 0],
                      [0.5 * self.dt**2, 0.5 * self.dt**2, 0],
                      [self.dt, self.dt, 0],
                      [self.dt, self.dt, 0],
                      [0, 0, self.dt]])
        u = np.array([acc[0], acc[1], gyro])
        self.x = self.A @ self.x + B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ self.H) @ self.P

# Extended Kalman Filter
class EKF(KalmanFilter):
    def __init__(self, dt, std_acc, std_gyro, std_meas, initial_state):
        super().__init__(dt, std_acc, std_gyro, std_meas, initial_state)
    
    def predict(self, acc, gyro):
        # Non-linear state transition
        theta = self.x[4]
        self.x[0] += self.x[2] * self.dt + 0.5 * acc[0] * self.dt**2 * np.cos(theta) - 0.5 * acc[1] * self.dt**2 * np.sin(theta)
        self.x[1] += self.x[3] * self.dt + 0.5 * acc[0] * self.dt**2 * np.sin(theta) + 0.5 * acc[1] * self.dt**2 * np.cos(theta)
        self.x[2] += acc[0] * self.dt * np.cos(theta) - acc[1] * self.dt * np.sin(theta)
        self.x[3] += acc[0] * self.dt * np.sin(theta) + acc[1] * self.dt * np.cos(theta)
        self.x[4] += gyro * self.dt
        # Jacobian of the state transition
        F = np.array([[1, 0, self.dt, 0, -0.5 * acc[0] * self.dt**2 * np.sin(theta) - 0.5 * acc[1] * self.dt**2 * np.cos(theta)],
                      [0, 1, 0, self.dt, 0.5 * acc[0] * self.dt**2 * np.cos(theta) - 0.5 * acc[1] * self.dt**2 * np.sin(theta)],
                      [0, 0, 1, 0, -acc[0] * self.dt * np.sin(theta) - acc[1] * self.dt * np.cos(theta)],
                      [0, 0, 0, 1, acc[0] * self.dt * np.cos(theta) - acc[1] * self.dt * np.sin(theta)],
                      [0, 0, 0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Q

# Initialize filters for the four cases
initial_state = [0, 0, 0, 0, 0]  # starting point of the true trajectory with initial velocities and angle set to 0
kf_imu = KalmanFilter(dt, imu_acc_noise_std, imu_gyro_noise_std, gps_noise_std, initial_state)
ekf_imu = EKF(dt, imu_acc_noise_std, imu_gyro_noise_std, gps_noise_std, initial_state)
kf_imu_gps = KalmanFilter(dt, imu_acc_noise_std, imu_gyro_noise_std, gps_noise_std, initial_state)
ekf_imu_gps = EKF(dt, imu_acc_noise_std, imu_gyro_noise_std, gps_noise_std, initial_state)

kf_imu_estimates = np.zeros((N, 2))
ekf_imu_estimates = np.zeros((N, 2))
kf_imu_gps_estimates = np.zeros((N, 2))
ekf_imu_gps_estimates = np.zeros((N, 2))

# Run the filters
for i in range(1, N):
    acc = imu_acc_measurements[i]
    gyro = imu_gyro_measurements[i]
    
    # KF (IMU only)
    kf_imu.predict(acc, gyro)
    kf_imu.update(kf_imu.x[:2])  # Use predicted state for update
    kf_imu_estimates[i] = kf_imu.x[:2]
    
    # EKF (IMU only)
    ekf_imu.predict(acc, gyro)
    ekf_imu.update(ekf_imu.x[:2])  # Use predicted state for update
    ekf_imu_estimates[i] = ekf_imu.x[:2]
    
    # KF (IMU + GPS)
    kf_imu_gps.predict(acc, gyro)
    kf_imu_gps.update(gps_measurements[i])
    kf_imu_gps_estimates[i] = kf_imu_gps.x[:2]
    
    # EKF (IMU + GPS)
    ekf_imu_gps.predict(acc, gyro)
    ekf_imu_gps.update(gps_measurements[i])
    ekf_imu_gps_estimates[i] = ekf_imu_gps.x[:2]

# Calculate RMSE
kf_imu_rmse = np.sqrt(np.mean((kf_imu_estimates - trajectory_true.T)**2))
ekf_imu_rmse = np.sqrt(np.mean((ekf_imu_estimates - trajectory_true.T)**2))
kf_imu_gps_rmse = np.sqrt(np.mean((kf_imu_gps_estimates - trajectory_true.T)**2))
ekf_imu_gps_rmse = np.sqrt(np.mean((ekf_imu_gps_estimates - trajectory_true.T)**2))

# Print RMSE
print(f'Kalman Filter (IMU only) RMSE: {kf_imu_rmse}')
print(f'Extended Kalman Filter (IMU only) RMSE: {ekf_imu_rmse}')
print(f'Kalman Filter (IMU + GPS) RMSE: {kf_imu_gps_rmse}')
print(f'Extended Kalman Filter (IMU + GPS) RMSE: {ekf_imu_gps_rmse}')

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(x_true, y_true, 'g-', label='True trajectory')
plt.plot(gps_measurements[:, 0], gps_measurements[:, 1], 'r.', label='GPS measurements')
plt.plot(kf_imu_estimates[:, 0], kf_imu_estimates[:, 1], 'b-', label='KF (IMU only)')
plt.plot(ekf_imu_estimates[:, 0], ekf_imu_estimates[:, 1], 'c-', label='EKF (IMU only)')
plt.plot(kf_imu_gps_estimates[:, 0], kf_imu_gps_estimates[:, 1], 'm-', label='KF (IMU + GPS)')
plt.plot(ekf_imu_gps_estimates[:, 0], ekf_imu_gps_estimates[:, 1], 'y-', label='EKF (IMU + GPS)')
plt.scatter(x_true[0], y_true[0], color='k', marker='o', s=100, label='Start point')
plt.legend()
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Robot trajectory estimation with KF and EKF (IMU only and IMU + GPS)')
plt.show()
