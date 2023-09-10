# Stereo-Inertial-Visual-Odometry-VIO-for-Aerial-Robotics
This repository showcases my work on estimating aerial robotic poses using onboard IMU and stereo cameras. Leveraging the Error State Kalman Filter (ESKF) algorithm, I fused data streams to generate real-time position and orientation trajectories. Dataset used: EuRoc from ETH Zurich.

## Overview

### Quadrotor Pose Estimation using IMU and Stereo Cameras

In this project, I delved into the challenge of estimating the pose of a quadrotor platform using data from its onboard IMU and stereo cameras. Utilizing the Error State Kalman Filter (ESKF) algorithm, I integrated the information from both these sensors. A substantial portion of my data insights was derived from the EuRoc dataset, originating from ETH Zurich.

## Code Structure
My code repository is meticulously organized, with the following key components:
- **setup.py**: For necessary package installations.
- **proj2_3 package**:
  - **util**: Contains tests to validate the code using `test.py`.
  - **dataset**: Features a segment of the EuRoc dataset with stereo imagery and IMU readings.
  - **code**: Houses primary code files. My core contribution is in `vio.py` where essential steps of the ESKF algorithm are implemented.

## Key Tasks & Implementations
- **Nominal State Update**: Used the ESKF algorithm to update the nominal state based on IMU measurements. Inputs included the current nominal state, angular velocity, linear acceleration, and timestep duration. The outputs consisted of updated system states for position, velocity, rotation, biases, and the gravity vector.
  
- **Covariance Update**: Adjusted the error state covariance matrix post each IMU update. Crucial parameters for noise covariance matrix adjustments included accelerometer noise density, gyroscope noise density, accelerometer random walk, and gyroscope random walk.

- **Measurement Update Step**: Designed to refine both the nominal state and the error state covariance matrix with image measurement data. The innovation vector, determined as the difference between measured and predicted image coordinates, was central to this process.

## Mathematical Derivation
One of the notable achievements was proving an assertion from the lecture materials. I showcased that a unit quaternion, constructed from a unit vector `ω ∈ R3` and an angle `θ`, and processed through the function `R = H(u_0, u)`, results in a rotation matrix `R ∈ SO(3)`. This matrix aligns with the one acquired through the Rodrigues Formula: `R = exp(ω^θ)`. This mathematical consistency affirmed the theoretical robustness of our strategy.

![attitude and position of quad](https://github.com/Saibernard/Stereo-Inertial-Visual-Odometry-VIO-for-Aerial-Robotics/assets/112599512/1ca02279-3282-4736-9676-80e2414d5b8b)

![velocity of quad](https://github.com/Saibernard/Stereo-Inertial-Visual-Odometry-VIO-for-Aerial-Robotics/assets/112599512/07c34054-d1fb-4529-b66b-a57fc459a73a)

![accelerometer bias](https://github.com/Saibernard/Stereo-Inertial-Visual-Odometry-VIO-for-Aerial-Robotics/assets/112599512/0876fb52-248b-41e9-83a7-d5ce32c2c862)

![gyroscope bias](https://github.com/Saibernard/Stereo-Inertial-Visual-Odometry-VIO-for-Aerial-Robotics/assets/112599512/a7c3d408-12cd-422c-9a44-7fba7059e740)

![g_vector](https://github.com/Saibernard/Stereo-Inertial-Visual-Odometry-VIO-for-Aerial-Robotics/assets/112599512/28d80e95-0dd6-48ca-a88d-d340a8d0fb0d)

![trace of covariance matrix](https://github.com/Saibernard/Stereo-Inertial-Visual-Odometry-VIO-for-Aerial-Robotics/assets/112599512/32b31b39-3122-44f8-b457-f501739bf5a5)


## Conclusion
Embarking on the journey of amalgamating data from stereo cameras and IMU for a quadrotor's pose estimation proved both challenging and enlightening. My adept application of the ESKF algorithm, fortified by rigorous mathematical validation, underlined the potency of merging visual and inertial data for pinpoint pose estimation. This endeavor is promising for real-world implementations, particularly in situations necessitating steadfast and real-time aerial robotic navigation.

