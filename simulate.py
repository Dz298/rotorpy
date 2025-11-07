from enum import Enum
import copy
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

class ExitStatus(Enum):
    """ Exit status values indicate the reason for simulation termination. """
    COMPLETE     = 'Success: End reached.'
    TIMEOUT      = 'Timeout: Simulation end time reached.'
    INF_VALUE    = 'Failure: Your controller returned inf motor speeds.'
    NAN_VALUE    = 'Failure: Your controller returned nan motor speeds.'
    OVER_SPEED   = 'Failure: Your quadrotor is out of control; it is going faster than 100 m/s. The Guinness World Speed Record is 73 m/s.'
    OVER_SPIN    = 'Failure: Your quadrotor is out of control; it is spinning faster than 100 rad/s. The onboard IMU can only measure up to 52 rad/s (3000 deg/s).'
    FLY_AWAY     = 'Failure: Your quadrotor is out of control; it flew away with a position error greater than 20 meters.'
    COLLISION    = 'Failure: Your quadrotor collided with an object.'

def simulate(world, initial_state, vehicle, controller, trajectory, wind_profile, imu, mocap, estimator, t_final, t_step, safety_margin, use_mocap, terminate=None, ext_force=np.array([0,0,0]), ext_torque=np.array([0,0,0]), 
            disturbance_toggle_times=None):
    """
    Perform a vehicle simulation and return the numerical results.

    Inputs:
        world, a class representing the world it is flying in, including objects and world bounds. 
        initial_state, a dict defining the vehicle initial conditions with appropriate keys
        vehicle, Vehicle object containing the dynamics
        controller, Controller object containing the controller
        trajectory, Trajectory object containing the trajectory to follow
        wind_profile, Wind Profile object containing the wind generator. 
        t_final, maximum duration of simulation, s
        t_step, the time between each step in the simulator, s
        safety_margin, the radius of the ball surrounding the vehicle position to determine if a collision occurs
        imu, IMU object that generates accelerometer and gyroscope readings from the vehicle state
        terminate, None, False, or a function of time and state that returns
            ExitStatus. If None (default), terminate when hover is reached at
            the location of trajectory with t=inf. If False, never terminate
            before timeout or error. If a function, terminate when returns not
            None.
        mocap, a MotionCapture object that provides noisy measurements of pose and twist with artifacts. 
        use_mocap, a boolean to determine in noisy measurements from mocap should be used for quadrotor control
        estimator, an estimator object that provides estimates of a portion or all of the vehicle state.
        ext_force, external force applied to the vehicle, shape=(3,)
        ext_torque, external torque applied to the vehicle, shape=(3,)
        disturbance_toggle_times: list of float, predetermined times when disturbances should be toggled on/off. If None, disturbances are always applied (not toggled).

    Outputs:
        time, seconds, shape=(N,)
        state, a dict describing the state history with keys
            x, position, m, shape=(N,3)
            v, linear velocity, m/s, shape=(N,3)
            q, quaternion [i,j,k,w], shape=(N,4)
            w, angular velocity, rad/s, shape=(N,3)
            rotor_speeds, motor speeds, rad/s, shape=(N,n) where n is the number of rotors
            wind, wind velocity, m/s, shape=(N,3)
        control, a dict describing the command input history with keys
            cmd_motor_speeds, motor speeds, rad/s, shape=(N,4)
            cmd_q, commanded orientation (not used by simulator), quaternion [i,j,k,w], shape=(N,4)
            cmd_w, commanded angular velocity (not used by simulator), rad/s, shape=(N,3)
        flat, a dict describing the desired flat outputs from the trajectory with keys
            x,        position, m
            x_dot,    velocity, m/s
            x_ddot,   acceleration, m/s**2
            x_dddot,  jerk, m/s**3
            x_ddddot, snap, m/s**4
            yaw,      yaw angle, rad
            yaw_dot,  yaw rate, rad/s
        imu_measurements, a dict containing the biased and noisy measurements from an accelerometer and gyroscope
            accel,  accelerometer, m/s**2
            gyro,   gyroscope, rad/s
        imu_gt, a dict containing the ground truth (no noise, no bias) measurements from an accelerometer and gyroscope
            accel,  accelerometer, m/s**2
            gyro,   gyroscope, rad/s
        mocap_measurements, a dict containing noisy measurements of pose and twist for the vehicle. 
            x, position (inertial)
            v, velocity (inertial)
            q, orientation of body w.r.t. inertial frame.
            w, body rates in the body frame. 
        exit_status, an ExitStatus enum indicating the reason for termination.
    """

    # Initialize state
    time = [0]
    state = [copy.deepcopy(initial_state)]
    state[0]['wind'] = wind_profile.update(0, state[0]['x'])
    
    # Initialize disturbance states
    force_on = False
    torque_on = False
    payload_attached = False  # Track if payload is currently attached
    payload_torque = np.array([0,0,0])
    current_ext_force = np.array([0,0,0])
    current_ext_torque = np.array([0,0,0])
    # Store payload mass and position for visualization (before it gets modified by attach/detach)
    payload_mass_value = vehicle.payload_mass if hasattr(vehicle, 'payload_mass') else 0.0
    payload_position_value = vehicle.payload_position.copy() if hasattr(vehicle, 'payload_position') and payload_mass_value > 0 else np.array([0.0, 0.0, 0.0])
    state[0]['ext_force'] = current_ext_force
    state[0]['ext_torque'] = current_ext_torque
    state[0]['accel'] = np.array([0,0,0])
    state[0]['gyro'] = np.array([0,0,0])
    state[0]['payload_attached'] = np.bool_(payload_attached)  # Track payload attachment state for visualization
    state[0]['payload_mass'] = payload_mass_value  # Store payload mass for visualization
    
    # Initialize toggle time tracking
    toggle_times = disturbance_toggle_times if disturbance_toggle_times is not None else []
    toggle_index = 0  # Index into toggle_times list

    if terminate is None:    # Default exit. Terminate at final position of trajectory.
        normal_exit = traj_end_exit(initial_state, trajectory, using_vio = False)
    elif terminate is False: # Never exit before timeout.
        normal_exit = lambda t, s: None
    else:                    # Custom exit.
        normal_exit = terminate

    imu_measurements = []
    mocap_measurements = []
    imu_gt = []
    state_estimate = []
    flat    = [trajectory.update(time[-1])]
    mocap_measurements.append(mocap.measurement(state[-1], with_noise=True, with_artifacts=False))
    if use_mocap:
        # In this case the controller will use the motion capture estimate of the pose and twist for control. 
        control = [controller.update(time[-1], mocap_measurements[-1], flat[-1])]
    else:
        control = [controller.update(time[-1], state[-1], flat[-1])]
    state_dot =  vehicle.statedot(state[0], control[0], t_step)
    imu_measurements.append(imu.measurement(state[-1], state_dot, with_noise=True))
    imu_gt.append(imu.measurement(state[-1], state_dot, with_noise=False))
    state_estimate.append(estimator.step(state[0], control[0], imu_measurements[0], mocap_measurements[0]))

    exit_status = None

    while True:
        exit_status = exit_status or safety_exit(world, safety_margin, state[-1], flat[-1], control[-1])
        exit_status = exit_status or normal_exit(time[-1], state[-1])
        exit_status = exit_status or time_exit(time[-1], t_final)
        if exit_status:
            break
        time.append(time[-1] + t_step)
        
        # Check if we should toggle disturbances based on predetermined times
        # Check after updating time so we use the current time
        should_toggle = False
        if toggle_times and toggle_index < len(toggle_times):
            # Check if current time has passed the next toggle time
            # Handle multiple toggles that may have occurred in one time step
            while toggle_index < len(toggle_times) and time[-1] >= toggle_times[toggle_index]:
                should_toggle = not should_toggle  # Toggle for each time that has passed
                toggle_index += 1
        
        # Toggle disturbances if needed
        if should_toggle:
            force_on = not force_on
            torque_on = not torque_on      
        
        # Handle payload attachment/detachment
        # Payload can be toggled independently if it's configured (payload_mass > 0)
        # When toggle_times exist, payload toggles with force/torque (same toggle state)
        # This works for both payload-only and force/torque experiments
        # Use payload_mass_value (stored at start) instead of vehicle.payload_mass
        # because detach_payload() resets vehicle.payload_mass to 0
        has_payload = payload_mass_value > 0
        
        if has_payload:
            # Determine if payload should be attached based on toggle state
            # When toggle_times exist, payload follows the same toggle as force/torque
            # This means payload attaches when force_on or torque_on is True
            should_attach_payload = torque_on or force_on
            
            if should_attach_payload and not payload_attached:
                # Attach payload only if not already attached
                # Restore payload_mass and position if they were reset by detach_payload()
                if vehicle.payload_mass == 0:
                    vehicle.payload_mass = payload_mass_value
                    vehicle.payload_position = payload_position_value.copy()
                vehicle.attach_payload()
                r_payload = vehicle.payload_position
                r_payload_to_com = r_payload - vehicle.com
                g_force = np.array([0, 0, -vehicle.payload_mass * vehicle.g])
                payload_torque = np.cross(r_payload_to_com, g_force)
                payload_attached = True
            elif not should_attach_payload and payload_attached:
                # Detach payload only if currently attached
                vehicle.detach_payload()
                payload_torque = np.array([0,0,0])
                payload_attached = False
        
        # Apply current disturbances
        if toggle_times:
            # Toggle disturbances on/off based on predetermined times
            current_ext_force = ext_force if force_on else np.array([0,0,0])
            current_ext_torque = ext_torque + payload_torque if torque_on else np.array([0,0,0])
        else:
            # No toggling - always apply disturbances if they are non-zero
            current_ext_force = ext_force
            current_ext_torque = ext_torque
        state[-1]['wind'] = wind_profile.update(time[-1], state[-1]['x'])
        # Update state with current disturbances
        state[-1]['ext_force'] = current_ext_force
        state[-1]['ext_torque'] = current_ext_torque
        state[-1]['payload_attached'] = np.bool_(payload_attached)  # Track payload attachment state for visualization
        state[-1]['payload_mass'] = payload_mass_value  # Store payload mass for visualization
        state.append(vehicle.step(state[-1], control[-1], t_step))
        state[-1]['accel'] = imu_gt[-1]['accel']
        state[-1]['gyro'] = imu_gt[-1]['gyro']
        # Update state with current disturbances
        state[-1]['ext_force'] = current_ext_force
        state[-1]['ext_torque'] = current_ext_torque
        state[-1]['payload_attached'] = np.bool_(payload_attached)  # Track payload attachment state for visualization
        state[-1]['payload_mass'] = payload_mass_value  # Store payload mass for visualization
        flat.append(trajectory.update(time[-1]))
        mocap_measurements.append(mocap.measurement(state[-1], with_noise=True, with_artifacts=mocap.with_artifacts))
        state_estimate.append(estimator.step(state[-1], control[-1], imu_measurements[-1], mocap_measurements[-1]))
        if use_mocap:
            control.append(controller.update(time[-1], mocap_measurements[-1], flat[-1]))
        else:
            control.append(controller.update(time[-1], state[-1], flat[-1]))
        state_dot = vehicle.statedot(state[-1], control[-1], t_step)
        imu_measurements.append(imu.measurement(state[-1], state_dot, with_noise=True))
        imu_gt.append(imu.measurement(state[-1], state_dot, with_noise=False))

    time    = np.array(time, dtype=float)    
    state   = merge_dicts(state)
    imu_measurements = merge_dicts(imu_measurements)
    imu_gt = merge_dicts(imu_gt)
    mocap_measurements = merge_dicts(mocap_measurements)
    control         = merge_dicts(control)
    flat            = merge_dicts(flat)
    state_estimate  = merge_dicts(state_estimate)

    return (time, state, control, flat, imu_measurements, imu_gt, mocap_measurements, state_estimate, exit_status)

def merge_dicts(dicts_in):
    """
    Concatenates contents of a list of N state dicts into a single dict by
    prepending a new dimension of size N. This is more convenient for plotting
    and analysis. Requires dicts to have consistent keys and have values that
    are numpy arrays.
    """
    dict_out = {}
    for k in dicts_in[0].keys():
        dict_out[k] = []
        for d in dicts_in:
            dict_out[k].append(d[k])
        dict_out[k] = np.array(dict_out[k])
    return dict_out


def traj_end_exit(initial_state, trajectory, using_vio = False):
    """
    Returns a exit function. The exit function returns an exit status message if
    the quadrotor is near hover at the end of the provided trajectory. If the
    initial state is already at the end of the trajectory, the simulation will
    run for at least one second before testing again.
    """

    xf = trajectory.update(np.inf)['x']
    yawf = trajectory.update(np.inf)['yaw']
    rotf = Rotation.from_rotvec(yawf * np.array([0, 0, 1])) # create rotation object that describes yaw
    if np.array_equal(initial_state['x'], xf):
        min_time = 1.0
    else:
        min_time = 0

    def exit_fn(time, state):
        cur_attitude = Rotation.from_quat(state['q'])
        err_attitude = rotf * cur_attitude.inv() # Rotation between current and final
        angle = norm(err_attitude.as_rotvec()) # angle in radians from vertical
        # Success is reaching near-zero speed with near-zero position error.
        if using_vio:
            # set larger threshold for VIO due to noisy measurements
            if time >= min_time and norm(state['x'] - xf) < 1 and norm(state['v']) <= 1 and angle <= 1:
                return ExitStatus.COMPLETE
        else:
            if time >= min_time and norm(state['x'] - xf) < 0.02 and norm(state['v']) <= 0.03 and angle <= 0.02:
                return ExitStatus.COMPLETE
        return None
    return exit_fn

def time_exit(time, t_final):
    """
    Return exit status if the time exceeds t_final, otherwise None.
    """
    if time >= t_final:
        return ExitStatus.TIMEOUT
    return None

def safety_exit(world, margin, state, flat, control):
    """
    Return exit status if any safety condition is violated, otherwise None.
    """
    if np.any(np.abs(state['v']) > 20):
        return ExitStatus.OVER_SPEED
    if np.any(np.abs(state['w']) > 100):
        return ExitStatus.OVER_SPIN

    if len(world.world.get('blocks', [])) > 0:
        # If a world has objects in it we need to check for collisions.  
        collision_pts = world.path_collisions(state['x'], margin)
        no_collision = collision_pts.size == 0
        if not no_collision:
            return ExitStatus.COLLISION
    return None