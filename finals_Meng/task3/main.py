import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from tracker_model import TrackerModel



def gravity_cancellation(dyn_model,q_,qd_, u):
    """
    Perform feedback solely the dynamic cancellation  on a robotic system.

    Parameters:
    - dyn_model (pin_wrapper): The dynamics model of the robot encapsulated within a 'pin_wrapper' object,
                               which provides methods for computing robot dynamics such as mass matrices,
                               Coriolis forces, etc.
    - u (numpy.ndarray): The control input to be applied to the robot, computed by a higher-level controller

    Returns:
    None

    This function computes the control inputs necessary to achieve desired joint positions and velocities by
    applying feedback linearization, using the robot's dynamic model to appropriately compensate for its
    inherent dynamics. The control law implemented typically combines proportional-derivative (PD) control
    with dynamic compensation to achieve precise and stable motion.
    """
 
    # here i compute the feeback linearization tau // the reordering is already done inside compute all teamrs
    dyn_model.ComputeAllTerms(q_, qd_)

    # control 
    tau_gravity_cancellation = dyn_model.res.g + u

    tau_FL = dyn_model._FromPinToExtVec(tau_FL)
  
    return tau_FL
constraints_flag = True


def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")
    


def main():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_d_all,qd_d_all, q_mes_all, qd_mes_all, u_mpc_all, time_all = [], [], [], [], [], []
    

    # Define the matrices
    #A, B = getSystemMatricesContinuos(num_joints)
    #Q, R = getCostMatrices(num_joints)
    
    # Measuring all the state
    num_states = 2 * num_joints
    #C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    tracker = TrackerModel( N_mpc, num_states, num_joints, num_states, sim.GetTimeStep(),constr_flag=constraints_flag)
    tracker.setSystemMatrices()
    p_w = 10000
    v_w = 10
    Qcoeff = np.array([p_w, p_w, p_w,p_w, p_w, p_w,p_w, v_w, v_w, v_w,v_w, v_w, v_w,v_w])
    Rcoeff = [0.1] * num_joints
    tracker.setCostMatrices(Qcoeff, Rcoeff)
    # Compute the matrices needed for MPC optimization
    S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar = tracker.propagation_model_tracker_fixed_std()
    H,Ftra = tracker.tracker_std(S_bar, T_bar, Q_hat, Q_bar, R_bar)
    # Define the constraints
    num_controls = num_joints
    joint_position_constr= np.array([4*np.pi]*num_controls)
    joint_velocity_constr = np.array([80]*num_controls)
    # stack joint position and velocity constraints
    max_state_constraints = np.hstack((joint_position_constr, joint_velocity_constr))
    min_state_constraints = np.hstack((-joint_position_constr, -joint_velocity_constr))
    B_in = {'max': np.array([35] * num_controls), 'min': np.array([-35] * num_controls)}
    B_out = {'max': max_state_constraints, 'min': min_state_constraints}
    tracker.setConstraintsMatrices(B_in,B_out,S_bar_C,T_bar_C)
    
    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference


    # Main control loop
    episode_duration = 5 # duration in seconds
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    sim.ResetPose()
    
    # testing loop
    u_mpc = np.zeros(num_joints)
    u_star = np.zeros(num_joints*N_mpc)
    for i in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        x_ref = []
        # generate the predictive trajectory for N steps
        for j in range(N_mpc):
            q_d, qd_d = ref.get_values(current_time + j*time_step)
            
            # here i need to stack the q_d and qd_d
            x_ref.append(np.vstack((q_d.reshape(-1, 1), qd_d.reshape(-1, 1))))
        
        x_ref = np.vstack(x_ref).flatten()
        

        # Compute the optimal control sequence
        u_star = tracker.computesolution(x_ref,x0_mpc,u_mpc, H, Ftra,initial_guess=u_star)
        # Return the optimal control sequence
        u_mpc += u_star[:num_joints]
       
        # Control command
        tau_cmd  = gravity_cancellation(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # print(cmd.tau_cmd)
        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        
        q_d, qd_d = ref.get_values(current_time)


        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        u_mpc_all.append(u_mpc.copy())
        time_all.append(current_time)
        q_d_all.append(q_d)
        qd_d_all.append(qd_d)

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print(f"Time: {current_time}")
    
    q_mes_all = np.array(q_mes_all)
    qd_mes_all = np.array(qd_mes_all)
    u_mpc_all = np.array(u_mpc_all)
    time_all = np.array(time_all)
    q_d_all = np.array(q_d_all)
    qd_d_all = np.array(qd_d_all)
    
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 12))
        
        # Position plot for joint i
        plt.subplot(3, 1, 1)
        plt.plot(q_mes_all[:, i], label=f'Measured Position - Joint {i+1}')
        plt.plot(q_d_all[:, i], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Position [rad]')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(3, 1, 2)
        plt.plot(qd_mes_all[:, i], label=f'Measured Velocity - Joint {i+1}')
        plt.plot(qd_d_all[:, i], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Velocity [rad/s]')
        plt.legend()

        # Control effort plot for joint i
        plt.subplot(3, 1, 3)
        plt.plot(u_mpc_all[:, i], label=f'Control Effort - Joint {i+1}')
        plt.title(f'Control Effort for Joint {i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Control Input')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
     
    
    
if __name__ == '__main__':
    
    main()