#PPO = 
#PPO is a model free learning algorithm
#PPO is a variation of policy gradient methods

import tensorflow as tf
import numpy as np
import math

total_episodes = 200
max_ep_duration = 1000

#Table 1 Net
def conv_net_1d(activation = 'relu', init = tf.keras.initializers.Orthogonal(), out_units = 6):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (5, 1), strides= (2), padding= 'valid', activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (3, 1), strides= (2), padding= 'valid', activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units= 256, activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Dense(units= 128, activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Dense(units= out_units, activation = 'linear', kernel_initializer = init))
    return model

#Table 2 Net
def conv_net_2d(activation = 'relu', init = tf.keras.initializers.Orthogonal(), out_units = 6):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (8, 8), strides= (4,4), padding= 'valid', activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (4, 4), strides= (2,2), padding= 'valid', activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (3, 3), strides= (1,1), padding= 'valid', activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units= 512, activation = activation, kernel_initializer = init))
    model.add(tf.keras.layers.Dense(units= out_units, activation = 'linear', kernel_initializer = init))
    return model

class ENV:
    def __init__(self):
        
        #Define spatial limits
        self.x_lim = [0, 10]
        self.y_lim = [0, 10]
        
        #Define number of waypoints with fixed distance between them
        self.n_wp = 100
        
        #Constants from paper
        self.D_ped = 0.85
        self.D_g = 0.4
        self.R_g = 10
        self.R_ped = 7
        self.R_so = 7
        self.R1_vel = 0.001
        self.R2_vel = 0.01
        self.t_reac = 0.8
        self.w_1 = 4.5
        self.w_2 = 5.5

    def __position_norm(self, pos1, pos2):
        return math.cdist(pos1, pos2)

    def __position_diff(self, cur_pos1, cur_pos2, pre_pos1, pre_pos2):
        return self.__position_norm(pre_pos1, pre_pos2) - self.__position_norm(cur_pos1, cur_pos2)
    
    def __detect_collision_with_object():
        

    def get_random_position(self):
        return [round(random.uniform(self.x_lim[0], self.x_lim[1]), 2), round(random.uniform(self.y_lim[0], self.y_lim[1]), 2)]

    def reset(self):
        
        
    def step(self):
        
        if(robot_position - goal_position <= 0.4):
            done = True
    

class AGENT:
    def __init__(self, policy_name = "A1-RD"):
        self.v_max = 0.8 #m/s
        self.w_max = 0.7 #??? no idea
        self.action_space = [[0,0], [0, -self.w_max], [0, self.w_max], [self.v_max, 0], [self.v_max, self.w_max/2], [self.v_max, -self.w_max/2]]
        
        if(policy_name == "A1-RD"):
            self.policy = self.a1_rd()
            
    def a1_rd(self, activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
        
        input_laser = tf.keras.Input(shape=(laser_input_shape,1))
        input_wp = tf.keras.Input(shape=(wp_input_shape,))
        
        h1 = tf.keras.layers.Conv1D(filters= 32, kernel_size = 5, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)(input_laser)
        h2 = tf.keras.layers.Conv1D(filters= 64, kernel_size = 3, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)(h1)
        h3 = tf.keras.layers.Flatten()(h2)
        h4 = tf.keras.layers.Dense(units= 256, activation = activation, kernel_initializer = init)(h3)
        
        combined = tf.keras.layers.Concatenate()([h4, input_wp])
        
        h5 = tf.keras.layers.Dense(units = 128, activation = activation, kernel_initializer = init)(combined)
        output = tf.keras.layers.Dense(units = out_units, activation = 'linear', kernel_initializer = init)(h5)
        
        model = tf.keras.Model(inputs = [input_laser, input_wp], outputs = output)

        return model
        
    def calculate_reward(self, robot_position, goal_position, robot_velocity, robot_ang_velocity):
        
        reward_g = 0
        r_wp1 = 0
        r_wp2 = 0
        r_so = 0
        r_ped = -self.R_ped
        
        if(self.__position_norm(robot_position, goal_position) < self.D_g):
            reward_g = self.R_g
            
        #TODO -> implement diff between waypoints (eq. 5 -> difference between distance of agent to next waypoint of the previous time step, and current distance)
        #Waypoints are groups of 2D coordinates (x,y)
        diff = self.__position_diff(robot_position, waypoint_position)
        if(diff > 0):
            r_wp1 = self.w_1*diff
        if(diff < 0):
            r_wp2 = self.w_2*diff
        reward_wp = r_wp1 + r_wp2
        
        #TODO -> implement collision detector, inside pedestrian circle detector, velocity zero for t_reac
        if(detect_collision_with_object):
            r_so = -self.R_so
        if(detect_in_pedestrian_circle or check_vel_zero):
            r_ped = 0
        reward_o = np.min((r_so, r_ped))
        
        if(robot_velocity == 0 and robot_ang_velocity == 0):
            reward_vel = -self.R1_vel
        elif(robot_velocity == 0):
            reward_vel = -self.R2_vel
        else:
            reward_vel = 0
        
        #Accumulate goal (g) reward, waypoint (wp) reward, obstacles (o) reward and velocity (vel) reward
        reward = reward_g + reward_wp + reward_o + reward_vel

        return reward

    def get_rewards(self,actions):
        # Initialize an empty list to store the rewards
        rewards = []

        # Loop through each action
        for action in actions:
            # Calculate the reward for the action
            reward = self.calculate_reward(action)

        # Add the reward to the list
        rewards.append(reward)

        # Convert the list of rewards to a tensor
        rewards = tf.convert_to_tensor(rewards)

        # Return the rewards
        return rewards

    def calculate_advantages(self, true_actions, predicted_actions):
        # Calculate the rewards for each action
        rewards = self.get_rewards(true_actions)

        # Calculate the baseline using the predicted actions
        baseline = tf.math.reduce_mean(predicted_actions, dim=1)

        # Calculate the advantages using the rewards and baseline
        advantages = rewards - baseline

        # Return the advantages
        return advantages


#PPO loss function
def ppo_loss(advantages, old_predictions, predictions):
    
    # Clip the predicted actions to ensure stability
    predictions = tf.clip_by_value(predictions, clip_value_min=1e-8, clip_value_max=1-1e-8)
    old_predictions = tf.clip_by_value(old_predictions, clip_value_min=1e-8, clip_value_max=1-1e-8)

    # Calculate the ratio of the new and old predictions
    ratio = predictions / old_predictions

    # Calculate the PPO loss using the ratio and advantages, and mean of the loss
    loss = tf.keras.ops.min(ratio * advantages, tf.clip_by_value(ratio, clip_value_min=1-0.2, clip_value_max=1+0.2) * advantages)
    loss = tf.math.reduce_mean(loss)

    return loss


# Train the policy network
for epoch in range(100):
    # Sample a batch of states and actions
    states, actions = sample_batch()

    # Forward pass through the policy network
    predictions = policy_network(states)

    # Calculate the advantages using the true and predicted actions
    advantages = calculate_advantages(actions, predictions)

    # Calculate the PPO loss using the advantages and predicted actions
    loss = ppo_loss(advantages, old_predictions, predictions)

    # Backward pass and update the weights



#Instantiate environment
env = ENV()

#Instantiate agent
agent = AGENT()

#Start of an episode
for ep in range(total_episodes):
    
    #Generate random start position
    start_pos = env.get_random_position()
    
    #Generate random goal position
    goal_pos = env.get_random_position()
    
    #Compute global plan (to go from start to goal) -> DONE BY SIMULATOR
    
    #Generate pedestrians -> DONE BY SIMULATOR
    
    
    #Downsample global plan to number of waypoints with fixed distance to each other -> DONE BY SOMETHING????
    #PLACEHOLDER FOR TESTING
    wp = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
    
    
    #Reset state (essentially get state information from the environment)
    state = env.reset()
    done = False
    
    #Start episodes
    for k in range(max_ep_duration):
        
        #Scan Laser data -> DONE BY SIMULATOR
        #PLACEHOLDER FOR TESTING
        laser_data = [[0.2,0.1],[3.1,8.3],[0.7,7.5]]
        
        #Create state (waypoints + laser data)
        raw_data = wp + laser_data
        state = raw_data
        
        #Choose action, based on current state and policy
        action = agent.policy(state)
        
        #Step in the environment, getting next state and reward
        new_state, reward, done, info = env.step(action)
    
        # End this episode when `done` is True
        if done:
            break
    
        #Update state variable
        state = new_state
    
    