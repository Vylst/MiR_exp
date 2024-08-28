import tensorflow as tf
import numpy as np
import math

total_episodes = 200
max_ep_duration = 1000

v_max = 0.8 #m/s
w_max = 0.7 #??? no idea
action_space = [[0,0], [0, -w_max], [0, w_max], [v_max, 0], [v_max, w_max/2], [v_max, -w_max/2]]

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
        
        self.actor = self.get_actor()
        self.critic = self.get_critic()
        
    def get_actor(self, activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
	
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
 
    def get_critic(self, activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
        
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
        
    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state))

        return [np.squeeze(sampled_actions.numpy())]
        
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


"""
Generates an episode by executing the current policy in the given env
"""
def generate_single_episode(env, agent):

    states = []
    actions = []
    rewards = []
    log_probs = []
    state = env.reset()
        
    for t in range(max_ep_duration):
        state = tf.expand_dims(tf.convert_to_tensor(state, float), 0)
        probs = agent.policy(state) # get each action choice probability with the current policy network
        action = np.random.choice(len(action_space), p = np.squeeze(probs.numpy())) # probablistic
        #action = np.argmax(probs.numpy()) # greedy
        
        # compute the log_prob to use this in parameter update
        log_prob = tf.math.log (tf.squeeze(probs)[action])
        
        # append values
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        
        # take a selected action
        state, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)

        if done:
            break
            
    return states, actions, rewards, log_probs


def evaluate_policy(env, policy_net):
    """
    Compute accumulative trajectory reward
    """
    states, actions, rewards, log_probs = generate_single_episode(env, policy_net)
    return np.sum(rewards)




def train_PPO(env, policy_net, policy_optimizer, value_net, value_optimizer, num_epochs, clip_val=0.2, gamma=0.99):
    
    # Generate an episode with the current policy network
    states, actions, rewards, log_probs = generate_single_episode(env, policy_net)
    T = len(states)
    
    # Create tensors
    states = np.vstack(states).astype(float)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device).view(-1,1)
    rewards = torch.FloatTensor(rewards).to(device).view(-1,1)
    log_probs = torch.FloatTensor(log_probs).to(device).view(-1,1)

    # Compute total discounted return at each time step
    Gs = []
    G = 0
    for t in range(T-1,-1,-1): # iterate in backward order to make the computation easier
        G = rewards[t] + gamma*G
        Gs.insert(0,G)
    Gs = torch.tensor(Gs).view(-1,1)
    
    # Compute the advantage
    state_vals = value_net(states).to(device)
    with torch.no_grad():
        A_k = Gs - state_vals
        
    for _ in range(num_epochs):
        V = value_net(states).to(device)
        
        # Calculate probability of each action under the updated policy
        probs = policy_net.forward(states).to(device)
                
        # compute the log_prob to use it in parameter update
        curr_log_probs = torch.log(torch.gather(probs, 1, actions)) # Use torch.gather(A,1,B) to select columns from A based on indices in B
        
        # Calculate ratios r(theta)
        ratios = torch.exp(curr_log_probs - log_probs)
        
        # Calculate two surrogate loss terms in cliped loss
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1-clip_val, 1+clip_val) * A_k
        
        # Calculate clipped loss value
        actor_loss = (-torch.min(surr1, surr2)).mean() # Need negative sign to run Gradient Ascent
        
        # Update policy network
        policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        policy_optimizer.step()
        
        # Update value net
        critic_loss = nn.MSELoss()(V, Gs)
        value_optimizer.zero_grad()
        critic_loss.backward()
        value_optimizer.step()
        
    return policy_net, value_net





















