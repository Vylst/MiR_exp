'''
Very important: https://medium.com/@sthanikamsanthosh1994/reinforcement-learning-part-8-proximal-policy-optimization-ppo-for-trading-9f1c3431f27d#:~:text=PPO%20is%20a%20type%20of,actions%20taken%20by%20the%20actor.
'''

import tensorflow as tf
import numpy as np
import math
import random

total_episodes = 10
max_ep_duration = 1000

'''
On-policy methods (those who learn solely from experience extracted from employing their policy, nad not from external or previously extracted data)
expect new samples at each policy update, so they require a lot of samples to converge. It complex environments this is difficult, so having a replay buffer is useful
'''
class Buffer:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        
class ActorNetwork(tf.keras.Model):
    def __init__(self, activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
        super(ActorNetwork, self).__init__()

        self.Conv1D1 = tf.keras.layers.Conv1D(filters= 32, kernel_size = 5, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)
        self.Conv1D2 = tf.keras.layers.Conv1D(filters= 64, kernel_size = 3, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units= 256, activation = activation, kernel_initializer = init)

        self.concat = tf.keras.layers.Concatenate()
        
        self.fc2 = tf.keras.layers.Dense(units = 128, activation = activation, kernel_initializer = init)
        self.fc3 = tf.keras.layers.Dense(units = out_units, activation = 'linear', kernel_initializer = init)

    def call(self, state):
        input_laser, input_wp = state
        x = self.Conv1D1(input_laser)
        x = self.Conv1D2(x)
        x = self.flatten(x)
        x = self.fc1(x)

        x = self.concat([x, input_wp])
        
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CriticNetwork(tf.keras.Model):
    def __init__(self, activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
        super(CriticNetwork, self).__init__()
        
        self.Conv1D1 = tf.keras.layers.Conv1D(filters= 32, kernel_size = 5, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)
        self.Conv1D2 = tf.keras.layers.Conv1D(filters= 64, kernel_size = 3, strides = 2, padding = 'valid', activation = activation, kernel_initializer = init)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units= 256, activation = activation, kernel_initializer = init)

        self.concat = tf.keras.layers.Concatenate()
        
        self.fc2 = tf.keras.layers.Dense(units = 128, activation = activation, kernel_initializer = init)
        self.fc3 = tf.keras.layers.Dense(units = out_units, activation = 'linear', kernel_initializer = init)

    def call(self, state):
        input_laser, input_wp = state
        x = self.Conv1D1(input_laser)
        x = self.Conv1D2(x)
        x = self.flatten(x)
        x = self.fc1(x)

        x = self.concat([x, input_wp])
        
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
class Agent:
    def __init__(self, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10, chkpt_dir='./'):
        
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
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.v_max = 0.8 #m/s
        self.w_max = 0.7 #??? no idea
        self.action_space = [[0,0], [0, -self.w_max], [0, self.w_max], [self.v_max, 0], [self.v_max, self.w_max/2], [self.v_max, -self.w_max/2]]

        self.actor = ActorNetwork(out_units = len(self.action_space))
        self.actor.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = alpha))
        self.critic = CriticNetwork(out_units = len(self.action_space))
        self.critic.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = alpha))
        self.memory = Buffer(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = tf.keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = tf.keras.models.load_model(self.chkpt_dir + 'critic')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        state = tf.expand_dims(state, 0)
     
        probs = self.actor(state)
        action = np.random.choice(len(probs.numpy()), p = np.squeeze(probs.numpy())) # probablistic
        #action = np.argmax(probs.numpy()) # greedy
        log_prob = tf.math.log(tf.squeeze(probs)[action])
        value = self.critic(state)


        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.actor(states)
                    new_probs = tf.math.log(tf.squeeze(probs))
                    new_probs = new_probs.numpy()[[[actions]]]

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(returns-critic_value, 2))
                    critic_loss = tf.keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        self.memory.clear_memory()
        
    # Calculates the norm between two points in 2D space
    def __position_norm(self, pos1, pos2):
        return math.dist(pos1, pos2)

    # Implements the diff function, described in the paper
    def __position_diff(self, cur_pos1, cur_pos2, pre_pos1, pre_pos2):
        return self.__position_norm(pre_pos1, pre_pos2) - self.__position_norm(cur_pos1, cur_pos2)
    
    # Assuming object collision happens when agent comes under 0.2me of object position -> DONE BY SENSOR?????????????
    def __detect_collision_with_object(self):
        if(True):
            return False
        else:
            return True
        
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
        if(self.__detect_collision_with_object()):
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
    
class ENV:
    def __init__(self):
        
        #Define spatial limits of environment
        self.x_lim = [0, 10]
        self.y_lim = [0, 10]
        
        #Define number of waypoints with fixed distance between them
        self.n_wp = 100
        
        self.robot_position = [0,0]
        self.goal_position = [0,0]
        
    # Provides coordinates of a random point in 2D space, bound to x_lim and y_lim
    def get_random_position(self):
        return [round(random.uniform(self.x_lim[0], self.x_lim[1]), 2), round(random.uniform(self.y_lim[0], self.y_lim[1]), 2)]

    def reset(self):
        self.robot_position = self.get_random_position()
        self.goal_position = self.get_random_position()
        
    def step(self):
        
        
        
        done = False
        if(self._position_norm(self.robot_position - self.goal_position) <= 0.4):
            done = True
            
    
    
    
if __name__ == '__main__':
    
    env = ENV()
    agent = Agent(gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=20)
    learn_iters = 0
    
    #Start of an episode
    for ep in range(total_episodes):
        
        #Generate random start position
        start_pos = env.get_random_position()
        
        #Generate random goal position
        goal_pos = env.get_random_position()
        
        #Compute global plan (to go from start to goal) -> DONE BY SIMULATOR!!!!!!!!!!!!!!!!!!
        
        #Generate pedestrians -> DONE BY SIMULATOR!!!!!!!!!!!!!!!!!!
        
        #Downsample global plan to number of waypoints with fixed distance to each other -> DONE BY SOMETHING?????????????
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
            raw_data = [laser_data, wp]
            
            #Choose action, based on current state and policy
            action = agent.choose_action(raw_data)
            
            #Step in the environment, getting next state and reward
            reward, done, info = env.step(action)
        
            # Every 20 steps, perform learning
            if k % 20 == 0:
                agent.learn()
                learn_iters += 1
        
            # End this episode when `done` is True
            if done:
                break
        
        
    
    
    
