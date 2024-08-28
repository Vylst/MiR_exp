import tensorflow as tf
import numpy as np
import pickle
import math
import random

max_ep_duration = 1000
v_max = 0.8 #m/s
w_max = 0.7 #??? no idea
action_space = [[0,0], [0, -w_max], [0, w_max], [v_max, 0], [v_max, w_max/2], [v_max, -w_max/2]]
num_states = 13 #Size of State Space
num_actions = 2 #Size of Action Space

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

class Buffer:
	def __init__(self, buffer_capacity=100000, batch_size=64):
		# Number of "experiences" to store at max
		self.buffer_capacity = buffer_capacity
		# Num of tuples to train on.
		self.batch_size = batch_size

		# Its tells us num of times record() was called.
		self.buffer_counter = 0

		# Instead of list of tuples as the exp.replay concept go
		# We use different np.arrays for each tuple element
		self.state_buffer = np.zeros((self.buffer_capacity, num_states))
		self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

	# Takes (s,a,r,s') obervation tuple as input
	def record(self, obs_tuple):
		# Set index to zero if buffer_capacity is exceeded,
		# replacing old records
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.action_buffer[index] = obs_tuple[1]
		self.reward_buffer[index] = obs_tuple[2]
		self.next_state_buffer[index] = obs_tuple[3]

		self.buffer_counter += 1


	def update(self, state_batch, action_batch, reward_batch, next_state_batch):
		# Training and updating Actor & Critic networks.
		# See Pseudo Code.
		with tf.GradientTape() as tape:
		
			#Get actions that the actor would choose for next_state
			target_actions = target_actor(next_state_batch, training=True)
			#Calculate y based on reward obtained with actions chosen for state and 0.99 of what the critic thinks of the actions chosen for the next_state
			y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
			#Calculate what the critic thinks of the actions chosen for the state
			critic_value = critic_model([state_batch, action_batch], training=True)
			#Calculate critic loss as the difference between y and what the critic thinks for the actions chosen for state
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

		critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
		critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

		with tf.GradientTape() as tape:
			actions = actor_model(state_batch, training=True)
			critic_value = critic_model([state_batch, actions], training=True)
			# Used `-value` as we want to maximize the value given
			# by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value)

		actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
		actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

	# We compute the loss and update parameters
	def learn(self):
		# Get sampling range
		record_range = min(self.buffer_counter, self.buffer_capacity)
		# Randomly sample indices
		batch_indices = np.random.choice(record_range, self.batch_size)

		# Convert to tensors
		state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

		self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b * tau + a * (1 - tau))


def get_actor(activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
	
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
 
def get_critic(activation = 'relu', init = tf.keras.initializers.Orthogonal(), laser_input_shape = 32, wp_input_shape=100, out_units = 6):
	
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


"""
`policy()` returns an action sampled from our Actor network 
"""
def policy(state):
	sampled_actions = tf.squeeze(actor_model(state))
	sampled_actions = sampled_actions.numpy()

	return [np.squeeze(sampled_actions)]


"""
## Training hyperparameters
"""
actor_model = get_actor()
critic_model = get_critic()
target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models (0.001 is default for adam)
critic_lr = 0.001
actor_lr = 0.001
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
gamma = 0.99	# Discount factor for future rewards
tau = 0.005	# Used to update target networks

# Instantiate replay buffer
buffer = Buffer(50000, 64)

# Instantiate environment
env = ENV()

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


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
    
    #Reset reward accumulator
    episodic_reward = 0
    
    #Start episode
    while True:

        #Scan Laser data -> DONE BY SIMULATOR
        #PLACEHOLDER FOR TESTING
        laser_data = [[0.2,0.1],[3.1,8.3],[0.7,7.5]]
        
        #Create state (waypoints + laser data)
        raw_data = wp + laser_data
        state = raw_data
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        action = policy(tf_state)
		# Recieve state and reward from environment.
        reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)


#Store actor and critic models locally
actor_model.save('actor.h5')
critic_model.save('critic.h5')

#Store scores locally
with open('scores.pkl', 'wb') as fp1:
    pickle.dump([ep_reward_list, avg_reward_list], fp1)
fp1.close()


