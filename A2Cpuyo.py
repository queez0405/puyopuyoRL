import gym
from gym_puyopuyo import register
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import pdb
import time
from datetime import datetime
now = datetime.now()

from LinearRegTF2 import LinearRegression
load_weights = False
number_of_updates = 2000000
load_weights_update_num = 400000
checkpoint_name = 'puyoA2C_checkpoint'
log_path = './logs/A2C/' + now.strftime("%Y%m%d-%H%M%S")

class ProbabilityDistribution(tf.keras.Model):
	def call(self, logits):
	# sample a random categorical action from given logits
		return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
	def __init__(self, num_actions):
		super().__init__('mlp_policy')
		# no tf.get_variable(), just simple Keras API
		self.hidden1 = tf.keras.Sequential([kl.Dense(128, activation='relu'),kl.Dense(128, activation='relu')])
		self.hidden2 = tf.keras.Sequential([kl.Dense(128, activation='relu'),kl.Dense(128, activation='relu')])
		self.value = kl.Dense(1, name='value')
		# logits are unnormalized log probabilities
		self.logits = kl.Dense(num_actions, name='policy_logits')
		self.dist = ProbabilityDistribution()

	def call(self, inputs):
		# inputs is a numpy array, convert to Tensor
		x = tf.convert_to_tensor(inputs)
		# separate hidden layers from the same input tensor
		hidden_logs = self.hidden1(x)
		hidden_vals = self.hidden2(x)
		return self.logits(hidden_logs), self.value(hidden_vals)

	def action_value(self, obs):
		# executes call() under the hood
		logits, value = self.predict(obs)
		action = self.dist.predict(logits)
		# a simpler option, will become clear later why we don't use it
		# action = tf.random.categorical(logits, 1)
		return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class A2CAgent:
	def __init__(self, model):
# hyperparameters for loss terms, gamma is the discount coefficient
		self.params = {
			'gamma': 0.99,
			'value': 0.5,
			'entropy': 0.0001
		}
		self.model = model
		self.model.compile(
		optimizer=ko.Adam(lr=0.0007),
			# define separate losses for olicy logits and value estimate
			loss=[self._logits_loss, self._value_loss]
			)
    
	def train(self, env, batch_sz=32, updates=number_of_updates):
		avg_reward = tf.keras.metrics.Mean(name='reward', dtype = tf.float32)
		# storage helpers for a single batch of data
		actions = np.empty((batch_sz,), dtype=np.int32)
		rewards, dones, values = np.empty((3, batch_sz))
		observation_space1 = np.append(env.observation_space.sample()[0],env.observation_space.sample()[1])
		observations = np.empty((batch_sz,) + observation_space1.shape)
		# training loop: collect samples, send to optimizer, repeat updates times
		ep_rew = 0
		ep_count = 1
		next_obs = env.reset()
		# if saved weights exist, load weights
		if load_weights == True:
			model.load_weights('./checkpoints/' + checkpoint_name + str(load_weights_update_num))
		for update in range(updates):
			for step in range(batch_sz):
				if type(next_obs) is tuple:
					next_obs = self.obs_rank_down(next_obs)
				observations[step] = next_obs.copy()
				actions[step], values[step] = self.model.action_value(next_obs[None, :])
				next_obs, rewards[step], dones[step], _ = env.step(actions[step])
				if rewards[step] == -1:
					rewards[step] += 1
				rewards[step] = np.log(rewards[step] + 1)
				ep_rew += rewards[step]
				if dones[step]:
					avg_reward.update_state(ep_rew)
					if ep_count % 20 == 0:
						tf.summary.scalar('reward',avg_reward.result(),ep_count)
						avg_reward.reset_states()
					next_obs = env.reset()
					logging.info("Episode: %04d, Reward: %0.2f" % (ep_count, ep_rew))	
					ep_count += 1
					ep_rew = 0
			next_obs = self.obs_rank_down(next_obs)
			_, next_value = self.model.action_value(next_obs[None, :])
			returns, advs = self._returns_advantages(rewards, dones, values, next_value)
			# a trick to input actions and advantages through same API
			acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
			# performs a full training step on the collected batch
			# note: no need to mess around with gradients, Keras API handles it
			losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
			logging.debug("[%d/%d] Losses: %s" % (update+1, updates, losses))
			# save weights
			if (update + 1) % 1000 == 0:
				model.save_weights('./checkpoints/' + checkpoint_name + str(updates))
		return ep_rew

	def test(self, env, render=False):
		obs, done, ep_reward = env.reset(), False, 0
		while not done:
			obs = self.obs_rank_down(obs)
			action, _ = self.model.action_value(obs[None, :])
			obs, reward, done, _ = env.step(action)
			if reward == -1:
				reward += 1
			reward = np.log(reward + 1)
			ep_reward += reward
			if render:
				env.render()
				print("reward of this state is:",reward)
				time.sleep(0.2)
		return ep_reward

	def obs_rank_down(self, obs):
		obs = list(obs)
		obs[0] = np.ravel(obs[0], order='K')
		obs[1] = np.ravel(obs[1].astype('float64'), order='K')
		return np.append(obs[0], obs[1])

	def _returns_advantages(self, rewards, dones, values, next_value):
		# next_value is the bootstrap value estimate of a future state (the critic)
		returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
		# returns are calculated as discounted sum of future rewards
		for t in reversed(range(rewards.shape[0])):
			returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
		returns = returns[:-1]
		# advantages are returns - baseline, value estimates in our case
		advantages = returns - values
		return returns, advantages
    
	def _value_loss(self, returns, value):
		# value loss is typically MSE between value estimates and returns
		return self.params['value']*kls.mean_squared_error(returns, value)

	def _logits_loss(self, acts_and_advs, logits):
		# a trick to input actions and advantages through same API
		actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
		# sparse categorical CE loss obj that supports sample_weight arg on call()
		# from_logits argument ensures transformation into normalized probabilities
		weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
		# policy loss is defined by policy gradients, weighted by advantages
		# note: we only calculate the loss on the actions we've actually taken
		actions = tf.cast(actions, tf.int32)
		policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
		# entropy loss can be calculated via CE over itself
		entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
		# here signs are flipped because optimizer minimizes
		return policy_loss - self.params['entropy']*entropy_loss

if __name__ == '__main__':
	logging.getLogger().setLevel(logging.INFO)
	summary_writer = tf.summary.create_file_writer(log_path)
	'''
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
		except RuntimeError as e:
			print(e)
	'''
	register()
	env = gym.make('PuyoPuyoEndlessTsu-v2')
	model = Model(num_actions=env.action_space.n)
	agent = A2CAgent(model)
	with summary_writer.as_default():
		rewards_history = agent.train(env)
