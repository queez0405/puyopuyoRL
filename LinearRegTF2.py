import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
#from RegExample2 import LinearRegression

class LinearRegression:
	def __init__(self, x_data, y_data):
		self.x_data = x_data
		self.y_data = y_data
		self.optimizer = tf.keras.optimizers.Adam(0.1)
		self.W = tf.Variable(np.random.randn(), name="weight")
		#self.b = tf.Variable(np.random.randn(), name="bias")
		self.n_samples = self.x_data.shape[0]

	def linear_regression(self, x):
		#return self.W * x + self.b
		return self.W * x

	def mean_square(self, y_pred, y_true):
		return tf.reduce_sum(tf.square(y_pred-y_true))/(2 * self.n_samples)

	def run_optimization(self):
		with tf.GradientTape() as g:
			pred = self.linear_regression(self.x_data)
			loss = self.mean_square(pred, self.y_data)
		#gradients = g.gradient(loss, [self.W, self.b])
		gradients = g.gradient(loss, [self.W])
		#self.optimizer.apply_gradients(zip(gradients, [self.W, self.b]))
		self.optimizer.apply_gradients(zip(gradients, [self.W]))
	def train(self, updates=1000):
		for step in range(1, updates):
			self.run_optimization()
			if step % 1000 == 0:
				#print("Updates :",step,"W :",self.W.numpy(),"b :",self.b.numpy())
				print("Updates :",step,"W :",self.W.numpy())


		return self.W
