import tensorflow as tf 
import numpy as np 
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

hm_epoches = 5
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128    # you could set it to any values such as 258, 512 etc

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(data):
	# define weights and biases for each layer. The matrix size will be re-defined by the width
	weight = tf.Variable(tf.random_normal([rnn_size, n_classes]))
	biases = tf.Variable(tf.random_normal([n_classes]))
	layer = {'weights':weight, 'biases':biases}

	data = tf.transpose(data, [1,0,2])
	data = tf.reshape(data, [-1, chunk_size])
	data = tf.split(0, n_chunks, data)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
	outputs, states = rnn.rnn(lstm_cell, data, dtype=tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
	return output

def train_neural_network(data):
	prediction = recurrent_neural_network(data)

	# now define the backproppgation (backprop using optimization)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# now setup the session and run the training model => epoch = feed forward + backprop
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epoches):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))
				_, ct = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
				epoch_loss += ct

			print("For epoch:", epoch, "in total epoches: ", hm_epoches, "total loss is: ", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy is: ", accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
