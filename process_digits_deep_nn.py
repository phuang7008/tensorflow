import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
features = 28 * 28

num_nodes_hidden_layers = [features, 600, 1000]  # you could add more nodes and hidden layers

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, features])
y = tf.placeholder('float')

def neural_network_model(data):
	# define weights and biases for each layer. The matrix size will be re-defined by the width
	height = features
	width = 0
	nodes = []
	for node in range(len(num_nodes_hidden_layers)):
		width = num_nodes_hidden_layers[node]
		weight = tf.Variable(tf.random_normal([height, width]))
		biases = tf.Variable(tf.random_normal([width]))
		nodes.append({'weights':weight, 'biases':biases})
		height = width

	weight = tf.Variable(tf.random_normal([height, n_classes]))
	biases = tf.Variable(tf.random_normal([n_classes]))
	output_layer = {'weights':weight, 'biases':biases}
	#nodes.append(output_layer)

	# (input_data * weights) + biases
	input_data = data
	for node in nodes:
		layer = tf.add(tf.matmul(input_data, node['weights']), node['biases'])
		input_data = tf.nn.relu(layer)   # this is the activitation function 'rectified linear'

	output = tf.matmul(input_data, output_layer['weights']) + output_layer['biases']
	return output

def train_neural_network(data, epoches=10):
	prediction = neural_network_model(data)

	# now define the backproppgation (backprop using optimization)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# now setup the session and run the training model => epoch = feed forward + backprop
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(epoches):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				_, ct = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
				epoch_loss += ct

			print("For epoch:", epoch, "in total epoches: ", epoches, "total loss is: ", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy is: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x, 10)
