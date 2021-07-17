
__author__ = "Majd Jamal | majdj@kth.se "

import tensorflow.keras.applications as app
import tensorflow.keras.datasets as ds

def loadData():
	""" Load data and return it as numpy array.
	"""
	(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
	return x_train, y_train, x_test, y_test

def model(dim = (32, 32, 3), Nclasses = 10):
	""" Creates and compiles a deep learning model.
	:param dim: Dimension of input data
	:param Nclasses: Number of unqiue classes
	:return net: Built and compiled MobileNet
	"""
	net = app.MobileNet(include_top=True,
						    weights=None,
						    input_shape=dim,
						    classes= Nclasses,
						    classifier_activation="softmax",
							)
	net.compile(
	    optimizer="adam",
	    loss="sparse_categorical_crossentropy",
	    metrics=["accuracy"],
	)

	return net

def train(model, X_train, y_train):
	""" Trains the model.
	:param model: Tensorflow network
	:param X_train: Training patterns, shape = (Npts, 32,32,3)
	:param y_train: Training labels, shape = (Npts, 1)
	:return model: trained network
	"""
	model.fit(X_train, y_train, epochs=20,
                    validation_split = 0.3)

	return model

def evaluate(model, X_test, y_test):
	"""	Evaluates the model
	:param model: Tensorflow network
	:param X_test: Test patterns, shape = (Npts, 32,32,3)
	:param y_test: Test labels, shape = (Npts, 1)
	:return test_acc: Accuracy on testing set
	"""
	test_loss, test_acc = model.evaluate(X_test,  y_test)
	return test_acc


def main():

	X_train, y_train, X_test, y_test = loadData()

	net = model()
	net = train(net, X_train, y_train)
	acc = evaluate(net, X_test, y_test)

	print('\n Accuracy is: ', str(round(acc * 100,2)) + '%')


main()
