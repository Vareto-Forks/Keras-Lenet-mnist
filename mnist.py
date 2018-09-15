from lenet import LeNet
from sklearn.cross_validation import train_test_split
from keras.datasets import mnist as keras_mnist
from keras.optimizers import SGD
from keras.utils import np_utils as keras_np_utils
import numpy as np
import cv2

# define some parameters
img_rows = 28
img_cols = 28
num_classes = 10
weightsPath = None#"weights/lenet_weights.hdf5"

print("downloading MNIST...")
# load/download mnist dataset
(trainX, trainY), (testX, testY) = keras_mnist.load_data()

# adapting images to tensorflow backend (channels_last)
x_train = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
x_test = testX.reshape(testX.shape[0], img_rows, img_cols, 1)

# convert target vectors to binary class matrices
y_train = keras_np_utils.to_categorical(trainY, num_classes)
y_test = keras_np_utils.to_categorical(testY, num_classes)


print(x_train[0].shape)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=img_cols, height=img_rows, depth=1, classes=num_classes, weightsPath=weightsPath)
model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# if no weights specified train the model
if weightsPath is None:
	print("[INFO] training...")
	model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

	print("[INFO] dumping weights to file...")
	model.save_weights(weightsPath, overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	# classify the digit
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)

	# resize the image from a 28 x 28 to 96 x 96
	image = (testData[i][0] * 255).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# show the image and prediction
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
