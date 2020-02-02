During my senior year of high school, I collaborated with two other students in a year-long class in which we managed our own project programming many iterations of neural networks, culminating in a neural network that could distinguish between pictures of airplanes and pictures of people with a high rate of success. We coded the first networks from scratch in Python to deepen our understanding of how a neural network actually works, without specialized neural network libraries. Then, we used Keras with a backend of Tensorflow to code subsequent networks.

#Repository Structure

The "Network from Scratch" folder contains a perceptron (single-layer neural network) and its dataset. This perceptron, run by the file Perceptron.py, was the culmination of what we had learned from our initial research and our work coding networks from scratch. When run, it analyzes a dataset of sonar bouncing off of an object to determine if the object was a rock or a metal rod. Mainly NT2019 and I worked on this part of the project, which (including the research and previous iterations that led up to it) spanned over half of our classtime.

The "Final Keras Network" folder contains our final leg of the project, which analyzes 10x10 grayscale images (resized with PhotoScape) and determines if they are a plane or a person. The training data is located in the "Faces and PLanes" folder, and the program that creates the neural network model is "Planes and People Network.py", which produces files model.yaml and model.h5; picturerunner.py uses these to files to test the trained model on a new dataset. Since we were using this test data in our demo, it requests the specific filename in order to run. You can check to see if you are a person or an airplane by adding your own 10x10 grayscale image and running picturerunner.py. Mainly Stroganogg and I worked on this part of the project, which took the remainder of the time in the class.

#Overview of How Neural Network Works

On a basic level, neural networks that use supervised learning work by feeding data through a sequence of algorithms, grouped into nodes called neurons that are organized into patterns called layers. During this stage of training the network, the network’s predicted result comes out through the output layer and is compared with the theoretical (already known) result. Based on whether or not it is correct, the program changes the algorithms in the neurons to improve its accuracy for future guesses. 

The process of building a neural network can be summarized in three steps:
1. Set up the network, including the structure of the neurons, size of the layers, and initial algorithms used.
2. Train the network, meaning feed the data through the network in order to build up its accuracy.
3. Export the trained network and test its predictions on other data sets.
