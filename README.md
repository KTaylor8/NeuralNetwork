# Neural Network 

I worked with two other high school students in a year-long class in which we managed our own project in which we programmed many iterations of neural networks, culminating in a neural network that could distinguish between pictures of airplanes and pictures of people with a high rate of success. We coded the first networks from scratch in Python to deepen our understanding of how a neural network actually works, without specialized neural network libraries. Then, we used Keras with a backend of Tensorflow to code subsequent networks.

On a basic level, neural networks that use supervised learning work by feeding data through a sequence of algorithms, grouped into nodes called neurons that are organized into patterns called layers. During this stage of training the network, the network's predicted result comes out through the output layer and is compared with the theoretical (already known) result. Based on whether or not it is correct, the program changes the algorithms in the neurons to improve its accuracy for future guesses.

The process of building a neural network can be summarized in three steps:
1. Set up the network, including the structure of the neurons, size of the layers, and initial algorithms used.
2. Train the network, meaning feed the data through the network in order to build up its accuracy.
3. Export the trained network and test its predictions on other data sets.
