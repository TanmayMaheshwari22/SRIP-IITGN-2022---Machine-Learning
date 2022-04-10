# SRIP-IITGN-2022-Machine-Learning

**Question-3-Implement two hidden layers neural network classifier from scratch in JAX [20 Marks]**

**Two hidden layers here means (input - hidden1 - hidden2 - output).
You must not use flax, optax, or any other library for this task.
Use MNIST dataset with 80:20 train:test split.
Manually optimize the number of neurons in hidden layers.
Use gradient descent from scratch to optimize your network. You should use the Pytree concept of JAX to do this elegantly.
Plot loss v/s iterations curve with matplotlib.
Evaluate the model on test data with various classification metrics and briefly discuss their implications.
**

I used MLP classifier on MNIST dataset to predict the output.
In the model i created four layers: input layer, 2 hidden layers and 1 output layer.
Initially I created a function for initialising weights and biases.
Then I created a function to predict the output.
I used relu function for hidden layers.
Third function involved making loading of MNIST data from PyTorch
Fourth Function was to create accuracy function,loss function , gradient descent function from scratch.
This function also involved creation of MLP model by calling above functions.
Eventually accuracy was calculated and no. of epochs were initialised as 5.

Accuracy of the training data=0.9607499837875366
Accuracy of the testing data=0.9536999464035034

At last a graph was plotted between loss function and no. of epochs using matplotlib library.

Citation:
@misc{Gordic2021GetStartedWithJAX,
  author = {Gordić, Aleksa},
  title = {Get started with JAX},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/get-started-with-JAX}},
}
