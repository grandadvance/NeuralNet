# NeuralNet
Studying the NN creation process: basic tasks solving and my own ideas realization.

**Goals:**

A. Complete chapter one of Nielsen book http://neuralnetworksanddeeplearning.com:
1. Describe mathematically the NeuralNet architecture on the iDroo board;
2. Compare notations between Nielsen's first chapter and architecture on IDROO, repeated by me;
3. Study the code from Nielsen's first chapter;
4. Afterward create code from scratch.


Let us assume that the structure of our neural network is represented as Python list as follows: 

nn_structure = [784, 30, 10] 

We also have training data formatted like: 

training_data = [ ( X_1, Y_1 ), ( X_2, Y_2 ), ... , ( X_50000, Y_50000 ) ]

![Nielsen Neural Net structure](images/Nielsen_NN_structure.png)

To begin with let us algebraically describe input, hidden and output layers.

When we pass from the input layer to the hidden layer, we need to multiply each input value by the appropriate weight, as shown in the picture above. Let z_i^2 denote the resulting weighted sums for the second layer. Thus, we obtain the following system of equations:

$$
\text{System 1:} \quad
\begin{cases}
z_1^2 = w_{11}^2 a_1^1 + w_{12}^2 a_2^1 + \dots + w_{1\,784}^2 a_{784}^1 \\
z_2^2 = w_{21}^2 a_1^1 + w_{22}^2 a_2^1 + \dots + w_{2\,784}^2 a_{784}^1 \\
\dots \\
z_{30}^2 = w_{30\,1}^2 a_1^1 + w_{30\,2}^2 a_2^1 + \dots + w_{30\,784}^2 a_{784}^1
\end{cases}
$$

Now, these weighted sums must serve as the input data for the next, third layer. However, they cannot simply be fed into the next layer, as they can become extremely large. To prevent this, we pass each weighted sum through the sigmoid function to map all \(z_{i}^{2}\) values into numbers between 0 and 1. We can then rewrite the system of equations as follows (denote each results by a_i^2):

$$
\text{System 2:} \quad
\begin{cases}
a_1^2 = \sigma(z_1^2) \\
a_2^2 = \sigma(z_2^2) \\
\dots \\
a_{30}^2 = \sigma(z_{30}^2)
\end{cases}
$$

Now, each \(a_{i}^{2}\) serves as the input data for the next, third layer. Following this logic, we multiply each resulting value \(a_{i}^{2}\) by the corresponding weight between the second and third layers to obtain an analogous system of equations:

$$
\text{System 3:} \quad
\begin{cases}
z_1^3 = w_{11}^3 a_1^2 + w_{12}^3 a_2^2 + \dots + w_{1\,30}^3 a_{30}^2 \\
z_2^3 = w_{21}^3 a_1^2 + w_{22}^3 a_2^2 + \dots + w_{2\,30}^3 a_{30}^2 \\
\dots \\
z_{10}^3 = w_{10\,1}^3 a_1^2 + w_{10\,2}^3 a_2^2 + \dots + w_{10\,30}^3 a_{30}^2
\end{cases}
$$

and

$$
\text{System 4:} \quad
\begin{cases}
a_1^3 = \sigma(z_1^3) \\
a_2^3 = \sigma(z_2^3) \\
\dots \\
a_{10}^3 = \sigma(z_{10}^3)
\end{cases}
$$

Our goal is to implement the gradient descent process to find the correct weights and biases according to the following formulas:

$$
\text{System 5:} \quad
\begin{cases}
w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla C(w) \\
b_{\text{new}} = b_{\text{old}} - \eta \cdot \nabla C(b)
\end{cases}
$$

Naturally, each variable such as $w$, $b$ or $\nabla C$ is a vector of values. So, let's derive them. During the neural network training, an error $C_{m}$ is generated at each of its outputs $a_{m}$. We need to find the MSE (Mean Squared Error) for all such errors. To do this, we take the derivatives as follows:

$$
\text{Equation 1:} \quad
MSE = C(a_m) = \sum_{m=1}^{10} C_m = 0.5 \sum_{m=1}^{10} (a_m^3 - y_m)^2
$$

In system No. 5, the nabla operator represents the gradient—the vector of partial derivatives of our MSE. Thus:

$$
\text{Equation 2:} \quad
\nabla C(a_m) = \sum_{m=1}^{10} \frac{dC}{da_m^3} = \sum_{m=1}^{10} (a_m^3)' (a_m^3 - y_m)
$$ 

However, our goal is not to find how the gradient depends on the output layer activations $a_{m}$. Our task is to determine the dependence of $\nabla C$ on the weights and biases of the hidden (second) and final (third) layers:

$$
\text{System 6:} \quad
\begin{cases}
\nabla C(w_l) = ? \\
\\
\nabla C(b_l) = ?
\end{cases}, where \text{ } l \text{ } is \text{ } the \text{ } index \text{ } of \text{ } corresponding \text{ } layers
$$

Let us rewrite the equation 2 as follows:

$$
\text{Equation 3:} \quad
\nabla C(z_m^3) = \sum_{m=1}^{10} \frac{dC}{dz_m^3} = \sum_{m=1}^{10} \sigma'(z_m^3) (\sigma(z_m^3) - y_m)
$$
Le us $$ z^3 $$ transform to vector like $$ {z^3_1, z^3_2, … ,z^3_{10}} $$, and $$y$$ transform to a vector like $$ {y_1, y_2, …, y_{10}} $$, The equation 3 becomes as follows ∶
$$
\nabla C(z^3) = \sigma'(z^3) \dot (\sigma(z^3) - y)
$$







