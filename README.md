<br>
<div align="center">
  <h2>A supplement to Michael Nielsen's "Neural Networks and Deep Learning"</h2>
  <h3><i>A detailed algebraic derivation of the BP1, BP2, BP3 and BP4 equations</i></h3>
</div>
<br>

Let us assume that the structure of our neural network is represented as Python list as follows: 

nn_structure = [784, 30, 10] 

We also have training data formatted like: 

training_data = $[ ( X_1, Y_1 ), ( X_2, Y_2 ), ... , ( X_{50 000}, Y_{50 000} ) ]$

![Nielsen Neural Net structure](images/Nielsen_NN_structure.png)
<p align="center"><small>Figure 1</small></p> <br>

To begin with let us algebraically describe input, hidden and output layers.

When we pass from the input layer to the hidden layer, we need to multiply each input value by the appropriate weight, as shown in the picture above. Let z_i^2 denote the resulting weighted sums for the second layer. Thus, we obtain the following system of equations:

$$
\begin{cases}
z_1^2 = w_{11}^2 a_1^1 + w_{12}^2 a_2^1 + \dots + w_{1\,784}^2 a_{784}^1 \\
z_2^2 = w_{21}^2 a_1^1 + w_{22}^2 a_2^1 + \dots + w_{2\,784}^2 a_{784}^1 \\
\dots \\
z_{30}^2 = w_{30\,1}^2 a_1^1 + w_{30\,2}^2 a_2^1 + \dots + w_{30\,784}^2 a_{784}^1
\end{cases}
$$
<p align="center"><small>System 1</small></p> <br>

Now, these weighted sums must serve as the input data for the next, third layer. However, they cannot simply be fed into the next layer, as they can become extremely large. To prevent this, we pass each weighted sum through the sigmoid function to map all \(z_{i}^{2}\) values into numbers between 0 and 1. We can then rewrite the system of equations as follows (denote each results by a_i^2):

$$
\begin{cases}
a_1^2 = \sigma(z_1^2 + b_1^2) \\
a_2^2 = \sigma(z_2^2 + b_2^2) \\
\dots \\
a_{30}^2 = \sigma(z_{30}^2 + b_{30}^2)
\end{cases}
$$

<p align="center"><small>System 2</small></p> <br>

Now, each \(a_{i}^{2}\) serves as the input data for the next, third layer. Following this logic, we multiply each resulting value \(a_{i}^{2}\) by the corresponding weight between the second and third layers to obtain an analogous system of equations:

$$
\begin{cases}
z_1^3 = w_{11}^3 a_1^2 + w_{12}^3 a_2^2 + \dots + w_{1\,30}^3 a_{30}^2 \\
z_2^3 = w_{21}^3 a_1^2 + w_{22}^3 a_2^2 + \dots + w_{2\,30}^3 a_{30}^2 \\
\dots \\
z_{10}^3 = w_{10\,1}^3 a_1^2 + w_{10\,2}^3 a_2^2 + \dots + w_{10\,30}^3 a_{30}^2
\end{cases}
$$
<p align="center"><small>System 3</small></p> <br>

and

$$
\begin{cases}
a_1^3 = \sigma(z_1^3 + b_1^3) \\
a_2^3 = \sigma(z_2^3 + b_2^3) \\
\dots \\
a_{10}^3 = \sigma(z_{10}^3 + b_10^3)
\end{cases}
$$
<p align="center"><small>System 4</small></p> <br>

Our goal is to implement the gradient descent process to find the correct weights and biases according to the following formulas:

$$
\begin{cases}
w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla C(w) \\
b_{\text{new}} = b_{\text{old}} - \eta \cdot \nabla C(b)
\end{cases}
$$
<p align="center"><small>System 5</small></p> <br>

Naturally, each variable such as $w$, $b$ or $\nabla C$ is a vector of values. So, let's derive them. During the neural network training, an error $C_{m}$ is generated at each of its outputs $a_{m}$. We need to find the MSE (Mean Squared Error) for all such errors. To do this, we take the derivatives as follows:

$$
MSE = C(a_m) = \sum_{m=1}^{10} C_m = 0.5 \sum_{m=1}^{10} (a_m^3 - y_m)^2
$$
<p align="center"><small>Equation 1</small></p> <br>

In system No. 5, the nabla operator represents the gradient—the vector of partial derivatives of our MSE. Thus:

$$
\nabla C(a_m) = \sum_{m=1}^{10} \frac{dC}{da_m^3} = \sum_{m=1}^{10} (a_m^3)' (a_m^3 - y_m)
$$ 
<p align="center"><small>Equation 2</small></p> <br>

However, our goal is not to find how the gradient depends on the output layer activations $a_{m}$. Our task is to determine the dependence of $\nabla C$ on the weights and biases of the hidden (second) and final (third) layers:

$$
\begin{cases}
\nabla C(w_l) = ? \\
\\
\nabla C(b_l) = ?
\end{cases}, where \text{ } l \text{ } is \text{ } the \text{ } index \text{ } of \text{ } corresponding \text{ } layers
$$
<p align="center"><small>System 6</small></p> <br>

Let us rewrite the equation 2 as follows:

$$
\nabla C(z_m^3) = \sum_{m=1}^{10} \frac{dC}{dz_m^3} = \sum_{m=1}^{10} \sigma'(z_m^3) (\sigma(z_m^3) - y_m)
$$
<p align="center"><small>Equation 3</small></p> <br>

Let us $z^3$ transform to vector like ${z^3_1, z^3_2, … ,z^3_{10}}$, and y transform to a vector like ${y_1, y_2, …, y_{10}}$. The equation 3 becomes as follows ∶

$$
\delta^3=\nabla C(z^3) = (\sigma(z^3) - y) \odot \sigma'(z^3)
$$
<p align="center"><small>Equation 4</small></p> <br>

or

$$
\delta^3= (\sigma(z^3) - y) \odot \sigma(z^3)\dot(1-\sigma(z^3))
$$
<p align="center">
 ________________________<br>
  <b>Equation 4' - BP1</b><br>
 ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
</p>
<br>

In Michael Nielsen's book, equation 4 is known as BP1. This formula demonstrates how the weighted sum of weights and biases between the second and third layers affects the final error. By the way dot between $(\sigma(z^3) - y)$ and $\sigma(z^3)(1-\sigma(z^3))$ is scalar product!

However, in system 5, we do not need $C(z^3)$, but rather $C(w^3)$. Therefore, we need to find $\nabla C(w^3)$. But how do we access this variable $w$? We have come very close! Please refer to Figure 2. For simplicity, I have collapsed all the layers into conventional circles:

![dC/dx explanation](images/dC_dx_explanation.png)
<p align="center"><small>Figure 2</small></p> <br>

In the simplified diagram above, our final error $dC$ is affected by either the variable from path 1, the variable from path 2, the variable from path 3, and so on. Suppose we want to find $\partial C / \partial z^3_m$. Notice the method I proposed to do this: we simply added a unit fraction and then filled it with the required element!

Note that when converting algebraic transformations into Python code, you need to apply transposition where necessary. To make the transition from element-wise to vector transformation clearer for you, let us recall what the formula for $\delta ^{3}$ looked like

$$
\nabla C(z^3) = (\sigma(z^3) - y) \odot \sigma'(z^3)
$$

and do the following:

$$
\text{BP1} : \delta^3 = \nabla C(z^3) = (\sigma(z^3) - y) \odot \sigma(z^3) \odot (1 - \sigma(z^3))
$$

$$
\frac{dC}{da^2} = \frac{dC}{da^3} \cdot \frac{da^3}{dz^3} \cdot \frac{dz^3}{da^2}
$$

<p align="center"><small>Equation 5</small></p> <br>

However, for this vector-matrix product to be algebraically feasible, we must swap the order of the weight matrix and the error $\delta$

Furthermore, we cannot use standard multiplication symbols since our formulas deal with matrices and vectors. Therefore, when expanding the differentials, we will directly replace the multiplication signs with their corresponding alternatives in certain places.

$$
\frac{dC}{da^2} = \frac{dz^3}{da^2} \cdot \frac{dC}{da^3} \cdot \frac{da^3}{dz^3} = w \cdot (a^3 - y) \odot \sigma(z^3) \odot (1 - \sigma(z^3))
$$

<p align="center"><small>Equation 6</small></p> <br>

It is also important to note that we use the transposition operation to make the Hadamard product possible. This is an artificial introduction at this step, driven by dimension matching considerations during multiplication.

$$
\frac{dC}{da^2} = \begin{pmatrix}
w_{11}^3 & w_{12}^3 & w_{13}^3 & \dots & w_{1\ 30}^3 \\
w_{21}^3 & w_{22}^3 & w_{23}^3 & \dots & w_{2\ 30}^3 \\
\dots & \dots & \dots & \ddots & \dots \\
w_{10\ 1}^3 & w_{10\ 2}^3 & w_{10\ 3}^3 & \dots & w_{10\ 30}^3
\end{pmatrix}^T \cdot \left( \begin{pmatrix}
a_1^3 \\
a_2^3 \\
\dots \\
a_{10}^3
\end{pmatrix} - \begin{pmatrix}
y_1 \\
y_{10} \\
\dots \\
y_{10}
\end{pmatrix} \right) \odot \sigma \begin{pmatrix}
z_1^3 \\
z_2^3 \\
\dots \\
z_{10}^3
\end{pmatrix} \odot \left( 1 - \sigma \begin{pmatrix}
z_1^3 \\
z_2^3 \\
\dots \\
z_{10}^3
\end{pmatrix} \right)
$$
<p align="center"><small>Equation 7</small></p> <br>

$$
\frac{dC}{da^2} = \begin{pmatrix}
w_{11}^3 & w_{12}^3 & w_{13}^3 & \dots & w_{1\,30}^3 \\
w_{21}^3 & w_{22}^3 & w_{23}^3 & \dots & w_{2\,30}^3 \\
\dots & \dots & \dots & \ddots & \dots \\
w_{10\,1}^3 & w_{10\,2}^3 & w_{10\,3}^3 & \dots & w_{10\,30}^3
\end{pmatrix}^T \cdot \delta^3
$$

<p align="center">Equation 8 <code>dimension of δ³ is 10x1</code></p>
<br> 

Or shortly:

<p align="center"> $\frac{dC}{da^2} = (w^3)^T \cdot \delta^3$</p>
<br> 

<p align="center">Equation 8' </p>
<br> 

Next. What happens if we continue moving from right to left and try to find $\partial C / \partial z^2$? This is exactly the spot in Figure 2 where I put e.t.c.:

![da/dz^2 explanation](images/da_dz2_explanation.png)
<p align="center"><small>Figure 3</small></p> <br>

$$
\delta^2 = \frac{dC}{dz^2} = \left( (w^3)^T \cdot \delta^3 \right) \odot \sigma'(z^2)
$$

<p align="center">
 ________________________<br>
  <b>Equation 9 - BP2</b><br>
 ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
</p>
<br>

Please note: we enclosed the operation $(w^3)^T \cdot \delta^3$ in parentheses because the order of multiplication matters. We perform the operation inside the parentheses first since, according to the steps described above, it precedes the Hadamard product!

Now, all that remains is to derive the dependence of the change in error $dC$ on the bias increments (BP3) and the dependence of the change in error $dC$ on the weights (BP4). To do this, we simply need to move one step further from right to left and take the derivative with respect to either the bias $b$ or the weight $w$. Let me remind system 2 for you:

$$
\begin{cases}
a_1^2 = \sigma(z_1^2 + b_1^2) \\
a_2^2 = \sigma(z_2^2 + b_2^2) \\
\dots \\
a_{30}^2 = \sigma(z_{30}^2 + b_{30}^2)
\end{cases}
$$

Now, similarly to what we did in Figure 3, we will take the derivative with respect to $b^{2}$ instead of $z^{2}$:

![da/db^2 explanation](images/da_db2_explanation.png)
<p align="center"><small>Figure 4</small></p> <br>

$$
\frac{dC}{db^2} = \left( (w^3)^T \cdot \delta^3 \right) \odot \sigma'(z^2) = \delta^2
$$

<p align="center">
 ________________________<br>
  <b>Equation 10 - BP3</b><br>
 ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
</p>
<br>

Before that let us state the key questions: how do the weights and biases between the first and second layers, as well as between the second and third layers, affect the final error? To answer this, let us first clarify exactly how we will use these weights and biases in Python. In the Network() class, Michael Nielsen suggests storing the weights self.w and biases self.b as large matrices nested within lists, as follows:

$$
\text{self.w} = \left[\text{ }
\begin{pmatrix}
w_{11}^2 & w_{12}^2 & w_{13}^2 & w_{14}^2 & \dots & w_{1\,784}^2 \\
w_{21}^2 & w_{22}^2 & w_{23}^2 & w_{24}^2 & \dots & w_{2\,784}^2 \\
w_{31}^2 & w_{32}^2 & w_{33}^2 & w_{34}^2 & \dots & w_{3\,784}^2 \\
\dots & \dots & \dots & \dots & \ddots & \dots \\
w_{30\,1}^2 & w_{30\,2}^2 & w_{30\,3}^2 & w_{30\,4}^2 & \dots & w_{30\,784}^2
\end{pmatrix}
,\text{ }
\begin{pmatrix}
w_{11}^3 & w_{12}^3 & w_{13}^3 & \dots & w_{1\,30}^3 \\
w_{21}^3 & w_{22}^3 & w_{23}^3 & \dots & w_{2\,30}^3 \\
\dots & \dots & \dots & \ddots & \dots \\
w_{10\,1}^3 & w_{10\,2}^3 & w_{10\,3}^3 & \dots & w_{10\,30}^3
\end{pmatrix}
\text{ }\right]
$$
<p align="center"><small>Equation 5</small></p> <br>

$$
\begin{gathered}
\text{self.b} = \left[\text{ }
\begin{pmatrix}
b_1^2 \\
b_2^2 \\
b_3^2 \\
\dots \\
b_{30}^2
\end{pmatrix}
,\text{ }
\begin{pmatrix}
b_1^3 \\
b_2^3 \\
\dots \\
b_{10}^3
\end{pmatrix}
\text{ }\right]
\end{gathered}
$$
<p align="center"><small>Equation 6</small></p> <br>




