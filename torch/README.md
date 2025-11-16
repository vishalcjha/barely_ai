## Deeplearning

At its core, deeplearning is a mathematical function that maps input to output. The reason for its popularity is its ability to solve, or better say approximate, solutions without requiring us to come up with an explanatory model for a given input and output. This makes these models universal, which sacrifices correctness for simplicity.

### Who should read this

I will walk through an example of a model based on [pytorch](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). I cannot do justice to explaining it, as I am also learning. But one can use this tutorial to get a sense of deep learning capabilities. If you are ready to black-box pieces, you should read this.

### Pytorch

A Python library providing building blocks for machine learning and its subset, deep learning. Some key concepts are:
* Tensor - a multidimensional data structure. This can also be thought of as a
multidimensional matrix.
* Model - a mathematical function with parameters.
* Prediction / inference - using a trained model to predict output.
* Parameters - weights and biases that a model learns. During model training these are updated and thus also called learnable parameters.
* Gradient - a vector containing partial derivatives of a function with respect to its parameters.
* Loss function - a function that calculates the distance between prediction and reality. Examples include Root Mean Square Error.
* Epochs - iterations in model training that update parameters to minimize the loss function.
* Learning rate - change in parameter value with each epoch. A small learning rate makes learning slow, but a high learning rate changes parameters drastically, making it impossible to find the best values.
*  Normalization - tensors are usually expressed in floating values. Each epoch involves lots of matrix multiplication, which can cause values to grow very large. To minimize this, tensors are normalized so that they sum to 1.
  

### Setting environment
#### Option 1 - [Use local jypyter notebook](discover_pi_jupyter.ipynb)
* Make sure to have python. I am using python `3.11`. I had trouble with latest version, but could have been just cache issues.
* I am running notebook in python virtual environment. 
  * `python3.11 -m venv .venv` to create virtual environment.
  * `source .venv/bin/activate` to activate virtual environment.
  * Follow libraries are installed in virtual environment using `pip install`
    * torch
    * jupyter
    * matplotlib
  
#### Option 2 - [Use Google Colab](./discover_pi.ipynb) 

## Discover PI
This example is highly motivated by finding relationship between celcius and fahrenheit example in book [**Deeplearning with Pytorch**](https://www.manning.com/books/deep-learning-with-pytorch).

Suppose we have a set of radii and corresponding areas. PI has not been invented yet---can we use deep learning to calculate PI? We will start by picking different mathematical functions with different degrees and iterate through our values of radii. In the beginning, we will give `PI a value of 1` and then update this by comparing predicted values to actual values. Remember, PI is our learning parameter, which is updated during each epoch. Once we have a loss value under an acceptable range, we can stop.

### Two phases of learning

- Forward - during this phase the model moves forward from one layer to another applying different mathematical functions. In our example we have just one layer. Here calculation flows from input toward prediction. In our example this is the `t_p = model(ip_radii, *params)` part of the training loop.
- Backward - learning parameters are updated using gradients. Here calculation flows from prediction toward input. This approach allows the use of the `chain rule` to find derivatives. Pytorch tensors with `(requires_grad = True)` do this backward propagation. In our example note `params = torch.tensor([1., 0.], requires_grad = True)`. Also, backward calculation is done by `loss.backward()`.

### Liberties / Simplification
-  Though we do not have a defined value for PI, we will be using it to make our sample data.
-  In actual learning processes, data is divided into 3 parts:  training, validation, and test. In our example we will train on the complete dataset.

Now in deeplearning, we do not have to come up with a model for a given problem set; we need to try different mathematical functions.

### First try

We will assume radius and area are related to each other using a factor of PI. So our model or function to calculate area will be:
```
    def area(radius):
        return PI * radius + bias
```
When we run our training loop for 1000 iterations (completely random), we find the loss value is not changing. This indicates our model cannot **fit** the relationship between input and output. As we are using a linear model, which is simple, it is not able to capture the relationship.

### Second try

How about making our model a bit more complex and hope it finds the relationship better?
```
    def area(radius):
        return PI * (radius ** 2) + bias  # making model complex by moving from linear to quadratic equation.
```
Now when we run this learning loop, we discover the loss value drops near zero. Though in this case 1000 was a high number of epochs, our returned parameter is `tensor([3.142, 0.000])`. The first part of the parameter is `3.141589879989624`, which is quite accurate.

### Conclusion

Deeplearning relies on training on a large dataset. One begins with a random estimate of parameters that get updated during each epoch. In the real world, one does not try different methods as we did in our example. There are sets of mathematical learning functions that capture the essence of different phenomena. The hard part is having data and the
ability to run training loops. And for sure, I am simplifying this to a very basic level, but my hope is to make it easy to understand for myself and others.

I hightly encourage you to run at [Google Colab](https://colab.research.google.com/drive/1UGpOLQbDSQmSrpxlIm9aZCeO3Kl6QnDv) OR [Jupyter notebook locally](./discover_pi_jupyter.ipynb). It takes less than a minute of your time.
