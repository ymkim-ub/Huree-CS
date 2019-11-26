import numpy as np
import random
loaded_data = np.loadtxt("data02.csv", delimiter = ",")
x_data = loaded_data[ : , 0:-1]
t_data = loaded_data[ : , [-1]]

W = np.random.rand(3, 1)
b = np.random.rand(1)
print("W= ", W, "W.shape= ", W.shape, ", b= ", b, ", b.shape= ", b.shape)
def loss_func(x, t):
	y = np.dot (x, W) + b
	return (np.sum( (t-y)**2 )/(len(x)) )
def numerical_derivative(f, x) :
	delta_x = 1e-4
	grad = np.zeros_like(x)
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + delta_x
		fx1 = f(x)		# f(x+delta_x)
		x[idx] = tmp_val - delta_x
		fx2 = f(x)		# f(x-delta_x)
		grad[idx] = (fx1 - fx2) / (2*delta_x)
		x[idx] = tmp_val
		it.iternext()
	return grad
def error_val(x, t):
	y = np.dot(x, W) + b
	return (np.sum ( (t - y) **2 )  / (len(x)) )
def predict(x):
	y = np.dot(x, W) + b
	return y
learning_rate = 1e-5
f = lambda x : loss_func (x_data, t_data)
print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", W, "b = ", b)
for step in range(10001):
	W -= learning_rate * numerical_derivative (f, W)
	b -= learning_rate * numerical_derivative (f, b)
	if (step % 400 == 0):
		print("step = ", step, "error value = ", error_val(x_data, t_data), "W= ", W, "b= ", b)
test_data = np.array([100, 98, 81])
result = predict(test_data)
print("predict data is ", result)