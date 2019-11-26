import numpy as np
def func1(input_obj):
	x = input_obj[0]
	return x**2
def numerical_derivative(f, x) :
	delta_x = 1e-4
	return (f(x+delta_x) - f(x-delta_x)) / (2*delta_x)
print(numerical_derivative(func1, np.array([3,0])))