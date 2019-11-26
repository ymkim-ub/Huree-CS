import numpy as np
def func1(input_obj):
	x = input_obj[0]
	return x**2
def numerical_derivative(f, x) :
	delta_x = 1e-4
	grad = np.zeros_like(x)
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + delta_x
		fx1 = f(x)	# f(x+delta_x)
		x[idx] = tmp_val - delta_x
		fx2 = f(x)	# f(x-delta_x)
		grad[idx] = (fx1 - fx2) / (2*delta_x)
		print("idx = ", idx, "grad[idx] =", grad[idx])
		print("grad = ", grad)
		x[idx] = tmp_val
		it.iternext()
	return grad
print(numerical_derivative(func1, np.array([3,0])))