import numpy as np
def sigmoid(x):
	return 1 / (1+np.exp(-x))
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
class LogicGate:
	def __init__(self, gate_name, xdata, tdata):
		self.name = gate_name
		self.__xdata = xdata.reshape(4,2)
		self.__tdata = tdata.reshape(4,1)
		self.__W = np.random.rand(2, 1)
		self.__b = np.random.rand(1)
		self.__learning_rate = 1e-2
	def __loss_func(self):
		delta = 1e-7
		z = np.dot(self.__xdata, self.__W) + self.__b
		y = sigmoid (z)
		return -np.sum ( (self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log(1-y+delta)) )
	def error_val(self):
		delta = 1e-7
		z = np.dot(self.__xdata, self.__W) + self.__b
		y = sigmoid (z)
		return -np.sum ( (self.__tdata*np.log(y+delta)+(1-self.__tdata)*np.log(1-y+delta)) )
	def train(self):
		f = lambda x : self.__loss_func ()
		for step in range(8001):
			self.__W -= self.__learning_rate * numerical_derivative (f, self.__W)
			self.__b -= self.__learning_rate * numerical_derivative (f, self.__b)
			if (step % 400 == 0):
				print("step = ", step, "error value = ", self.error_val())
	def predict(self, input_data):
		z = np.dot(input_data,self.__W) + self.__b
		y = sigmoid (z)
		if y > 0.5 :
			result = 1
		else:
			result = 0
		return y, result
xdata = np.array([ [0,0], [0,1], [1,0], [1,1] ])
tdata = np.array([0, 0, 0, 1])
AND_obj = LogicGate("AND_GATE", xdata, tdata)
AND_obj.train()
print(AND_obj.name, "\n")
test_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])
for input_data in test_data:
	(sigmoid.val, logical_val) = AND_obj.predict(input_data)
	print(input_data, " = ", logical_val, "\n")
tdata = np.array([0, 1, 1, 1])
OR_obj = LogicGate("OR_GATE", xdata, tdata)
OR_obj.train()
print(OR_obj.name, "\n")
test_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])
for input_data in test_data:
	(sigmoid.val, logical_val) = OR_obj.predict(input_data)
	print(input_data, " = ", logical_val, "\n")
tdata = np.array([1, 1, 1, 0])
NAND_obj = LogicGate("NAND_GATE", xdata, tdata)
NAND_obj.train()
print(NAND_obj.name, "\n")
test_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])
for input_data in test_data:
	(sigmoid.val, logical_val) = NAND_obj.predict(input_data)
	print(input_data, " = ", logical_val, "\n")
tdata = np.array([0, 1, 1, 0])
XOR_obj = LogicGate("XOR_GATE", xdata, tdata)
XOR_obj.train()
print(XOR_obj.name, "\n")
test_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])
for input_data in test_data:
	(sigmoid.val, logical_val) = XOR_obj.predict(input_data)
	print(input_data, " = ", logical_val, "\n")

print("correct XOR\n")
input_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])
s1 = []
s2 = []
new_input_data = []
final_output = []
for index in range (len(input_data)):
	s1 = NAND_obj.predict(input_data[index])
	s2 = OR_obj.predict(input_data[index])

	new_input_data.append(s1[-1])
	new_input_data.append(s2[-1])
	(sigmoid.val, logical_val) = AND_obj.predict(np.array(new_input_data))
	final_output.append(logical_val)
	new_input_data = []
for index in range(len(input_data)):
	print(input_data[index], " = ", final_output[index], end='')
	print("\n")