import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def train(data_dict):
	theta = np.zeros(len(data_dict['inputs'][0]))
	T = 100 #epochs
	wts = np.zeros(len(data_dict['inputs'][0]))

	max_fval = float("-inf")

	for y_i in data_dict['inputs']:
		if np.amax(y_i) > max_fval:
			max_fval=np.amax(y_i)

	step_size = max_fval / 10
	np.random.seed(96)
	
	sum_weights = 0
	
	for _ in range (T):
		wts = step_size*theta

		sum_weights = np.add(sum_weights, wts)

		idx = np.random.randint(0, len(data_dict['inputs']))
		x_i = data_dict['inputs'][idx]
		lb_i = data_dict['labels'][idx]
		
		if lb_i * np.dot(wts, x_i) < 1:
			theta += lb_i* x_i
			step_size /= 10


	weights = sum_weights / T
	weights[0] = 1
	return weights


def test(data_dict, w):
	op = np.sign(np.dot(data_dict['inputs'], w))
	return op

def draw(data_dict, w):

	fig = plt.figure()
	ax= fig.add_subplot(1,1,1)

	pos1 = get_hplane(0, [w[1], w[2]], w[0], 1)
	pos2 = get_hplane(10,[w[1], w[2]], w[0], 1)
	ax.plot([-5,10],[pos1,pos2],'r--')

	mid1 = get_hplane(0, [w[1], w[2]], w[0], -0.175)
	mid2 = get_hplane(10,[w[1], w[2]], w[0], -0.175)
	ax.plot([-5,10],[mid1,mid2],'k')

	neg1 = get_hplane(0, [w[1], w[2]], w[0], -1.35)
	neg2 = get_hplane(10,[w[1], w[2]], w[0], -1.35)
	ax.plot([-5,10],[neg1,neg2],'r--')

	for i in range(len(data_dict['inputs'])):
		if data_dict['labels'][i] == -1:
			ax.scatter(data_dict['inputs'][i][1], data_dict['inputs'][i][2], s=30, c='m', alpha=0.5, edgecolors='b')
		else:
			ax.scatter(data_dict['inputs'][i][1], data_dict['inputs'][i][2], s=30, c='y', alpha=0.5, edgecolors='g')

	plt.show()
        
def get_hplane(x, w, b, v):
    return (-w[0] * x - b + v) / w[1]
       

def main():
	
	X0, y = make_blobs(n_samples=100, n_features = 2, centers=2,
cluster_std=1.05, random_state=10)
	X1 = np.c_[np.ones((X0.shape[0])), X0] # add one to the x-values to incorporate bias

	train_percent_idx = int(0.8 * len(X1))
	train_input, test_input = X1[:train_percent_idx], X1[train_percent_idx:]
	train_labels, test_labels = y[:train_percent_idx], y[train_percent_idx:]

	train_data_dict = {}
	test_data_dict = {}

	train_data_dict['inputs'] = train_input
	train_data_dict['labels'] = train_labels

	test_data_dict['inputs'] = test_input
	test_data_dict['labels'] = test_labels

	for i in range(len(train_labels)):
		if train_labels[i] == 0:
			train_labels[i]=-1

	for i in range(len(test_labels)):
		if test_labels[i] == 0:
			test_labels[i]=-1
	
	op_wt =   (train(train_data_dict))

	op = test(test_data_dict, op_wt)
	count = 0

	for i in range(len(test_labels)):
		if test_labels[i]== op[i]:
			count+=1

	accuracy = count/ len(test_labels)
	print ('Accuracy = ', str(accuracy * 100) + '%')
	draw(train_data_dict, op_wt)
		

if __name__== '__main__':
	main()
