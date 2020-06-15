import argparse
import numpy as np
from sklearn import preprocessing

def euclid_dist(pt1, pt2):
	dsquare= 0
	for i in range(len(pt1)):
		dsquare+= (pt1[i]-pt2[i])**2
	d = np.sqrt(dsquare)
	return d

def predict(k, train_input, train_labels, test_row):

	dist_arr = []

	for i in range(len(train_input)):
		dist_arr.append([i, euclid_dist(train_input[i], test_row)])

	dist_arr.sort(key = lambda d: d[1])

	num_l0, num_l1 = 0, 0

	j=0
	while(j<k):
		idx = dist_arr[j]
		
		if int(train_labels[idx[0]]) == 0:
			num_l0 += 1
		else:
			num_l1 += 1 

		j+=1

	if num_l0 > num_l1:
		return 0
	else:
		return 1

def get_accuracy(k, train_input, train_labels, test_input, test_labels):

	num_corr = 0
	
	for i, trow in enumerate(test_input):
		
		predict_val = predict(k, train_input, train_labels, trow)

		if predict_val == int(test_labels[i]):
			num_corr += 1

	accuracy = num_corr/len(test_labels)

	return accuracy

def main():
	parser = argparse.ArgumentParser(description='KNN')
	parser.add_argument('--dataset', type=str, help='dataset')
	parser.add_argument('--num-epochs', type=int, help='num epochs')
	parser.add_argument('--k', type=int, help='k val')
	args = parser.parse_args()

	minmax_scaler = preprocessing.MinMaxScaler()

	data = np.loadtxt(args.dataset, dtype=str, delimiter=',')
	dataset = np.array(data[1:]).astype(np.float)
	dataset = minmax_scaler.fit_transform(dataset)
	

	accuracies = []
	for epoch in range(args.num_epochs):
		np.random.shuffle(dataset)
		y = dataset[:, -1] 
		X = dataset[:, :-1]
		X1 = dataset
		train_percent_idx = int(0.80 * len(X1))
		train_input, test_input = X[:train_percent_idx], X[train_percent_idx:]
		train_labels, test_labels = y[:train_percent_idx], y[train_percent_idx:]

		accuracy = get_accuracy(args.k, train_input, train_labels, test_input, test_labels)
		print ('accuracy for epoch ', epoch+1, ' : ', accuracy)
		accuracies.append(accuracy)


	print ('final accuracy averaged over', args.num_epochs, 'epochs: ', sum(accuracies)/args.num_epochs)

	

if __name__ == "__main__":
	main()
