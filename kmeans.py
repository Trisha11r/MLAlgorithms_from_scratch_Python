import argparse
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys

#compute euclidean distance
def compute_euclidean(pt1, pt2):
	dsquare= 0
	for i in range(len(pt1)):
		dsquare+= (pt1[i]-pt2[i])**2
	d = np.sqrt(dsquare)
	return d

#compute manhattan distance
def compute_manhattan(pt1, pt2):
	d = 0
	for i in range(len(pt1)):
		d += np.abs((pt1[i]-pt2[i]))
	return d

#find intial random centroid
def random_centroid(pt_set, k):
	centroid = []
	idx = np.random.choice(len(pt_set)-1, k)
	for i in range(k):
		centroid.append(pt_set[idx[i]])
	return centroid

#assign appropriate cluster to the data points
def assign_cluster(centroids, data_pts, mode= 'euclidean'):
	cassigned = []
	for i in range(len(data_pts)):
		d_centroids = []
		for c in centroids:
			if mode == 'euclidean':
				d_centroids.append(compute_euclidean(c,data_pts[i]))
			elif mode == 'manhattan':
				d_centroids.append(compute_manhattan(c,data_pts[i]))

		assignment = np.where(d_centroids == min(d_centroids))
		cassigned.append(assignment[0][0])
	return cassigned

#compute updated centroid of the current cluster assignment
def compute_centroid(data_pts, cluster_array, k):
	clusters = []
	for i in range(k):
		clusters.append(np.zeros((1,5)))

	for i in range(len(data_pts)):
		a = np.vstack((clusters[cluster_array[i]],np.array(data_pts[i])))
		clusters[cluster_array[i]] = a
		
	mean_vals = []
	for i in range(k):
		mean_vals.append(np.mean(clusters[i], axis = 0))

	new_centroid = np.array(mean_vals)
	return new_centroid


def main():
	parser = argparse.ArgumentParser(description='Kmeans')
	parser.add_argument('--dataset', type=str, help='dataset path')
	parser.add_argument('--metric', type=str, help='distance metric', default = 'euclidean')
	args = parser.parse_args()
	minmax_scaler = preprocessing.MinMaxScaler()
	#k-value taken to be 2
	k=2
	
	data = np.loadtxt(args.dataset, dtype=str, delimiter=',')
	dataset = np.array(data[1:]).astype(np.float)
	labels = dataset[:, -1] 
	dataset = dataset[:, :-1]
	#data normalization
	dataset = minmax_scaler.fit_transform(dataset)

	#intial centroid and cluster assignment 
	init_centroid = random_centroid(dataset, k)
	clusters = assign_cluster(init_centroid, dataset, args.metric)
	
	#update clusters until convergence (when subsequently same clusters obtained)
	i=0
	while(1):
		prev_clusters = clusters
		newc = compute_centroid(dataset, clusters, k)	
		clusters = assign_cluster(newc, dataset, args.metric)
		if (clusters == prev_clusters):
			break

	#Calculation of percentages of each diagnosis types in the clusters obtained
	zero_clusters = 0
	one_clusters = 0
	zc_lb_0 = 0
	zc_lb_1 = 0
	oc_lb_0 = 0
	oc_lb_1 = 0

	for i in range(len(clusters)):
		if clusters[i] ==0:
			zero_clusters+= 1
			if labels[i]==0:
				zc_lb_0 += 1
			elif labels[i]==1:
				zc_lb_1 += 1

		elif clusters[i] ==1:
			one_clusters += 1
			if labels[i]==0:
				oc_lb_0 += 1
			elif labels[i]==1:
				oc_lb_1 += 1

	print("Percentage of cluster zero contents:")
	print("Diagnosis 0:" , zc_lb_0/zero_clusters*100)
	print("Diagnosis 1:" , zc_lb_1/zero_clusters*100)
	print("Percentage of cluster one contents:")
	print("Diagnosis 0:" , oc_lb_0/one_clusters*100)
	print("Diagnosis 1:" , oc_lb_1/one_clusters*100)



if __name__ == '__main__':
	main()