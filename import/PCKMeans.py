import numpy as np
from scipy import spatial as sdist
import copy
from sklearn.metrics import pairwise_distances
import time

def l2_distance(point1, point2):
	return sum([(float(i) - float(j)) ** 2 for (i, j) in zip(point1, point2)])

def tolerance(tol, dataset):
	n = len(dataset)
	dim = len(dataset[0])
	averages = [sum(dataset[i][d] for i in range(n)) / float(n) for d in range(dim)]
	variances = [sum((dataset[i][d] - averages[d]) ** 2 for i in range(n)) / float(n) for d in range(dim)]
	return tol * sum(variances) / dim

def PCKMeans(x, k, const_mat, weighted = False, tol=1e-4, max_iter = 20000, init_centroids = []):

	#n = number of instances / d = number of features
	n, d = np.shape(x)

	tol = tolerance(tol, x)
	
	#Initialize centroids
	if len(init_centroids) == 0:

		centroids = np.random.rand(k, d)
		dataset_diameter = np.max(pairwise_distances(x, metric='euclidean'))

		for i in range(k):
			centroids[i, :] = centroids[i, :] + np.min(x, 0)
	else:
		centroids = init_centroids

	start_time = time.time()
	#Get maximally separated instances
	max_distance = np.power(np.max(x, axis = 0) - np.min(x, axis = 0), 2)

	#Initialize partition
	partition = np.ones(n) * -1

	iters = 0

	while iters < max_iter:

		iters += 1

		#Assign each instance to its closest cluster centroid
		for i in range(n):

			instance = x[i,:]

			#Compute squared euclidean distances
			squared_diffs = np.sum(np.power(centroids - instance, 2), axis = 1)

			#Compute penalties
			penalties = np.zeros(k)

			for l in range(k):

				for j in range(n):

					#if the second instance has a label
					if partition[j] != -1:

						#if ML(i,j) and instance i is going to be assigned to a label other than the label of instance j
						if const_mat[i,j] == 1 and l != partition[j]:

							if weighted:

								penalties[l] += np.sum(np.power(instance - x[j,:], 2))
							else:

								penalties[l] += 1

						#if ML(i,j) and instance i is going to be assigned to a label equal to the label of instance j
						if const_mat[i,j] == -1 and l == partition[j]:

							if weighted:

								penalties[l] += np.sum((max_distance - np.power(instance - x[j,:], 2)))

							else:

								penalties[l] += 1

			partition[i] = np.argmin((squared_diffs + penalties)/2)

		#Recompute centroids
		old_centroids = copy.deepcopy(centroids)
		for i in range(k):

			cluster = x[np.where(partition == i)[0],:]
			if cluster.shape[0] > 0:
				centroids[i,:] = np.mean(cluster, axis = 0)
		
		#Compute centroid shift for stopping criteria
		shift = sum(l2_distance(centroids[i], old_centroids[i]) for i in range(k))
		#Alternative stopping criteria
		if shift <= tol:
			break

	return partition, iters, time.time() - start_time