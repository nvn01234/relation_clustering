from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import os
import glob
from scipy.spatial import distance

N_CLUSTERS = 50

def calculate_distance_matrix():
    vectors = np.loadtxt("data/vectors.txt")
    distance_matrix = np.zeros([len(vectors), len(vectors)])
    distances = []
    for i in range(len(vectors) - 1):
        for j in range(i+1, len(vectors)):
            v1 = vectors[i]
            v2 = vectors[j]
            dis = distance.euclidean(v1, v2)
            distance_matrix[i, j] = distance_matrix[j, i] = dis
            distances.append(dis)
    np.save("data/distance_matrix.npy", distance_matrix)
    np.save("data/distances.npy", distances)
    print("max distance: %.4f, average distance: %.4f, min distance: %.4f" % (np.max(distances), np.average(distances), np.min(distances)))

def clustering():
    vectors = np.loadtxt("data/vectors.txt")
    eps = 10
    min_samples = 10
    print("eps: %.4f" % eps)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)
    labels = np.array(clusterer.labels_, dtype='int32')
    np.savetxt("data/clusters.txt", labels, fmt="%d")


if __name__ == "__main__":
    # calculate_distance_matrix()
    clustering()
