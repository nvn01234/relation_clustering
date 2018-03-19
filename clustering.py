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
    distances = np.load("data/distances.npy")
    eps = np.average(distances)
    clusterer = DBSCAN(eps=eps).fit(vectors)
    labels = clusterer.labels_
    np.savetxt("data/cluters.txt", labels)


if __name__ == "__main__":
    # calculate_distance_matrix()
    clustering()
