from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import os
import glob
from scipy.spatial import distance

N_CLUSTERS = 50

def calculate_distance_matrix():
    vectors = np.loadtxt("data/vectors.txt")
    distance_matrix = np.zeros([len(vectors), len(vectors)])
    for i in range(len(vectors) - 1):
        for j in range(i+1, len(vectors)):
            v1 = vectors[i]
            v2 = vectors[j]
            distance_matrix[i, j] = distance_matrix[j, i] = distance.euclidean(v1, v2)
    np.save("data/distance_matrix.npy", distance_matrix)
    print("max distance: %.4f, average distance: %.4f, min distance: %.4f" % (np.max(distance_matrix), np.average(distance_matrix), np.min(distance_matrix)))

def main():
    vectors = np.loadtxt("data/vectors.txt")
    distance_matrix = np.load("data/distance_matrix.npy")
    min_dis = np.min(distance_matrix)
    clusterer = DBSCAN(eps=min_dis).fit(vectors)
    labels = clusterer.labels_
    # with open("data/metadata.txt", "r", encoding="utf8") as f:
    #     metadata = [l.strip() for l in f.readlines()]
    #
    # for f in glob.glob("data/clusters/*.txt"):
    #     os.remove(f)
    # if not os.path.exists("data/clusters"):
    #     os.mkdir("data/clusters")
    # for cluster in range(N_CLUSTERS):
    #     cluster_data = []
    #     for label, word in zip(labels, metadata):
    #         if label == cluster:
    #             cluster_data.append(word)
    #     with open("data/clusters/cluster_%d.txt" % cluster, "w", encoding="utf8") as f:
    #         f.write("\n".join(cluster_data))
    print("")


if __name__ == "__main__":
    calculate_distance_matrix()
    # main()
