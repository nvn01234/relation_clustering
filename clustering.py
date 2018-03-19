from sklearn.cluster import KMeans
import numpy as np

N_CLUSTERS = 50

def main():
    vectors = np.loadtxt("data/vectors.txt")
    clusterer = KMeans(n_clusters=N_CLUSTERS).fit(vectors)
    labels = clusterer.labels_
    with open("data/metadata.txt", "r", encoding="utf8") as f:
        words = [l.strip() for l in f.readlines()]

    clusters = np.zeros([N_CLUSTERS, 0])
    clusters = list(map(list, clusters))
    for label, word in zip(labels, words):
        clusters[label].append(word)
    clusters = [", ".join(cluster) for cluster in clusters]
    with open("data/clusters.txt", "w", encoding="utf8") as f:
        f.write("\n".join(clusters))



if __name__ == "__main__":
    main()
