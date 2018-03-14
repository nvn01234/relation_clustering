from gensim.models.wrappers import FastText


def main():
    print("read word embeddings")
    word2vec = FastText.load_fasttext_format("data/embeddings/wiki.vi.bin").wv

    print("read relations")
    vectors = []
    metadata = []
    with open("data/RelationNormalize.txt", "r", encoding="utf8") as f:
        for line in f:
            w = line.strip().lower()
            if w in word2vec:
                metadata.append(w)
                vec = word2vec.word_vec(w)
                vec = list(map(str, vec))
                vectors.append("\t".join(vec))
            elif w.replace(" ", "_") in word2vec:
                w = w.replace(" ", "_")
                metadata.append(w)
                vec = word2vec.word_vec(w)
                vec = list(map(str, vec))
                vectors.append("\t".join(vec))
    del word2vec

    print("saving data")
    with open("data/vectors.txt", "w", encoding="utf8") as f:
        f.write("\n".join(vectors))
    del vectors
    with open("data/metadata.txt", "w", encoding="utf8") as f:
        f.write("\n".join(metadata))
    del metadata


if __name__ == "__main__":
    main()
