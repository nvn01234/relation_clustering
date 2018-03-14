from gensim.models.wrappers import FastText


def main():
    print("read word embeddings")
    word2vec = FastText.load_fasttext_format("data/embeddings/wiki.vi.bin").wv

    print("read relations")

    with open("data/RelationNormalize.txt", "r", encoding="utf8") as fin, \
         open("data/relations_x.txt", "w", encoding="utf8") as fx, \
         open("data/relations_y.txt", "w", encoding="utf8") as fy:
        for line in fin:
            w = line.strip().lower()
            if w in word2vec:
                fx.write(w)
                fy.write(" ".join(list(word2vec.word_vec(w))) + "\n")
            elif w.replace(" ", "_") in word2vec:
                w = w.replace(" ", "_")
                fx.write(w)
                fy.write(" ".join(list(word2vec.word_vec(w))) + "\n")


if __name__ == "__main__":
    main()
