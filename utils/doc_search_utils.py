from gensim import corpora, models, similarities
import jieba
import numpy as np

with open("../data/test_part.text", "r", encoding="utf-8") as f:
    text_lst = f.readlines()

def get_similar_doc(query):
    text_lst_base = [[i for i in jieba.lcut(item)] for item in text_lst]

    dictionary = corpora.Dictionary(text_lst_base)

    corpus = [dictionary.doc2bow(item) for item in text_lst_base]

    tf = models.TfidfModel(corpus)

    num_features = len(dictionary.token2id.keys())

    index = similarities.MatrixSimilarity(tf[corpus], num_features=num_features)

    query_words = [word for word in jieba.lcut(query)]

    query_vec = dictionary.doc2bow(query_words)

    sims = index[tf[query_vec]]
    sims_sort = np.argsort(sims)[::-1][:3]
    print(sims_sort)

    res = set()
    for i in sims_sort.tolist():
        if i !=0:
            res.add(text_lst[i-1])

        res.add(text_lst[i])

        if i != len(sims):
            res.add(text_lst[i + 1])

    return "".join(list(res))



if __name__ == '__main__':
    with open("../data/test_part.text", "r", encoding="utf-8") as f:
        text_lst = f.readlines()

    for text in text_lst:
        text = text.strip()

    print(get_similar_doc("如何通过中央显示屏进行副驾驶员座椅设置？"))

