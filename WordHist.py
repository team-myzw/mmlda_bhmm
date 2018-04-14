# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy

#品詞の推定結果
def concept_prob( word, concept, C, W, first=False, Nyx_path="result/Nyx.txt"):

    Nyx = numpy.zeros((C, W)) if first else numpy.loadtxt(Nyx_path)

    prob = (Nyx[concept,word]+ 0.01) / (sum(Nyx[:,word]) + C * 0.01)
    return prob

#品詞推定を考慮して各概念の単語情報を作成
def makeHist(wordAll, numOfAllConc, first=False, Nyx_path="result/Nyx.txt"):
    # print"Start\n"

    N, W = wordAll.shape
    W_O = [[0.0 for i in range(W)] for j in range(N)]
    W_M = [[0.0 for i in range(W)] for j in range(N)]
    W_P = [[0.0 for i in range(W)] for j in range(N)]

    for n in range(N):
        for word in range( W ):
            if wordAll[n][word] > 0.0:
                prob = concept_prob(word, 1, numOfAllConc, W, first, Nyx_path)
                W_O[n][word] = wordAll[n][word] * prob
                prob = concept_prob(word, 2, numOfAllConc, W, first, Nyx_path)
                W_M[n][word] = wordAll[n][word] * prob
                prob = concept_prob(word, 3, numOfAllConc, W, first, Nyx_path)
                W_P[n][word] = wordAll[n][word] * prob
    return W_O, W_M, W_P
