# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy

#品詞の推定結果
def concept_prob( word, concept, C, W ):

    path = "result/Nyx.txt"
    if os.path.exists(path):
        Nyx = numpy.loadtxt(path)
    else:
        Nyx = numpy.zeros((C, W))
    
    prob = (Nyx[concept,word]+ 0.01) / (sum(Nyx[:,word]) + C * 0.01)
    return prob

#品詞推定を考慮して各概念の単語情報を作成
def makeHist(wordAll, numOfAllConc):
    print"Start\n"

    #wordAll = numpy.loadtxt("result/words.All")

    N, W = wordAll.shape
    W_O = [[0.0 for i in range(W)] for j in range(N)]
    W_M = [[0.0 for i in range(W)] for j in range(N)]
    W_R = [[0.0 for i in range(W)] for j in range(N)]

    for n in range(N):
        for word in range( W ):
            if wordAll[n][word] > 0.0:
                prob = concept_prob(word, 1, numOfAllConc, W)
                W_O[n][word] = wordAll[n][word] * prob
                prob = concept_prob(word, 2, numOfAllConc, W)
                W_M[n][word] = wordAll[n][word] * prob
                prob = concept_prob(word, 3, numOfAllConc, W)
                W_R[n][word] = wordAll[n][word] * prob

    #numpy.savetxt( "result/words.Object", W_O, fmt="%0.1f", delimiter="\t" )
    #numpy.savetxt( "result/words.Motion", W_M, fmt="%0.1f", delimiter="\t" )
    #numpy.savetxt( "result/words.Reward", W_R, fmt="%0.1f", delimiter="\t" )
    return W_O, W_M

    print "Finish\n"

def makeHist_for_recog(wordAll):
    print"Start\n"

    N, W = wordAll.shape
    W_O = [[0.0 for i in range(W)] for j in range(N)]
    W_M = [[0.0 for i in range(W)] for j in range(N)]
    #W_R = [[0.0 for i in range(W)] for j in range(N)]

    for n in range(N):
        for word in range( W ):
            if wordAll[n][word] > 0.0:
                prob = concept_prob(word, 1)
                W_O[n][word] = wordAll[n][word] * prob
                prob = concept_prob(word, 2)
                W_M[n][word] = wordAll[n][word] * prob
                #prob = concept_prob(word, 3)
                #W_R[n][word] = wordAll[n][word] * prob

    numpy.savetxt( "result/words.Object", W_O, fmt="%0.1f", delimiter="\t" )
    numpy.savetxt( "result/words.Motion", W_M, fmt="%0.1f", delimiter="\t" )
    #numpy.savetxt( "result/words.Reward", W_R, fmt="%0.1f", delimiter="\t" )

    print "Finish\n"
    #return W_O, W_M, W_R
    return W_O, W_M
