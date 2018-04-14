# -*- coding: utf-8 -*-
#!/usr/bin/env python

import codecs
import numpy
import random
import math
import codecs
import os


# モデルのハイパパラメータ
ALPHA_WORD = 0.1

NUM_OF_ITER = 50

BOS = u"BOS"
EOS = u"EOS"

# 学習文のロード
# 学習文内の単語種類により辞書の作成
def LoadSentences( filename ):
    sentences = codecs.open( filename, "r").readlines()

    # 辞書の作成
    wordToIndex = {}
    index = 0
    for words in sentences:
        words = words.split()
        for w in words:
            if not wordToIndex.has_key( w ):
                wordToIndex[w] = index
                index += 1
    wordToIndex[BOS] = index
    wordToIndex[EOS] = index + 1
    sentences = codecs.open( filename, "r").readlines()
    # 学習データのインデックシング
    senData = []
    for words in sentences:
        words = words.split()
        ww = []
        ww.append( wordToIndex[BOS] )
        for w in words:
            ww.append( wordToIndex[w] )
        ww.append( wordToIndex[EOS] )
        senData.append( ww )
    return wordToIndex, senData


def SampleOne(probs):
    z = sum(probs)
    remaining = random.uniform(0.0, z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    print "Error in SampleOne -> p: ", probs, " z:%f, i:%d \n" % (z, i)


def SampleTag(t, Y, X, Nyy, Nyx, wordsNum, NUM_OF_TAGS, ALPHA_TAG):
    prevY = Y[t-1]
    nextY = Y[t+1]
    currY = Y[t]
    currX = X[t]

    Nyy[prevY, currY] -= 1
    Nyy[currY, nextY] -= 1
    Nyx[currY, currX] -= 1

    #print "Before: ", t, " -> ", prevY, currY, nextY

    p = [0.0] * NUM_OF_TAGS
    for tag in range(1, NUM_OF_TAGS+1):
        # p[tag] = P(tag | prevY) * P(nextY | tag) * P(X[t] | tag)
        p1 = ( Nyy[prevY, tag] + ALPHA_TAG ) / ( sum( Nyy[prevY, :] ) + NUM_OF_TAGS * ALPHA_TAG )
        p2 = ( Nyy[tag, nextY] + ALPHA_TAG ) / ( sum( Nyy[tag  , :] ) + NUM_OF_TAGS * ALPHA_TAG )
        p3 = ( Nyx[tag, currX] + ALPHA_WORD * (1.0/wordsNum)) / ( sum( Nyx[tag  , :] ) + ALPHA_WORD)
        #p3 = ( Nyx[tag, currX] + ALPHA_WORD) / ( sum( Nyx[tag  , :] ) + wordsNum * ALPHA_WORD   )
        p[tag-1] = p1 * p2 * p3

    newY = SampleOne(p) + 1

    #print "After: ", t, " -> ", prevY, newY, nextY

    Nyy[prevY, newY] += 1
    Nyy[newY, nextY] += 1
    Nyx[newY, currX] += 1
    return newY


def CalcLogLikelihood(senData, Ysw, Nyy, Nyx, wordsNum, NUM_OF_TAGS, ALPHA_TAG):
    logLik = 0.0
    #print "Ysw:",Ysw
    #print "senData:",senData
    for s_i in range(len(Ysw)):
        Y = Ysw[s_i]
        X = senData[s_i]
        #print "len(X)", len(X)
        #print "Y", Y
        for i in range(1, len(X)):
            prevY = Y[i-1]
            currY = Y[i]
            currX = X[i]
            p = ( Nyy[prevY, currY] + ALPHA_TAG ) / ( sum( Nyy[prevY, :] ) + NUM_OF_TAGS * ALPHA_TAG ) * \
                ( Nyx[currY, currX] + ALPHA_WORD * (1.0/wordsNum)) / ( sum( Nyx[currY, :] ) + ALPHA_WORD   )
                #( Nyx[currY, currX] + ALPHA_WORD) / ( sum( Nyx[currY, :] ) + wordsNum * ALPHA_WORD   )
            logLik += math.log( p )
    return logLik

def sample_idx(prob ):
    accm_prob = [0,] * len(prob)
    for i in range(len(prob)):
        accm_prob[i] = prob[i] + accm_prob[i-1]

    rnd = random.random() * accm_prob[-1]
    for i in range(len(prob)):
        if rnd <= accm_prob[i]:
            return i



def SampleCorpus(senData, wordToIndex, NUM_OF_TAGS, NUM_OF_ALL_TAGS, ALPHA_TAG, NUM_OF_CONC, iterNum = NUM_OF_ITER ):
    # ************************** 初期化 **************************
    numOfWords = len( wordToIndex )
    Nyy = numpy.zeros( (NUM_OF_ALL_TAGS, NUM_OF_ALL_TAGS) )
    Nyx = numpy.zeros( (NUM_OF_ALL_TAGS, numOfWords) )

    concepts = codecs.open("mapConcept.txt", "r",)
    scores = codecs.open("mapConcept_scored.txt", "r").readlines()
    Concepts ={}
    for word in concepts:
        data = word.split("\t")
        Concepts.setdefault(data[0],int(data[1]))
    concepts.close()

    fw = open( "result/initLabels.txt", "w" )
    Ysw = []
    for s_i in range(len(senData)):
        #print "s_i:", s_i
        #print "Ysw:", Ysw
        y = []
        for w in senData[s_i]:
            if w == wordToIndex[BOS]:
                y.append( 0 )
            elif w == wordToIndex[EOS]:
                y.append( NUM_OF_TAGS+1 )
            else:
                for k, v in sorted(wordToIndex.items(), key=lambda x:x[1]):
                        if w == v:
                            for key, value in Concepts.items():
                                if key == k:
                                    if value == NUM_OF_CONC+1:
                                        y.append(random.randint(1,NUM_OF_TAGS))
                                        #y.append(4)
                                    else:
                                        conc = []
                                        prob = []
                                        for l, score in enumerate(scores[w].split("\t")):
                                            if not score == "\n":
                                                if l % 2 == 0:
                                                    prob.append(float(score))
                                                else:
                                                    conc.append(int(score))
                                        idx = sample_idx(prob)
                                        y.append(conc[idx])
                                    break
                            break
        Ysw.append(y)
        for i in range( len(y) ):
            fw.write( "%d\t" % y[i] )
        fw.write( "\n" )
        for i in range(1, len(y)):
            Nyy[y[i-1], y[i]] += 1
            Nyx[y[i], senData[s_i][i]] += 1
    concepts.close()
    fw.close()


    # ********************* サンプリング処理 **********************
    maxLikelihood = -1e100
    likelihoods = numpy.zeros( (iterNum, 1) )
    for n in range(iterNum):
        for s_i in range(len(Ysw)):
            Y = Ysw[s_i]
            X = senData[s_i]
            for t in range(1, len(Y)-1):
                Y[t] = SampleTag(t, Y, X, Nyy, Nyx, numOfWords, NUM_OF_TAGS, ALPHA_TAG)
            Ysw[s_i] = Y
        likelihoods[n, 0] += CalcLogLikelihood(senData, Ysw, Nyy, Nyx, numOfWords, NUM_OF_TAGS, ALPHA_TAG)

        # 学習結果の保存
        if maxLikelihood < likelihoods[n, 0]:
            # 推定されたパラメータ
            numpy.savetxt( "result/Nyy.txt", Nyy, delimiter="\t", fmt="%d" )
            numpy.savetxt( "result/Nyx.txt", Nyx, delimiter="\t", fmt="%d" )

            # 推定された品詞
            fw = codecs.open( "result/sentences_labels.txt", "w",  )
            fw_l = open( "result/labels.txt", "w", )
            count = 0
            for Y in Ysw:
                for x in senData[count]:
                    for k, v in sorted(wordToIndex.items(), key=lambda x:x[1]):
                        if x == v:
                            fw.write(k+"\t")
                            break
                fw.write("\n")
                count += 1
                for y in Y:
                    fw_l.write( "%d\t" % y )
                    fw.write( "%d\t" % y )
                fw.write( "\n" )
                fw_l.write( "\n" )
            fw.close()
            fw_l.close()
    # ************************************************************

    # 学習結果の保存：各イテレーションの尤度
    numpy.savetxt( "result/likelihoods.txt", likelihoods, delimiter="\t", fmt="%.10f" )
    return


# 初期状態は必ず文頭なので，初期確率は必要ない
def Viterbi(words, wordToIndex, transProb, emissProb):
    X = []
    for w in words:
        X.append( wordToIndex[w] )

    numOfData = len(X)
    bestPath = [-1] * numOfData
    bestProb = 0.0

    numOfStates = transProb.shape[0]
    viterbiPath = numpy.zeros( (numOfData, numOfStates), numpy.dtype(int) )
    viterbiProb = numpy.zeros( (numOfData, numOfStates) )

    # Recursion
    for t in range(numOfData):
        # Initialization
        if t == 0:
            for s in range(numOfStates):
                viterbiPath[t, s] = s
                if s == 0:
                    viterbiProb[t, s] = 1.0
                else:
                    viterbiProb[t, s] = 0.0
        else:
            for s_curr in range(numOfStates):
                probs = [0.0] * numOfStates
                for s_prev in range(numOfStates):
                    probs[s_prev] = viterbiProb[t-1, s_prev] * transProb[s_prev , s_curr] * emissProb[s_curr, X[t]]
                viterbiProb[t, s_curr] = max( probs )
                viterbiPath[t, s_curr] = probs.index( max( probs ) )

    # Path backtracking
    bestPath[-1] = viterbiProb[-1, :].argmax()
    for t in range( len(bestPath)-1, 0, -1 ):
        bestPath[t-1] = viterbiPath[ t, bestPath[t] ]
    bestProb = max( viterbiProb[-1, :] )
    return bestProb, bestPath


def LoadAndCalcHMMParams(NUM_OF_TAGS, ALPHA_TAG):
    Nyy = numpy.loadtxt( "result/Nyy.txt" )
    Nyx = numpy.loadtxt( "result/Nyx.txt" )

    # 文頭・文末を含んだ状態数
    numOfAllPossibleStates = Nyy.shape[0]

    # 文頭・文末を含まない状態数
    numOfStates = numOfAllPossibleStates - 2

    # 辞書内の単語数
    numOfWords = Nyx.shape[1]

    """
    # 初期遷移確率を文頭・文末以外の状態で均等
    initProb = [0.0] * numOfAllPossibleStates
    for i in range(1, numOfStates+1):
        initProb[i] = 1.0/numOfStates
    """

    # 遷移・出力確率の計算
    transitionProb = numpy.zeros( Nyy.shape )
    emissionProb   = numpy.zeros( Nyx.shape )

    # 遷移確率
    for i in range(numOfAllPossibleStates-1):
        sum_i = sum(Nyy[i, :]) + NUM_OF_TAGS * ALPHA_TAG
        for j in range(1, numOfAllPossibleStates):
            transitionProb[i, j] = (Nyy[i, j] + ALPHA_TAG) / sum_i

    # 出力確率
    for i in range(1, numOfAllPossibleStates):
        #sum_i = sum(Nyx[i, :]) + numOfWords * ALPHA_WORD
        sum_i = sum(Nyx[i, :]) + ALPHA_WORD
        for j in range(numOfWords):
            emissionProb[i, j] = (Nyx[i, j] + ALPHA_WORD*(1.0/numOfWords)) / sum_i
            #emissionProb[i, j] = (Nyx[i, j] + ALPHA_WORD) / sum_i

    numpy.savetxt( "result/tP.txt", transitionProb, delimiter="\t", fmt="%.4f" )
    numpy.savetxt( "result/eP.txt", emissionProb, delimiter="\t", fmt="%.4f" )

    #return initProb, transitionProb, emissionProb
    return transitionProb, emissionProb


def CalcStatesProb(Y, X, tP, eP):
    prob = 1.0
    for t in range(1, len(X)):
        prob *= tP[Y[t-1], Y[t]] * eP[Y[t], X[t]]
    return prob


def GetAllCombinations( Zs ):
    lenAllComb = 1
    lenZs = len( Zs )
    lenZsComb = [1] * lenZs
    for i in range( lenZs ):
        lenAllComb *= Zs[i]
        for z in Zs[i+1 : lenZs]:
            lenZsComb[i] *= z
    allComb = numpy.zeros( ( lenAllComb, lenZs ), dtype=int )

    for j in range( lenZs ):
        nums = range( Zs[j] )
        cntNum = 0
        for i in range( 0, lenAllComb, lenZsComb[j] ):
            for k in range( lenZsComb[j] ):
                allComb[i+k, j] = nums[cntNum]
            cntNum += 1
            if cntNum == Zs[j]:
                cntNum = 0
    return allComb


def GenerateStates(T, S):
    Zs = [S] * T
    allComb = GetAllCombinations( Zs )
    return allComb


def FindMaxPathBF(X, transProb, emissProb):
    Ymax = None
    Pmax = -1
    Ys = GenerateStates(len(X), transProb.shape[0])
    for Y in Ys:
        p = CalcStatesProb(Y, X, transProb, emissProb)
        if Pmax < p:
            Pmax = p
            Ymax = list(Y)
    return Pmax, Ymax


def main(numOfConc, numOfAllConc):

    NUM_OF_CONC = numOfConc
    NUM_OF_TAGS = numOfAllConc
    NUM_OF_ALL_TAGS = NUM_OF_TAGS + 2
    ALPHA_TAG  = 0.1 * 1.0 / (NUM_OF_ALL_TAGS)

    # print NUM_OF_TAGS, NUM_OF_ALL_TAGS

    wordToIndex, senData = LoadSentences( "LearnData/sentences.txt" )

    SampleCorpus( senData, wordToIndex, NUM_OF_TAGS, NUM_OF_ALL_TAGS, ALPHA_TAG, NUM_OF_CONC )
    transitionProb, emissionProb = LoadAndCalcHMMParams(NUM_OF_TAGS, ALPHA_TAG)
