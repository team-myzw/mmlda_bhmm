# -*- coding: utf-8 -*-
#!/usr/bin/env python

import codecs
import numpy
import random
import os
import math
import shutil
import time
import codecs

from multiprocessing import Process, Lock


BOS = "<s>"
EOS = "</s>"

SMALL_LOG_PROB = -1e8
SMALL_NUMBER = 1e-8

class InferredWords:
    def __init__( self, word, score ):
        self.word = word
        self.score = score


class StateInfo:
    def __init__( self, sId, wId, w ):
        self.sId = sId
        self.wId = wId
        self.w   = w




#################################################################################################################
def MakeDir( foldername ):
    try:
        os.makedirs( foldername )
    except:
        print "Folder %s already exists \n" % foldername
        pass


def FindKeyFromValue( dictionary, val ):
    key = None
    for k, v in dictionary.items():
        if v == val:
            key = k
            break
    return key


def CalcLogProb( val ):
    logVal = SMALL_LOG_PROB
    try:
        logVal = math.log( val )
    except:
        pass
    return logVal


def SampleIndex( P, NUM_OF_CONCEPT, NUM_OF_SAMPLE_INDEX ):
    indices = [0] * (NUM_OF_CONCEPT + 2)
    for n in range(NUM_OF_SAMPLE_INDEX):
        # 累積確率密度
        cumP = [0] * len(P)
        cumP[0] = P[0]
        for i in range(1, len(P)):
            cumP[i] = cumP[i-1] + P[i]

        # サンプリング用の乱数生成
        r = random.random() * cumP[-1]

        # インディクスの探索
        ind = -1
        for i in range( len(P) ):
            if cumP[i] >= r:
                ind = i
                break
        if ind == -1:
            exit( "Error: In SampleIndex function, negative-index" )

        indices[ind] += 1

    ind = indices.index( max( indices ) )

    return ind
#################################################################################################################


#################################################################################################################
def LoadWordLists( filename ):
    wordLists = []
    lines = codecs.open( filename, "r" )
    for l in lines:
        wordLists.append( l )
    return wordLists


def CreateWordsMap( wordLists ):
    wordsMap = {}
    wordsMap[ BOS ] = 0
    cnt = 1
    for w in wordLists:
        wordsMap[ w.strip() ] = cnt
        cnt += 1
    wordsMap[ EOS ] = cnt
    return wordsMap


# 文の訓練データを読込
def LoadSentenceData( filename ):
    lines = codecs.open( filename, "r" ).readlines()
    wordsInSentences = []

    # 文にBOS，EOSがあるかどうかを確認
    bIsBOSandEOSexist = False
    if lines[0].split()[0] == BOS:
        bIsBOSandEOSexist = True

    for line in lines:
        words = []
        if not bIsBOSandEOSexist:
            words.append( BOS )
        line = line.split()
        for l in line:
            words.append( l )
        if not bIsBOSandEOSexist:
            words.append( EOS )
        wordsInSentences.append( words )
    return wordsInSentences


# 概念マップの読込
# 現状これは，最大なものしか入っていない．
# 今後，相互情報量×予測確率の値を保持．値がバラバラなので，概念毎に正規化すべき？
def LoadMapConcepts( filename ):
    lines = codecs.open( filename, "r" )
    mapConcepts = {}
    for line in lines:
        line = line.split()
        mapConcepts[ line[0] ] = int(line[1])
    mapConcepts[ BOS ] = 0
    mapConcepts[ EOS ] = max( mapConcepts.values() ) + 1
    return mapConcepts


# 全シーンにおける予測単語．予め計算しておく．
# 入力：単語リスト，ファイル名（物体，動き，場所）
# 出力：ソートされた<単語，スコア>の辞書
def LoadAndSortMaxWordsPerScenes( wordLists, filename, con ):
    allScores = numpy.loadtxt( filename )
    infWordsInAllScenes = []
    numline = 0
    for scores in allScores:
        sortedIndices = [ i[0] for i in sorted( enumerate( scores ), key=lambda x:x[1], reverse=True ) ]
        infWords = []
        for i in sortedIndices:
            infWords.append( InferredWords( wordLists[i], scores[i] ) )

        numline += 1
        infWordsInAllScenes.append( infWords )
    return infWordsInAllScenes
#################################################################################################################

#################################################################################################################
#bhmmの品詞推定した結果を使用する
def CalcConceptTransition( foldername, bhmmfile ):
    conCount = numpy.loadtxt(bhmmfile)
    nI, nJ = conCount.shape
    conTrans = numpy.zeros( [nI, nJ] )
    for i in range( nI ):
        # sum out over rows
        sum_i = sum( conCount[i, :]  )
        if sum_i > 0:
            for j in range( nJ ):
                conTrans[i, j] = conCount[i, j] / sum_i
    numpy.savetxt( foldername + "/SenResult/conceptsTransition.txt", conTrans, delimiter="\t", fmt="%.5g" )
    return conTrans
#################################################################################################################


#################################################################################################################
#言語モデルの計算
def CalcLanguageModel( foldername, wordLists, wordsInSentences, delta = 0.001 ):
    wordsMap = CreateWordsMap( wordLists )
    langModel = numpy.zeros( [len(wordsMap), len(wordsMap)] )
    for words in wordsInSentences:
        for i in range( len(words) - 1 ):
            langModel[ wordsMap[ words[i] ], wordsMap[ words[i+1] ] ] += 1.0
    numpy.savetxt( foldername + "/SenResult/cntLangModel.txt", langModel, delimiter="\t", fmt="%d" )
    nI, nJ = langModel.shape
    for i in range( nI ):
        # sum out over rows
        sum_i = sum( list( langModel[i, :] ) ) + delta * len(wordsMap)
        if sum_i > 0:
            for j in range( nJ ):
                # 言語モデルの調整
                langModel[i, j] = (langModel[i, j] + delta) / sum_i
    numpy.savetxt( foldername + "/SenResult/langModel.txt", langModel, delimiter="\t", fmt="%.5g" )
    return wordsMap, langModel
#################################################################################################################


#################################################################################################################
def FindViterbiPath( wordsMap, langModel, inferredWords, concepts, wordsNBest ):
    wordStates = []
    # 必ずBOS後，EOS前
    for i in range( 1, len(concepts)-1 ):
        states = []
        for s in range( wordsNBest ):
            cC = concepts[i] - 1
            cW = inferredWords[cC][s].word.strip()
            states.append( StateInfo( s, wordsMap[cW], cW ) )
        wordStates.append( states )


    bestPath     = [-1] * len(concepts)
    bestStPath   = [-1] * (len(concepts) - 1)
    bestScore    = 0.0
    viterbiPath  = numpy.zeros( [wordsNBest, len(bestPath)-1], numpy.dtype(int) )
    viterbiScore = numpy.zeros( [wordsNBest, len(bestPath)-1] )

    for t, ct in enumerate( range( 1, len(concepts) ) ):
        for s in range( wordsNBest ):
            if ct == 1:
                cC = concepts[ct] - 1
                cW = inferredWords[cC][s].word.strip()
                cS = inferredWords[cC][s].score

                viterbiPath [s, t] = s
                lm = CalcLogProb( langModel[ wordsMap[BOS], wordsMap[cW] ] )
                viterbiScore[s, t] = math.log( cS ) + lm
            elif ct == len(concepts) - 1:
                pC = concepts[ct-1] - 1
                pW = inferredWords[pC][s].word.strip()
                viterbiPath [s, t] = s
                lm = CalcLogProb( langModel[ wordsMap[pW], wordsMap[EOS] ] )
                viterbiScore[s, t] = viterbiScore[s, t-1] + lm
            else:
                score = [-1] * wordsNBest
                cC = concepts[ct] - 1
                cW = inferredWords[cC][s].word.strip()
                cS = inferredWords[cC][s].score

                #単語予測の結果が0の時の処理（適当）
                if cS == 0.0:
                    cS = SMALL_NUMBER

                pC = concepts[ct-1] - 1
                for si in range( wordsNBest ):
                    pW = inferredWords[pC][si].word.strip()
                    lm = CalcLogProb( langModel[ wordsMap[pW] , wordsMap[cW] ] )
                    score[si] = viterbiScore[si, t-1] + math.log( cS ) + lm
                viterbiScore[s, t] = max( score )
                viterbiPath [s, t] = score.index( viterbiScore[s, t] )


    bestStPath[-1] = viterbiScore[:, -1].argmax()
    for t in range( len(bestStPath)-1, 0, -1 ):
        bestStPath[t-1] = viterbiPath[ bestStPath[t] , t ]

    bestPath[0] = wordsMap[BOS]
    for t, i in enumerate( range( 1, len(concepts)-1 ) ):
        bestPath[i] = wordStates[t][ bestStPath[t] ].wId
    bestPath[-1] = wordsMap[EOS]
    bestScore = max( viterbiScore[:, -1] )
    return bestPath, bestScore


# 文のサンプリング，スコアが高いN個（NBest）を返す
# 与えられたシーンから，各概念のカテゴリを予測した結果を入力
# 返り値は，文のサンプル，スコアと長さ
def SampleSentence( foldername, lock, cntSentence, langModel, wordsMap, wordLists, mapConcepts, conTrans, infWordsOfAScene, wordsNBest, NUM_OF_CONCEPT, NUM_OF_SAMPLE_INDEX ):
    # 文末が見つかるまで，概念の生成を行う
    concepts = []
    concepts.append( mapConcepts[ BOS ] )
    while 1:
        cId = SampleIndex( conTrans[ concepts[-1], : ], NUM_OF_CONCEPT, NUM_OF_SAMPLE_INDEX )
        concepts.append( cId )
        if cId == mapConcepts[ EOS ]:
            break

    print "concepts",concepts
    # time.sleep( 10 )

    # Viterbiアルゴリズムを用いて，最尤パスを探索
    bestPath, bestScore = FindViterbiPath( wordsMap, langModel, infWordsOfAScene, concepts, wordsNBest )

    # 概念遷移の確率の計算
    pCs = 0.0
    for i in range( 1, len(concepts) ):
        pCs += math.log( conTrans[ concepts[i-1], concepts[i] ] )

    # Vitebiパスより，文サンプルを生成
    # sentence = u""
    sentence = ""
    for v in bestPath:
        w = FindKeyFromValue( wordsMap, v )
        sentence += w + " "
    bestScore += pCs

    # 結果をテキストファイルで保存
    lock.acquire()
    fw = codecs.open( foldername + "/Samples/%03dSamples.txt" % cntSentence, "a" )
    fw.write( "%d\t%.05g\t%s\n" % (len(concepts)-2, bestScore, sentence) )
    fw.close()
    fw = codecs.open( foldername + "/Samples/%03dConcepts.txt" % cntSentence, "a" )
    for c in concepts:
        fw.write( "%d\t" % c )
    fw.write( "%d\n" % len(concepts) )
    fw.close()
    lock.release()


def GenerateSentence( foldername, cntSentence, langModel, wordsMap, wordLists, mapConcepts, conTrans, infWordsOfAScene, numOfSamples, wordsNBest, NUM_OF_CONCEPT, NUM_OF_SAMPLE_INDEX, NBest ):

    MakeDir( foldername + "/Samples" )
    # サンプル生成，並列処理で行う
    lock = Lock()
    procs = []
    for n in range( numOfSamples ):
        proc = Process( target=SampleSentence, args=( foldername, lock, cntSentence, langModel, wordsMap, wordLists, mapConcepts, conTrans, infWordsOfAScene, wordsNBest, NUM_OF_CONCEPT, NUM_OF_SAMPLE_INDEX) )
        proc.start()
        procs.append( proc )
    print "Waiting for threads ... \n"
    for proc in procs:
        proc.join()

    # 生成したサンプルの読込
    senLens      = []
    allSentences = []
    allScores    = []
    dummyScore   = 0.0
    cntLens      = 0.0
    lines = codecs.open( foldername + "/Samples/%03dSamples.txt" % cntSentence, "r" )
    for line in lines:
        line = line.split( "\t" )
        l   = int  ( line[0] )
        sc  = float( line[1] )
        sen = line[2].strip()
        allSentences.append( sen )
        allScores.append( sc )
        dummyScore += sc
        cntLens    += l
        senLens.append( l )

    dummyScore = dummyScore / cntLens
    maxLen = max( senLens )

    # サンプル中から，スコアを文の長さで調整し，その中のNBestの結果を出す
    sentences = []
    scores    = []
    for i in range( len(allScores) ):
        allScores[i] += (maxLen - senLens[i]) * dummyScore

    sortedIndices = [ i[0] for i in sorted( enumerate( allScores ), key=lambda x:x[1], reverse=True ) ]
    for n in range( NBest ):
        i = sortedIndices[n]
        sentences.append( allSentences[i] )
        scores.append( allScores[i] )
    return sentences, scores
#################################################################################################################


def main( foldername, NUM_OF_CONCEPT, NUM_OF_SAMPLES, NUM_OF_WORDS, NUM_OF_SAMPLE_INDEX ):

    print "Sentence generator \n"
    try:
        shutil.rmtree( foldername + "/Samples" )
    except:
        print "No such directory"
        pass
    MakeDir( foldername + "/SenResult" )

    #単語リストの読み込み
    wordLists = LoadWordLists( foldername + "/result/wordlist.txt" )
    #概念選択の結果の読み込み
    mapConcepts = LoadMapConcepts( foldername + "/mapConcept.txt" )
    #教示文の読み込み
    wordsInSentences = LoadSentenceData( "sentences.txt" )

    #文法の読み込み
    conTrans = CalcConceptTransition( foldername, foldername + "/result/Nyy.txt")

    # 言語モデルの学習
    wordsMap, langModel = CalcLanguageModel( foldername, wordLists, wordsInSentences )

    # 入力シーンのロード
    allInfWordsInAllScenes = []
    for c in range(NUM_OF_CONCEPT):
        allInfWordsInAllScenes.append( LoadAndSortMaxWordsPerScenes( wordLists, foldername + "/hmm_Pw/%03d.txt" % c, c ) )
    sceneNum = len( allInfWordsInAllScenes[0] )

    fw = codecs.open( foldername + "/SenResult/sentences.txt", "w" )
    for s in range( sceneNum ):
        # 文生成
        print "**************************************************************"
        infWordOfAScene = []
        for c in range( len(allInfWordsInAllScenes) ):
            infWordOfAScene.append( allInfWordsInAllScenes[c][s] )

        sentences, scores = GenerateSentence( foldername, s, langModel, wordsMap,
                                wordLists, mapConcepts, conTrans,
                                infWordOfAScene, NUM_OF_SAMPLES, NUM_OF_WORDS, NUM_OF_CONCEPT, NUM_OF_SAMPLE_INDEX, NBest=1 )
        for i in range( len(sentences) ):
            fw.write( "%.05g\t%s\n" % (scores[i], sentences[i]) )
            print  scores[i], sentences[i]
        print "**************************************************************"
