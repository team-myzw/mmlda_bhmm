# -*- coding: utf-8 -*-
#!/usr/bin/env python

import math
import numpy
import os
import copy
import codecs

SMALL_NUMBER = 1e-8


# フォルダの作成
def MakeDir( foldername ):
    try:
        os.makedirs( foldername )
    except:
        print "%s already exist" % foldername
        pass


# 相互情報量の計算
# Nwzのカウントをもとに計算
def CalcMutualInformation( Nwz ):
    W, Z = Nwz.shape
    MIwz = numpy.zeros( (W, Z) )
    Nall = numpy.sum( Nwz )
    for w in range( W ):
        for z in range( Z ):
            nw1z1 = Nwz[ w, z ]
            nw0z1 = numpy.sum( Nwz[:, z] ) - Nwz[w, z]
            nw1z0 = numpy.sum( Nwz[w, :] ) - Nwz[w, z]
            nw0z0 = Nall - nw1z0 - nw0z1 - nw1z1
            Iw1z1 = float( nw1z1 ) / Nall * math.log( float( nw1z1 * Nall ) / ( (nw1z1 + nw1z0) * (nw0z1 + nw1z1) ) + SMALL_NUMBER )
            Iw0z1 = float( nw0z1 ) / Nall * math.log( float( nw0z1 * Nall ) / ( (nw0z1 + nw0z0) * (nw0z1 + nw1z1) ) + SMALL_NUMBER )
            Iw1z0 = float( nw1z0 ) / Nall * math.log( float( nw1z0 * Nall ) / ( (nw1z1 + nw1z0) * (nw0z0 + nw1z0) ) + SMALL_NUMBER )
            Iw0z0 = float( nw0z0 ) / Nall * math.log( float( nw0z0 * Nall ) / ( (nw0z0 + nw0z1) * (nw0z0 + nw1z0) ) + SMALL_NUMBER )
            if math.isnan(Iw1z1):
                Iw1z1 = 0.0
            if math.isnan(Iw0z1):
                Iw0z1 = 0.0
            if math.isnan(Iw1z0):
                Iw1z0 = 0.0
            if math.isnan(Iw0z0):
                Iw0z0 = 0.0
            MIwz[w, z] = Iw1z1 + Iw1z0 + Iw0z1 + Iw0z0
    return MIwz


# 予測した単語の確率の計算
def CalcWordsProbability( Nwz, alpha = 0.1 ):
    W, Z = Nwz.shape
    Pwz  = numpy.zeros( (W, Z) )
    for w in range( W ):
        for z in range( Z ):
            Pwz[ w, z ] = float( Nwz[w, z] + alpha ) / ( numpy.sum( Nwz[:, z] ) + W * alpha )
    return Pwz


def CalcPw( Pwz ):
    W, Z = Pwz.shape
    Pw = [0.0] * W
    for w in range( W ):
        for z in range( Z ):
            Pw[w] += Pwz[w, z]
    return Pw


# 相互情報量を重みとして予測した単語のスコアの計算
def InferWords( MIwz, Pwz ):
    W, Z = MIwz.shape
    Pw = CalcPw( Pwz )
    wPw = [0.0] * W
    for w in range( W ):
        maxMI_w = max( list( MIwz[w, :] ) )
        wPw[w] = maxMI_w * Pw[w]
    return Pw, wPw



# 重み付けされた単語確率から最大な最大な概念を探す
def FindMaxConcepts( weightedWordsProbs ):
    numOfConcepts = len( weightedWordsProbs )
    W = len( weightedWordsProbs[0] )
    maxConcepts = []
    maxScores   = []
    sortedConcepts = []
    sortedScores   = []
    for w in range( W ):
        scores = [0.0] * numOfConcepts
        for c in range( numOfConcepts-1 ):
            scores[c] = weightedWordsProbs[c][w]
        sortedInd = [ i[0] for i in sorted( enumerate( scores ), key=lambda x:x[1], reverse=True ) ]
        sortedSrc = [0.0] * numOfConcepts
        for c in range( numOfConcepts ):
            sortedSrc[c] = scores[ sortedInd[c] ]
            sortedInd[c] += 1
        sortedConcepts.append( sortedInd )
        sortedScores.append( sortedSrc )
        maxConcepts.append( sortedInd[0] )
        maxScores.append( sortedSrc[0] )
    return maxConcepts, maxScores, sortedConcepts, sortedScores


def CalcMIWithInferredWords( Nwz ):
    MIwz = CalcMutualInformation( Nwz )
    Pwz  = CalcWordsProbability( Nwz )
    Pw, wPw = InferWords( MIwz, Pwz )
    return MIwz, Pwz, Pw, wPw


def CalcMaxConceptOfWords( MIwzs, threshold ):
    numOfConcepts = len(MIwzs)
    W = MIwzs[0].shape[0]
    maxConcepts = []
    maxScores   = []
    sortedConcepts = []
    sortedScores   = []
    maxScorePerConcepts = []
    for w in range( W ):
        scores = [0.0] * numOfConcepts
        for c in range( numOfConcepts ):
            scores[c] = max( list( MIwzs[c][w, :] ) )
        sortedInd = [ i[0] for i in sorted( enumerate( scores ), key=lambda x:x[1], reverse=True ) ]
        sortedSrc = [0.0] * numOfConcepts
        for c in range( numOfConcepts ):
            sortedSrc[c] = scores[ sortedInd[c] ]
            sortedInd[c] += 1
        sortedConcepts.append( sortedInd )
        sortedScores.append( sortedSrc )
        if sortedSrc[0] < threshold:
            maxConcepts.append(numOfConcepts+1)
            maxScores.append(sortedSrc[0])
        else:
            maxConcepts.append( sortedInd[0] )
            maxScores.append( sortedSrc[0] )
        maxScorePerConcepts.append( scores )
    return maxConcepts, maxScores, sortedConcepts, sortedScores, maxScorePerConcepts


def CreateMapConcepts( wordsListFilename, maxConceptsFilename, sortedScoreFilename, sortedConceptFilename, threshold ):
    words = codecs.open( wordsListFilename, "r").readlines()
    mapConcepts = numpy.loadtxt( maxConceptsFilename, dtype=int )
    scores = numpy.loadtxt( sortedScoreFilename, dtype=float )
    sortedConcept = numpy.loadtxt( sortedConceptFilename, dtype=int )
    W, C = scores.shape
    fw = codecs.open( "mapConcept.txt", "w" )
    fw_init = codecs.open( "mapConcept_init.txt", "w" )
    fw_score = codecs.open( "mapConcept_scored.txt", "w")

    #print "len words,",len(words)
    #print "len mapConecepts:",len(mapConcepts)
    for i in range( len(words) ):
        fw.write( "%s\t%d\n" % ( words[i].strip(), mapConcepts[i] ) )
        fw_init.write( "%s\t%d\n" % ( words[i].strip(), mapConcepts[i] ) )
        if mapConcepts[i] == C+1:
            fw_score.write("0.0\n")
        else:
            for j in range(C):
                if scores[i][j] >= threshold:
                    fw_score.write(str(scores[i][j])+"\t" + str(sortedConcept[i][j]) + "\t")
                else:
                    break
            fw_score.write("\n")


def CalcHmmPw( conZ, Pz, Nwz, Nyx, concept):
    S  = conZ.shape
    S, Z = Pz.shape
    C, W = Nyx.shape
    W = W-2

    hmmPw = numpy.zeros( (S, W) )
    for s in range(S):
        for w in range(W):
            weight = (Nyx[concept,w]+ 0.01) / (sum(Nyx[:,w]) + C * 0.01)
            hmmPw[s,w] = weight * Pz[s, int(conZ[s])-1] * (Nwz[w,int(conZ[s])-1] + 0.01) / ( sum( Nwz[:,int(conZ[s])-1]) + W * 0.01)

    return hmmPw


def hmm_PW(recogdir, learndir, conName, nyxfile):

    if conName == "Object":
        concept = 1
        foldername = learndir + "/000"
        recogfolger = recogdir + "/000"
        filename = foldername + "/Nmwz001.txt"
    elif conName == "Motion":
        concept = 2
        foldername = learndir + "/001"
        recogfolger = recogdir + "/001"
        filename = foldername + "/Nmwz001.txt"
    elif conName == "Place":
        concept = 3
        foldername = learndir + "/002"
        recogfolger = recogdir + "/002"
        filename = foldername + "/Nmwz001.txt"
#    elif conName == "Person":
#        concept = 4
#        foldername = learndir + "/003"
#        recogfolger = recogdir + "/003"
#        filename = foldername + "/Nmwz002.txt"


    Pz = numpy.loadtxt(recogfolger + "/theta.txt")
    conZ = numpy.loadtxt( recogfolger + "/ClassResult.txt" )

    Nyx = numpy.loadtxt(nyxfile)
    Nwz = numpy.loadtxt( filename )
    hmmPw = CalcHmmPw( conZ, Pz, Nwz, Nyx, concept)

    return hmmPw

#助詞クラスの単語予測
def make_particle_file(pwfile, nyxfile, conNum):
    wPw = numpy.loadtxt(pwfile)
    Nyx = numpy.loadtxt(nyxfile)

    S, W = wPw.shape
    C, W = Nyx.shape
    W = W-2

    wPw_dw = numpy.zeros( (S, W) )

    for w in range(W):
        count = []
        for c in range(C):
            count.append(Nyx[c][w])

        wPw_dw[0][w] = (float(count[conNum + 1]) + 0.01)/(sum(count)+0.01*W)

    for s in range(1, S ):
        for w in range( W ):
            wPw_dw[s][w] = wPw_dw[0][w]


    return wPw_dw



def main(threshold, n_mwz_list):
    print "Start ... \n"
    #modelName = "recogModel/model"
    # 予測した単語の発生頻度の読込用
    #Filenames = [ modelName + "/000/Nmwz001.txt",       # 物体概念
    #              modelName + "/001/Nmwz001.txt",       # 動作概念
    #              modelName + "/002/Nmwz001.txt"       # 報酬概念
    #		]

    foldername = "WordsInfer"
    MakeDir( foldername )

    # 相互情報量のみの概念選択
    MIwzs = []

    # 重み付けされた単語予測からの概念選択
    weightedPws = []

    for i in range( len(n_mwz_list) ):
        MIwz, Pwz, Pw, wPw = CalcMIWithInferredWords( n_mwz_list[i] )
        weightedPws.append( wPw )
        MIwzs.append( copy.deepcopy( MIwz ) )

        numpy.savetxt( foldername + "/" + "%03dMIwz.txt" % i, MIwz, fmt="%0.10f", delimiter="\t" )
        numpy.savetxt( foldername + "/" + "%03dPwz.txt"  % i, Pwz , fmt="%0.10f", delimiter="\t" )
        numpy.savetxt( foldername + "/" + "%03dwPw.txt"  % i, wPw , fmt="%0.10f", delimiter="\t" )
        numpy.savetxt( foldername + "/" + "%03dPw.txt"   % i, Pw  , fmt="%0.10f", delimiter="\t" )

    # 重み付けされた単語確率の結果
    maxConcepts, maxScores, sortedConcepts, sortedScores = FindMaxConcepts( weightedPws )
    numpy.savetxt( foldername + "/" + "maxConcepts.txt"   , maxConcepts   , fmt="%d"    , delimiter="\t" )
    numpy.savetxt( foldername + "/" + "maxScores.txt"     , maxScores     , fmt="%0.10g", delimiter="\t" )
    numpy.savetxt( foldername + "/" + "sortedConcepts.txt", sortedConcepts, fmt="%d"    , delimiter="\t" )
    numpy.savetxt( foldername + "/" + "sortedScores.txt"  , sortedScores  , fmt="%0.10g", delimiter="\t" )

    # 相互情報量のみの結果
    maxConcepts, maxScores, sortedConcepts, sortedScores, maxScorePerConcepts = CalcMaxConceptOfWords( MIwzs, threshold )
    numpy.savetxt( foldername + "/" + "maxConcepts_mi.txt"   , maxConcepts   , fmt="%d"    , delimiter="\t" )
    numpy.savetxt( foldername + "/" + "maxScores_mi.txt"     , maxScores     , fmt="%0.10g", delimiter="\t" )
    numpy.savetxt( foldername + "/" + "sortedConcepts_mi.txt", sortedConcepts, fmt="%d"    , delimiter="\t" )
    numpy.savetxt( foldername + "/" + "sortedScores_mi.txt"  , sortedScores  , fmt="%0.10g", delimiter="\t" )
    numpy.savetxt( foldername + "/" + "maxScorePerConcepts.txt"  , maxScorePerConcepts  , fmt="%0.10g", delimiter="\t" )

    CreateMapConcepts("LearnData/wordlist.txt", "WordsInfer/maxConcepts_mi.txt", "WordsInfer/sortedScores_mi.txt", "WordsInfer/sortedConcepts_mi.txt", threshold )

    print "Finish ... \n"
