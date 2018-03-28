# -*- coding: utf-8 -*-
#!/usr/bin/env python

from collections import Counter
import codecs
import os
import re
import myio
import glob
import numpy
import copy

def mkdir( dir ):
    try:
        os.mkdir( dir )
    except:
        pass


# 単語辞書を作成
def MakeDic( f ):
    dic = []
    words = codecs.open( f , "r").read().split()
    for w in words:
        if not w in dic:
            dic.append(w)
    return dic


def makeWordInfo( numOfSen ):

    # 辞書作成
    wordDic = MakeDic( "sentences.txt" )

    # 辞書保存
    myio.SaveArray( wordDic , "result/wordlist.txt" )

    #辞書の読み込み
    dictionary = LoadDict( "result/wordlist.txt" )
    #学習用のヒストグラム作成
    histo = MakeWordsHist( "sentences.txt", "senIndex.txt",  numOfSen, dictionary )

    #認識の用の空のヒストグラム作成
    indexes = codecs.open("senIndex.txt", "r").readlines()
    # N = len(indexes)
    c = Counter(indexes)
    N = len(c.most_common()) * 4 # TODO 後で修正
    hist_null = numpy.zeros( (N, len(dictionary)))

    numpy.savetxt( "result/words.All", histo, delimiter="\t", fmt="%.0f" )
    numpy.savetxt( "RecogData/words.All", hist_null, delimiter="\t", fmt="%.0f" )

def LoadDict( filename ):
    dictionary = {}
    lines = codecs.open( filename, "r" ).readlines()
    for i, l in enumerate( lines ):
        dictionary[l.strip()] = i
    return dictionary


#単語情報（ヒストグラム）の作成
def MakeWordsHist( filename, indexfile, step, dictionary ):
    #教示文の読み込み
    lines = codecs.open( filename, "r").readlines()
    #データに対応した教示文のインデックスのファイルを読み込む
    indexes = codecs.open( indexfile, "r").readlines()

    #観測したシーンの数
    # N = len(indexes)
    c = Counter(indexes)
    N = len(c.most_common())*4 #TODO 後で修正
    #単語数
    D = len( dictionary )
    print N, D

    h = numpy.zeros( (N, D) )
    #ヒストグラムを作成
    cnt = 0
    old_index = 0
    for i, index in enumerate(indexes):
        if old_index != int(index):
            old_index = int(index)
            cnt += 1
            # print cnt
            # print i
        l = lines[i].split()
        for w in l:
            w_index = dictionary[w]
            h[cnt, w_index] += 1
    return h
