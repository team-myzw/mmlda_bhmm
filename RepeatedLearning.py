# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import shutil
import os
import time
import bhmm
import wordInfoGen
import WordsInference
import WordHist
import CheckResult
import Fig_Data
import SentenceGenerator
import serket as srk
import mlda
from mlda import mlda as mlda_p
#from segmentate import segmentate

def mkdir( dir ):
    try:
        os.mkdir( dir )
    except:
        pass

def remove( filename ):
    try:
        os.remove( filename )
    except:
        pass

def move( filename ):
    try:
        shutil.move( filename )
    except:
        pass

def rmtree( filename ):
    try:
        shutil.rmtree( filename )
    except:
        pass

def Learn_1st( numOfRepeat, numOfAllConc, numOfConc, numOfSen, threshold):
    #単語情報の作成
    # wordInfoGen.makeWordInfo( numOfSen )

    #ヒストグラムをmMLDAの学習用フォルダにコピーする
    shutil.copy("LearnData/words.All", "result/words.All")
    # remove("LearnData/Words.All")
    # shutil.copy("result/words.All", "LearnData/words.All")
    shutil.copy("LearnData/words.All", "LearnData/word.Object")
    shutil.copy("LearnData/words.All", "LearnData/word.Motion")
    shutil.copy("LearnData/words.All", "LearnData/word.Reward")
    # shutil.copytree("LearnData","RecogData")

    ###################        mMLDAの学習&認識            ##############################
    os.system( "./mmlda -learn LearnData LearnParams.txt learnModel" )
    os.system( "./mmlda -recog RecogData RecogParams.txt learnModel recogModel" )
    ###########################################################################################

    #相互情報量の計算
    WordsInference.main(threshold)
    #文法の学習
    bhmm.main(numOfConc, numOfAllConc)

    #ここまでの結果を移す
    # shutil.copytree("result", "Results/" + str(i).zfill(3) + "/result")
    # shutil.copytree("learnModel", "Results/" + str(i).zfill(3) + "/learnModel")
    # shutil.copytree("recogModel", "Results/" + str(i).zfill(3) + "/recogModel")
    # shutil.copytree("WordsInfer", "Results/" + str(i).zfill(3) + "/WordsInfer")

def Learn( numOfRepeat, numOfAllConc, numOfConc, numOfSen, threshold):

    V_O = np.loadtxt("LearnData/vision.Object")
    M_M = np.loadtxt("LearnData/motion.Motion")
    W_ALL = np.loadtxt("LearnData/words.All")
    
    for i in range(numOfRepeat):
        #各概念の単語情報を作成
        W_O, W_M = WordHist.makeHist(W_ALL, numOfAllConc, first=(i==0))

        ###################　　　　　mMLDAの学習&認識　　           ##############################
        # os.system( "./mmlda_fast -learn LearnData LearnParams.txt learnModel" )
        # os.system( "./mmlda -recog RecogData RecogParams.txt learnModel recogModel" )
        ###########################################################################################

        obs1 = srk.Observation( V_O )     # 物体情報
        obs2 = srk.Observation( W_O )     # 物体単語
        obs3 = srk.Observation( M_M )     # 動作情報
        obs4 = srk.Observation( W_M )     # 動作単語
        #obs5 = srk.Observation( np.loadtxt("LearnData/place.Place") )      # 場所情報
        #obs6 = srk.Observation( np.loadtxt("LearnData/word.Place") )       # 場所単語

        # 正解ラベルを入れておくと精度出してくれる
        object_category = np.loadtxt( "LearnData/CorrectObject.txt" )
        motion_category = np.loadtxt( "LearnData/CorrectMotion.txt" )
        
        #mlda1 = mlda.MLDA(10, [200,200,200], category=object_category)
        #mlda2 = mlda.MLDA(10, [200], category=motion_category)
        #mlda3 = mlda.MLDA(10, [100,100])

        w = 50
        mlda1 = mlda.MLDA(5, [w, w])#, category=object_category)
        mlda2 = mlda.MLDA(5, [w, w])#, category=motion_category)
        #mlda3 = mlda.MLDA(4, [w, w])
        mlda_top = mlda.MLDA(5, [w, w])
        
        mlda1.connect( obs1, obs2 )
        mlda2.connect( obs3, obs4 )
        #mlda3.connect( obs5, obs6 )
        mlda_top.connect( mlda1, mlda2 )
        #mlda_top.connect( mlda1, mlda2, mlda3 )
    
        for it in range(5):
            print it
            mlda1.update()
            mlda2.update()
            #mlda3.update()
            mlda_top.update()

        # 観測ノード数 + MLDA数
        module_num = 7
            
        n_mwz_obj, _ = mlda_p.load_model("module%03d_mlda" % (module_num * i + 4))
        n_mwz_mot, _ = mlda_p.load_model("module%03d_mlda" % (module_num * i + 5))
    
        #相互情報量の計算
        WordsInference.main(threshold, [n_mwz_obj[1].T, n_mwz_mot[1].T])
        #文法の学習
        bhmm.main(numOfConc, numOfAllConc)



def Sentence_Generator(numOfRepeat, numOfAllConc, numOfConc, numOfSamples, numOfWords, numOfSampleIndex):

    for i in range(1, numOfRepeat+1):

        foldername ="Results/" + str(i).zfill(3)
        mkdir(foldername + "/hmm_Pw")

        #各概念の単語予測
        hmmPw = WordsInference.hmm_PW( foldername + "/recogModel/model", foldername  + "/learnModel/model", "Object", foldername + "/result/Nyx.txt")
        np.savetxt( foldername + "/hmm_Pw/000.txt", hmmPw , fmt="%.10f", delimiter="\t" )
        hmmPw = WordsInference.hmm_PW( foldername + "/recogModel/model", foldername  + "/learnModel/model", "Motion", foldername + "/result/Nyx.txt")
        np.savetxt( foldername + "/hmm_Pw/001.txt", hmmPw , fmt="%.10f", delimiter="\t" )
        hmmPw = WordsInference.hmm_PW( foldername + "/recogModel/model", foldername  + "/learnModel/model", "Reward", foldername + "/result/Nyx.txt")
        np.savetxt( foldername + "/hmm_Pw/002.txt", hmmPw , fmt="%.10f", delimiter="\t" )
#        hmmPw = WordsInference.hmm_PW( foldername + "/recogModel/model", foldername  + "/learnModel/model", "Person", foldername + "/result/Nyx.txt")
#        np.savetxt( foldername + "/hmm_Pw/003.txt", hmmPw , fmt="%.10f", delimiter="\t" )

        #助詞クラスの単語予測
        for j in range(numOfAllConc-numOfConc):
            hmmPw = WordsInference.make_particle_file(foldername + "/hmm_Pw/000.txt", foldername + "/result/Nyx.txt", numOfConc+j)
            np.savetxt(foldername + "/hmm_Pw/" + str(numOfConc+j).zfill(3) + ".txt", hmmPw , fmt="%.10f", delimiter="\t")

        #文生成
        SentenceGenerator.main( foldername, numOfAllConc, numOfSamples, numOfWords, numOfSampleIndex)

"""
def Recog(words, numOfAllConc):

    # 文字列をBoWに
    segmentate()
    
    V_O = np.loadtxt("RecogData/vision.Object")
    M_M = np.loadtxt("RecogData/motion.Motion")
    W_ALL = np.loadtxt("RecogData/words.All")
    
    #各概念の単語情報を作成
    W_O, W_M = WordHist.makeHist(W_ALL, numOfAllConc)
    
    obs1 = srk.Observation( V_O )     # 物体情報
    obs2 = srk.Observation( W_O )     # 物体単語
    obs3 = srk.Observation( M_M )     # 動作情報
    obs4 = srk.Observation( W_M )     # 動作単語
    #obs5 = srk.Observation( np.loadtxt("RecogData/place.Place") )      # 場所情報
    #obs6 = srk.Observation( np.loadtxt("RecogData/word.Place") )       # 場所単語
    
    # 正解ラベルを入れておくと精度出してくれる
    object_category = np.loadtxt( "RecogData/CorrectObject.txt" )
    motion_category = np.loadtxt( "RecogData/CorrectMotion.txt" )
    
    #mlda1 = mlda.MLDA(10, [200,200,200], category=object_category)
    #mlda2 = mlda.MLDA(10, [200], category=motion_category)
    #mlda3 = mlda.MLDA(10, [100,100])
    
    w = 50
    mlda1 = mlda.MLDA(4, [w, w], category=object_category)
    mlda2 = mlda.MLDA(4, [w, w], category=motion_category)
    #mlda3 = mlda.MLDA(4, [w, w])
    mlda_top = mlda.MLDA(4, [w, w])
    
    mlda1.connect( obs1, obs2 )
    mlda2.connect( obs3, obs4 )
    #mlda3.connect( obs5, obs6 )
    mlda_top.connect( mlda1, mlda2 )
    #mlda_top.connect( mlda1, mlda2, mlda3 )
    
    for it in range(5):
        print it
        mlda1.update("module032_mlda")
        mlda2.update("module033_mlda")
        #mlda3.update()
        mlda_top.update("module034_mlda")
""" 
 


def main():
    #パラメータ
    ###############################################
    numOfRepeat = 5        #繰り返し学習する回数
    numOfAllConc = 4        #助詞を含んだ概念クラスの数
    numOfConc = 2           #助詞を除いた概念クラスの数
    numOfSen = 10           #1シーン当たりの教示文
    threshold = 0.008       #相互情報量の閾値
    numOfSamples = 10       #文生成の試行回数
    numOfWords = 10         #文生成時に使用する単語予測数
    numOfSampleIndex = 10   #概念をサンプリングする回数
    ################################################

    #学習
    Learn( numOfRepeat, numOfAllConc, numOfConc, numOfSen, threshold )
    #単語予測＆文生成
    """
    Sentence_Generator(numOfRepeat, numOfAllConc, numOfConc, numOfSamples, numOfWords, numOfSampleIndex)

    CheckResult.accCheck(numOfRepeat)
    CheckResult.conceptCheck(numOfRepeat, numOfConc)
    Fig_Data.main()
    """

if __name__ == '__main__':
    main()
