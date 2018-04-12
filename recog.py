# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import bhmm
import WordsInference
import WordHist
import serket as srk
import mlda
from mlda import mlda as mlda_p

def Recog(numOfAllConc,model_num):

    W_ALL = np.loadtxt("RecogData/recog_words.txt")
    V_O = np.loadtxt("LearnData/vision.Object")
    M_M = np.loadtxt("LearnData/motion.Motion")
    P_P = np.loadtxt("LearnData/place.Place")

    V_O = np.zeros([W_ALL.shape[0], V_O.shape[1]])
    M_M = np.zeros([W_ALL.shape[0], M_M.shape[1]])
    P_P = np.zeros([W_ALL.shape[0], P_P.shape[1]])

    W_O, W_M, W_P = WordHist.makeHist(
        W_ALL, numOfAllConc, first=False, Nyx_path="result_{0}/Nyx.txt".format(model_num)
        )


    obs1 = srk.Observation( V_O )     # 物体情報
    obs2 = srk.Observation( W_O )     # 物体単語
    obs3 = srk.Observation( M_M )     # 動作情報
    obs4 = srk.Observation( W_M )     # 動作単語
    obs5 = srk.Observation( P_P )     # 場所情報
    obs6 = srk.Observation( W_P )     # 場所単語


    w = 50
    mlda1 = mlda.MLDA(5, [w, w])#, category=object_category)
    mlda2 = mlda.MLDA(5, [w, w])#, category=motion_category)
    mlda3 = mlda.MLDA(4, [w, w])#, category=place_category)
    mlda_top = mlda.MLDA(6, [w, w, w])

    mlda1.connect( obs1, obs2 )
    mlda2.connect( obs3, obs4 )
    mlda3.connect( obs5, obs6 )
    mlda_top.connect( mlda1, mlda2, mlda3 )

    for it in range(5):
        print it
        mlda1.update("result_{0}/module026_mlda".format(model_num) )
        mlda2.update("result_{0}/module027_mlda".format(model_num) )
        mlda3.update("result_{0}/module028_mlda".format(model_num) )
        mlda_top.update("result_{0}/module029_mlda".format(model_num) )

if __name__ == '__main__':
    Recog(numOfAllConc=5,model_num=0)
