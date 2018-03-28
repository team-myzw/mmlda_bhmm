# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy
import glob
import matplotlib.pyplot as plt
import pylab as pl
import codecs

def graph_acc( concName ):

    accfile = codecs.open("Results/accResult.txt","r","sjis").readlines()

    t = []
    learn = []
    recog = []

    count = 1

    for accs in accfile:
        acc = accs.replace("\n","").split("\t")
        t.append( count )
        if concName == "object":
            learn.append(float(acc[0]))
            recog.append(float(acc[1]))
        elif concName == "motion":
            learn.append(float(acc[2]))
            recog.append(float(acc[3]))
        elif concName == "reward":
            learn.append(float(acc[4]))
            recog.append(float(acc[5]))
        elif concName == "person":
            learn.append(float(acc[6]))
            recog.append(float(acc[7]))
        else:
            print("error:not concename")
            return
        count += 1

    pl.plot(t,learn,"r-o", label = "learn_"+concName )
    pl.plot(t,recog,"r-*", label = "recog_"+concName )


    #ラベルの追加
    plt.legend(loc='upper right', bbox_to_anchor=(1.48, 1.0))# 凡例を表示
    plt.gcf().subplots_adjust(right=0.7)
    plt.ylim(0.4,1.1)
    pl.xlabel('Repeat count') # X 軸
    pl.ylabel('Acc') # Y 軸
    axes = pl.gca()
    axes.grid(True)
    figname = "Results/Acc_" + concName + ".png"
    pl.savefig(figname)
    pl.clf()

def main():

    print "Start\n"
    graph_acc("object")
    graph_acc("motion")
    graph_acc("reward")
    # graph_acc("person")
    print"Finish\n"
