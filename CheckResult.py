# -*- coding: utf-8 -*-
#!/usr/bin/env python

import codecs
import numpy

#分類精度の結果をまとめる
def accCheck(numfile):
    acclist = numpy.zeros( (numfile, 8) )
    for i in range(1, numfile+1):
        learn_objects = codecs.open( "Results/" + str(i).zfill(3) + "/learnModel/model/ObjectConfMat.txt", "r").readlines()
        recog_objects = codecs.open( "Results/" + str(i).zfill(3) + "/recogModel/model/ObjectConfMat.txt", "r").readlines()
        learn_motions = codecs.open( "Results/" + str(i).zfill(3) + "/learnModel/model/MotionConfMat.txt", "r").readlines()
        recog_motions = codecs.open( "Results/" + str(i).zfill(3) + "/recogModel/model/MotionConfMat.txt", "r").readlines()
        learn_rewards = codecs.open( "Results/" + str(i).zfill(3) + "/learnModel/model/RewardConfMat.txt", "r").readlines()
        recog_rewards = codecs.open( "Results/" + str(i).zfill(3) + "/recogModel/model/RewardConfMat.txt", "r").readlines()


        learn_object = learn_objects[0].replace("\n","").split("\t")
        recog_object = recog_objects[0].replace("\n","").split("\t")
        learn_motion = learn_motions[0].replace("\n","").split("\t")
        recog_motion = recog_motions[0].replace("\n","").split("\t")
        learn_reward = learn_rewards[0].replace("\n","").split("\t")
        recog_reward = recog_rewards[0].replace("\n","").split("\t")


        acclist[i-1][0] = float(learn_object[1])
        acclist[i-1][1] = float(recog_object[1])
        acclist[i-1][2] = float(learn_motion[1])
        acclist[i-1][3] = float(recog_motion[1])
        acclist[i-1][4] = float(learn_reward[1])
        acclist[i-1][5] = float(recog_reward[1])


    numpy.savetxt( "Results/accResult.txt", acclist, fmt="%.4f", delimiter="\t" )

#単語数の確認
def numwordCheck(numfile):
    numwords = []
    for i in range(1,numfile+1):
        wordlist = codecs.open("Results/" + str(i).zfill(3) +"/result/wordlist.txt", "r").readlines()
        numwords.append(len(wordlist))

    f = codecs.open("Results/numWords.txt", "w")
    for i in numwords:
        f.write(str(i)+"\n")


def select_concept(wordsListFilename, nyxfile, foldername):
    words = codecs.open( wordsListFilename, "r" ).readlines()
    Nyx = numpy.loadtxt(nyxfile)
    fw = codecs.open( foldername + "/mapConcept.txt", "w" )

    C, W = Nyx.shape

    for w in range(W-2): #BHMMの時はW-2にする
        index = numpy.argmax(Nyx[:,w])
        fw.write("%s\t%d\n" % (words[w].strip(), index))

    fw.close()

def conceptCheck(numfile, numConc):
    Conc = codecs.open("correctConcept.txt","r").readlines()
    fw = codecs.open("Results/accConc.txt","w")
    for i in range(1,numfile+1):
        foldername ="Results/"+str(i).zfill(3)
        wordlist = foldername + "/result/wordlist.txt"
        Nyx = foldername + "/result/Nyx.txt"
        select_concept(wordlist, Nyx, foldername)

        f = codecs.open(foldername + "/conceptCheck.txt", "w")
        lines = codecs.open(foldername + "/mapConcept.txt", "r").readlines()

        sum_conc = 0

        for j, line in enumerate(lines):
            line = line.split("\t")
            data = line[1].replace("\n","")
            conc = Conc[j].replace("\n","")
            if int(data) >= numConc + 1:
                if int(conc) == numConc + 1:
                    f.write("1\n")
                    sum_conc += 1
                else:
                    f.write("0\n")
            else:
                if int(data) == int(conc):
                    f.write("1\n")
                    sum_conc += 1
                else:
                    f.write("0\n")

        acc = float(sum_conc)/len(lines)
        f.write(str(acc) + "\n")
        fw.write(str(acc) + "\n")
