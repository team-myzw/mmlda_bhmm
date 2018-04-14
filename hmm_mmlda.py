#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import glob
from pomegranate import *
import pomegranate
import json
pomegranate.utils.disable_gpu()

import sys
sys.path.append("/home/robocup/Desktop/crest/.")
from mlda import mlda as mlda_p

OBJECT_DICT = {0:"button",1:"table",2:"doll",3:"fruit",4:"drink"}
PLACE_DICT = {0:"bed_room",1:"kitchen",2:"bath_room",3:"door"}
ACTION_CLASS = [2,3,3,3,3]
MOTION_CLASS = 5 # gp-class + 2(start,end)
WORDLIST = np.loadtxt("LearnData/wordlist.txt",dtype=str)

multi_n = 100

class HMM_MLDA(object):
    """
    mmldaをhmmによって時間拡張したモデル
    hmmはpomegranateで実装
    """

    def __init__(self,):
        self.model = None
        self.object_matrix = None
        self.motion_matrix = None
        self.place_matrix = None
        self.place_matrix_wprd = None

    def sampling_multi(self,theta):
        initial_value = {'dim'+str(i):prob for i,prob in enumerate(theta)}
        tmp_discrite = DiscreteDistribution(initial_value)
        tmp_independent = IndependentComponentsDistribution([tmp_discrite for i in  range(multi_n)])
        while (True):
            sample = tmp_independent.sample(1)[0].tolist()
            if sum(np.isin(sample,None)) == 0:
                break
        return sample

    def get_theta_from_dist(self,distribution):
        """
        多項分布のpomegranateの分布のパラメータを取得する

        return: theta
        """
        param = distribution.parameters[0]
        theta = [param['dim'+str(i)] for i in range(len(param))]
        return theta

    def calc_prob(self,Nwz):
        """
        NwzからPwzを計算
        """
        alpha = 0.1
        W, Z = Nwz.shape
        Pwz  = np.zeros( (W, Z) )
        for w in range( W ):
            for z in range( Z ):
                Pwz[ w, z ] = float( Nwz[w, z] + alpha ) / ( np.sum( Nwz[:, z] ) + W * alpha )
        return np.array(Pwz)

    def load_mmlda_param(self,param_dir="result_0"):
        """
        パラメータのload
        """
        # n_mwz のload
        object_n_mwz, _ = mlda_p.load_model(os.path.join(param_dir, "module006_mlda/"))
        motion_n_mwz, _ = mlda_p.load_model(os.path.join(param_dir, "module007_mlda/"))
        place_n_mwz, _ = mlda_p.load_model(os.path.join(param_dir, "module008_mlda/"))
        top_n_mwz, _ = mlda_p.load_model(os.path.join(param_dir, "module009_mlda/"))

        self.object_matrix = np.matrix(self.calc_prob(top_n_mwz[0].T).T) * np.matrix(self.calc_prob(object_n_mwz[0].T).T)
        self.motion_matrix = np.matrix(self.calc_prob(top_n_mwz[1].T).T) * np.matrix(self.calc_prob(motion_n_mwz[0].T).T)
        self.place_matrix = np.matrix(self.calc_prob(top_n_mwz[2].T).T) * np.matrix(self.calc_prob(place_n_mwz[0].T).T)
        self.place_matrix_wprd = np.matrix(self.calc_prob(top_n_mwz[2].T).T) * np.matrix(self.calc_prob(place_n_mwz[1].T).T)

    def set_hmm_data(self,data_path="result_0/module009_mlda/Pdz.txt",meta_data_paht="LearnData/action_nums.txt"):
        """
        hmmの学習データを入力
        """
        all_theta = np.loadtxt(data_path)
        action_num_lists = np.loadtxt(meta_data_paht,dtype=np.int32)
        seqs = []
        for i in range(len(action_num_lists)):
            tmp = all_theta[sum(action_num_lists[:i]):sum(action_num_lists[:i+1])]
            seqs.append(tmp)
        n_components = all_theta.shape[1]

        # seqsに基づいてmulti_n回サンプリング
        multi_n = 100
        # seqsの各Zのサンプリング N=100
        multi_sample_seqs = []
        for seq in seqs:
            multi_sample_seq = []
            for s in seq:
                initial_value = {'dim'+str(i):prob for i,prob in enumerate(s)}
                tmp_discrite = DiscreteDistribution(initial_value)
                tmp_independent = IndependentComponentsDistribution([tmp_discrite for i in  range(multi_n)])
                while (True):
                    sample = tmp_independent.sample(1)[0].tolist()
                    if sum(np.isin(sample,None)) == 0:
                        break
                multi_sample_seq.append(sample)
            multi_sample_seqs.append(multi_sample_seq)
        return multi_sample_seqs

    def set_hmm(self,n_components):
        """
        hmmの初期値をセット
        """
        transmat = np.abs(np.random.randn(n_components, n_components))
        transmat = (transmat.T / transmat.sum( axis=1 )).T
        start_probs = np.abs( np.random.randn(n_components) )
        start_probs /= start_probs.sum()
        initial_value = {'dim'+str(i):1.0 / n_components for i in range(n_components)}
        states = []
        for s in range(n_components):
            tmp_discrite = DiscreteDistribution(initial_value)
            tmp_independent = IndependentComponentsDistribution([tmp_discrite for i in  range(multi_n)])
            states.append(tmp_independent)
        self.model = HiddenMarkovModel.from_matrix(transmat, states, start_probs, merge='None')

    def fit_hmm(self,multi_sample_seqs):
        self.model.fit(multi_sample_seqs, verbose=True, n_jobs=8, algorithm = "baum-welch")

    def calc_recog_data(self,sentence):
        """
        mmlda_bhmmによって文章情報のみから他の情報を推定
        """
        # sentence2bow(sentence)
        # recog()
        pz = np.loadtxt("RecogData/module009_mlda/Pdz.txt")[0] #キッチンの机に置く
        return pz

    def viterbi(self,end_state,trans_num):
        """
        ビタビアルゴリズムによってプランニングを行う
        """
        nan_list = [np.nan for i in range(multi_n)]
        z_path = [nan_list] * trans_num
        z_path.append(end_state)
        tmp_logp, path = self.model.viterbi(z_path)
        print tmp_logp
        return path

    def load_hmm_model(self,model_path="./hmm_mmlda.json"):
        with open(model_path,"r") as f:
            tmp = json.load(f)
            self.model = HiddenMarkovModel.from_json(tmp)

    def dump_hmm_model(self,model_path="./hmm_mmlda.json"):
        with open(model_path,"w") as f:
            json.dump(self.model.to_json(),f)

    def motion_prob2bow(self,motion_prob,object_id):
        """
        motionのbowを計算
        """
        motion_bow = motion_prob[MOTION_CLASS * object_id : MOTION_CLASS * (object_id+1)]
        start_index = MOTION_CLASS*(object_id +1) -1
        end_index = MOTION_CLASS*(object_id +1) -2
        normaliz = (motion_prob[start_index] + motion_prob[end_index]) / 2.0
        motion_bow = motion_bow / normaliz
        return np.around(motion_bow,0)

if __name__ == '__main__':
    hmm_mmlda = HMM_MLDA()

    # learning
    # multi_sample_seqs = hmm_mmlda.set_hmm_data()
    # all_theta = np.loadtxt("result_0/module009_mlda/Pdz.txt")
    # n_components = all_theta.shape[1]
    # hmm_mmlda.set_hmm(n_components)
    # hmm_mmlda.fit_hmm(multi_sample_seqs)
    # hmm_mmlda.dump_hmm_model(model_path="./hmm_mmlda.json")

    hmm_mmlda.load_hmm_model(model_path="./hmm_mmlda.json")
    # recognition
    pz = hmm_mmlda.calc_recog_data("キッチンの机に置く")
    hmm_mmlda.load_mmlda_param()
    # print pz
    goal_state = hmm_mmlda.sampling_multi(pz)
    num_trance = 2
    path = hmm_mmlda.viterbi(goal_state,num_trance)


    for state_step in range(1,num_trance+2):
        z_theta = hmm_mmlda.get_theta_from_dist(path[state_step][1].distribution.distributions[0])
        object_w = np.matrix(z_theta) * hmm_mmlda.object_matrix
        motion_w = np.matrix(z_theta) * hmm_mmlda.motion_matrix
        place_w_word = np.matrix(z_theta) * hmm_mmlda.place_matrix_wprd
        object_id = np.argmax(object_w)
        print ""
        print "state_step: ", state_step
        print "object_id: ", object_id, OBJECT_DICT[object_id], object_w.max()
        motion_bow = hmm_mmlda.motion_prob2bow(np.array(motion_w)[0],object_id)
        print "moiton bow: ", motion_bow[:ACTION_CLASS[object_id]]
        print "place: ", WORDLIST[np.argmax(place_w_word)]

        # place_name_id = np.argsort(np.array(place_w_word))[0][::-1][:5]
        # print place_name_id
        # for index in place_name_id:
        #     print WORDLIST[index]

        motion_w = np.matrix(z_theta) * hmm_mmlda.place_matrix
