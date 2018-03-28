# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy as np

from word_client import WordClient

word_client = WordClient()


def segmentate(sentence):
    word_dir = "RecogData/words/"
    try:
        os.mkdir(word_dir)
    except:
        pass

    word_client.load_codebook("LearnData/wordlist.txt")
    word_client.update(word_dir)
    word_client.setAccept()

    word_client.sentence_data_cb(sentence) # 入力文字列
    word_client.setReject()
    word_client.split_sentences(word_dir)
    word_client.sumalize_utter(word_dir)
    sentences_file_path = os.path.join(word_dir,"sentences.txt")
    word_client.update_word_codebook(sentences_file_path)
    # 単語列のBoW化
    word_client.words2histogram(
        os.path.join(word_dir,"sentences.txt"),
        os.path.join(word_dir,"words.All") # BoW
    )
    #word_client.dump_codebook(os.path.join(word_dir,"codebook.txt"))
