from __future__ import division, print_function, absolute_import
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile
import numpy as np
import os
import tensorflow as tf
import pickle
import argparse
from preprocess import preprocess
from Corrector import Lexicon, Corrector
from Levenshtein import levenshtein, index, easy_levenshtein
import progressbar
import random
import time
import socket

# labelspace = ['au', 'bp', 'c', 'd', 'e', 'f', 'g', 'hmn', 'it', 'j', 'k', 'l', 'o', 'q', 'r', 's', 'v', 'w', 'x', 'y', 'z']
labelspace = [chr(ord('a')+i) for i in range(26)]
axiss = ['ax', 'ay', 'rax', 'ray', 'gx', 'gy', 'gz', 'rgx', 'rgy', 'rgz', 'gs', 'rgs']
        

# additional function # # # # # # # # #
def labelEncoder(char):
    r = [0 for c in labelspace]
    r[labelspace.index(char)] = 1
    return r
    
def labelDecoder(arr):
    return arr.index(max(arr))

def loadFile(filename):
    X_test = []
    word_data = preprocess(filename)
    for data in word_data:
        img = []
        width = len(data['ax'])
        for i in range(width):
            img.append([data[axis][i] for axis in axiss])
        X_test.append(img)

    num_steps = 100
    num_inputs = len(axiss)
    X_test = np.asarray(X_test)
    X_test = X_test.reshape([-1, num_steps, num_inputs])
    return X_test


class LSTMsocket():
    """docstring for LSTMsocket"""
    def __init__(self):

        frozen_graph="./all_alpha.pb"
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
            
        # construct a graph object that hold test.pb.  
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                restored_graph_def,
                input_map=None,
                return_elements=None,
                name="")

        x=graph.get_tensor_by_name("x:0")
        #y=graph.get_tensor_by_name("labels:0")
        pred =graph.get_tensor_by_name("softmax_layer/prediction:0")
        weight = graph.get_tensor_by_name("init_weights/weights_in:0")

        lex = Lexicon()
        crtr = Corrector(lexicon=lex)
        # for fn in os.listdir(os.getcwd()+'/../penwriting/validate/'):
        #     if ".json" in fn:
        #         try:
        #             data = loadFile('../penwriting/validate/'+fn)
        #             dataset.append(data)
        #             groundtruth.append(fn.split('.')[0])
        #         except Exception as e:
        #             continue


        # socket setup
        HOST = '127.0.0.1'
        PORT = 8001

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

        print('LSTM classifier start at: %s:%s' %(HOST, PORT))
        print('wait for input...')
        # socket setup

        with tf.Session(graph=graph) as sess:

            while True:
                conn, addr = self.s.accept()
                print('Connected by', addr)

                data = conn.recv(1024)
                print(type(data))
                print(data.decode('utf-8'))

                filename = data.decode('utf-8')
                X_test = loadFile(filename)
                results = sess.run(pred, feed_dict={x:X_test})
                word = ''
                for i, result in enumerate(results):
                    pred_label = labelDecoder(result.tolist())
                    label = labelspace[pred_label]
                    word += label

                word = crtr.correction(word)    

                conn.send(word.encode('utf-8'))
                

    def __exit__(self, exc_type, exc_value, traceback):
        self.s.close()

if __name__ == '__main__':

    socket = LSTMsocket()