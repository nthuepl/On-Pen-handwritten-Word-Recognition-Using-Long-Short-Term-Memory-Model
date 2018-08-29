import pymongo
import os
import pickle
import json

# cnt = [0 for i in range(26)]
# for fn in os.listdir(os.getcwd()+'/train/'):
#     if ".json" in fn:
#         word = fn.split('.')[0]
#         for char in word:
#             cnt[ord(char)-ord('a')] += 1
# for i in range(26):
#     print(chr(i+ord('a')), cnt[i])


# for fn in os.listdir(os.getcwd()+'/train/'):
#     if ".json" in fn:
#         os.system("python preprocess.py train/"+fn)
#         # os.system("mv "+fn+" "+"train/"+fn)

# for fn in os.listdir(os.getcwd()):
#     if ".json" in fn:
#         os.system("python preclass.py %s" %(fn))
#         os.system("mv %s train/%s" % (fn, fn))

# try:
#     client = pymongo.MongoClient("localhost", 27017)
# except Exception as e:
#     raise
# namespace = [chr(ord('a')+i) for i in range(26)]
# db = client.train_set
# count = 0
# for label in namespace:
#     collection = db[label]
#     print(label, collection.count())
#     count += collection.count()

# print("total "+str(count)+" tuple of data")

# import matplotlib.pyplot as plt
# import numpy as np
# # namespace = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
# namespace = ['gy']
# colorspace = ['b', 'g', 'r', 'c', 'm', 'y']
# db = client.train
# collection = db['g']
# cursor = collection.find({})
# for data in cursor:
#     for i, domain in enumerate(namespace):
#         plt.plot(data[domain], colorspace[i])
        
# plt.show()

# try:
#     client = pymongo.MongoClient("localhost", 27017)
# except Exception as e:
#     raise
# pickleObject = {}
# namespace = [chr(ord('a')+i) for i in range(26)]
# db = client.train_set

# for label in namespace:
#     pickleObject[label] = []

# for label in namespace:
#     collection = db[label]
#     cursor = collection.find({})
#     for data in cursor:
#         pickleObject[label].append(data)
# pickle.dump(pickleObject, open('train_set.dat', 'wb'))

# train_set = pickle.load(open('train_set.dat', 'r'))
# cnt = 0
# for key, values in train_set.items():
#     if len(values) > 0:
#         cnt += 1
# print(cnt)

# import matplotlib.pyplot as plt
# import numpy as np
# fp = open('experiment/record11', 'r')
# curve = []
# for line in fp.readlines():
#     try:
#         curve.append(float(line[:-1]))
#     except Exception as e:
#         pass
# plt.plot(curve)
# plt.show()
# fp.close()

# cnt = 0
# path = os.getcwd()+'/test_set/raw/'
# for fn in os.listdir(path):
#     if ".json" in fn:
#         cnt += 1
#         os.system("python preprocess.py test_set/raw/%s" % (fn))
#         # print("(%d/%d)" % (cnt, len(os.listdir(path))))

# try:
#     client = pymongo.MongoClient("localhost", 27017)
# except Exception as e:
#     raise
# # classspace = ['au', 'bp', 'c', 'd', 'e', 'f', 'g', 'hmn', 'it', 'j', 'k', 'l', 'o', 'q', 'r', 's', 'v', 'w', 'x', 'y', 'z']
# classspace = [chr(ord('a')+i) for i in range(26)]
# db = client.test_set
# count = 0
# for label in classspace:
#     collection = db[label]
#     print(label, collection.count())
#     count += collection.count()
# print("total "+str(count)+" tuple of data")

# obj = {}
# for label in classspace:
#     obj[label] = []

# for label in classspace:
#     collection = db[label]
#     cursor = collection.find({})
#     for data in cursor:
#         input_data = {}
#         for key in data.keys():
#             if "id" not in key:
#                 input_data[key] = data[key]
#         obj[label].append(input_data)
# output_filename = 'test_set.dat'
# pickle.dump(obj, open(output_filename, 'wb'))
# print("data is record in %s" % output_filename)

# def levenshtein(seq1, seq2):  
#     size_x = len(seq1) + 1
#     size_y = len(seq2) + 1
#     matrix = [[0 for i in range(size_y)] for j in range(size_x)]
#     for x in range(size_x):
#         matrix[x][0] = x
#     for y in range(size_y):
#         matrix[0][y] = y

#     for x in range(1, size_x):
#         for y in range(1, size_y):
#             if seq1[x-1] == seq2[y-1]:
#                 matrix[x][y] = min(
#                     matrix[x-1][y] + 1,
#                     matrix[x-1][y-1],
#                     matrix[x][y-1] + 1
#                 )
#             else:
#                 matrix[x][y] = min(
#                     matrix[x-1][y] + 1,
#                     matrix[x-1][y-1] + 1,
#                     matrix[x][y-1] + 1
#                 )
#     for x in range(1, size_x):
#         for y in range(1, size_y):
#             if x != y:
#                 matrix[x][y] += abs(x-y)
#     return (matrix[-1][-1])

# str1 = 'abcf'
# str2 = 'aedf'
# print(levenshtein(str1, str2))

# import random
# au_w, bp_w, ce_w, fl_w, hn_w, rv_w = (random.random(), random.random(), random.random(), random.random(), random.random(), random.random())
# sub_w, del_w, ins_w = (1+random.random(), 1+random.random(), 1+random.random())

# jsonObj = {}
# jsonObj["sub"] = sub_w
# jsonObj["del"] = del_w
# jsonObj["ins"] = ins_w
# jsonObj["au"] = au_w
# jsonObj["bp"] = bp_w
# jsonObj["ce"] = ce_w
# jsonObj["fl"] = fl_w
# jsonObj["hn"] = hn_w
# jsonObj["rv"] = rv_w

# filename = 'param.json'
# json_str = json.dumps(jsonObj)
# output = open(filename, 'w')
# output.write(json_str)
# output.close()

# from QTGraph import QTGraph
# acceleration = [1 for i in range(500)]
# gyroscope = [1 for i in range(500)]
# graph = QTGraph([acceleration, gyroscope], ['acc', 'gyro'], sourcesNum=2, x_range=500)
# graph.show()