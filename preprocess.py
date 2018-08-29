import numpy as np
from scipy.signal import butter, lfilter, freqz, argrelextrema
import math
import json
import argparse
import pickle
import pymongo

# numpy parameters
order = 4
fs = 100.0
cutoff = 2.2

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def normalize(src, nor_len=200):

	# length normalization
    dest = [0 for i in range(nor_len)]
    org_len = len(src)
    for i in range(nor_len):
        expected_index = float(i)*org_len/nor_len
        real_index = i*org_len/nor_len
        dist = expected_index-real_index
        if real_index < len(src)-1:
            dest[i] = src[int(real_index)+1]*(dist) + src[int(real_index)]*(1-dist)
        else:
            dest[i] = src[int(real_index)]

    # value normalization
    MAX = max(dest)
    MIN = min(dest)
    for i, v in enumerate(dest):
    	dest[i] = (v-MIN)/(MAX-MIN)

    return dest

def toClass(char):
	# classspace = [chr(ord('a')+i) for i in range(26)]
	# classspace = ['au', 'bp', 'c', 'd', 'e', 'f', 'g', 'hmn', 'it', 'j', 'k', 'l', 'o', 'q', 'r', 's', 'v', 'w', 'x', 'y', 'z']
	classspace = [chr(ord('a')+i) for i in range(26)]
	# classspace = ['a', 'u']
	for Class in classspace:
		if char in Class:
			return Class
	return 'None'

def preprocess(filename, return_segmentation=False):

	fp = open(filename, "r")
	jsdata = json.loads(fp.read())
	fp.close()
	result = []

	# step 1 : segmentation # # # # # # #
	# # segment data according to total sum of gyroscope curve # # # # # #
	valid_curve_length = min( [ len(c) for c in [ jsdata["gx"], jsdata["gy"], jsdata["gz"] ] ] )

	# # # # low-pass filter gyroscope curve # #
	gyroForceCurve = [ math.sqrt( jsdata["gx"][i]**2 + jsdata["gy"][i]**2 + jsdata["gz"][i]**2 ) for i in range( valid_curve_length ) ]
	low_pass_gyroForceCurve = butter_lowpass_filter( gyroForceCurve, cutoff, fs, order )

	gf = list(gyroForceCurve)
	tmp_gf = list(gyroForceCurve)
	lpgf = butter_lowpass_filter( tmp_gf, cutoff, fs, 1 )

	# # # # find local minimum in the curve, as candidates of segmentation point # # # #
	local_minimums, = argrelextrema( np.array( low_pass_gyroForceCurve ), np.less )
	local_minimums = local_minimums.tolist()
	if low_pass_gyroForceCurve[0] < low_pass_gyroForceCurve[1]:
		local_minimums.insert(0, 0)
	if low_pass_gyroForceCurve[-1] < low_pass_gyroForceCurve[-2]:
		local_minimums.append(len(low_pass_gyroForceCurve)-1)

	# # # # judge segmentation candidates' point, segmentation finish # # # #
	segmentation = []
	threshold = ( max( low_pass_gyroForceCurve ) + min( low_pass_gyroForceCurve ) ) / 2
	entropy = max( low_pass_gyroForceCurve ) - min( low_pass_gyroForceCurve )
	threshold_coef = float(1)/3.5
	# print(local_minimums)

	window = []
	windows = []
	head = 0
	tail = 0
	for i, point in enumerate(local_minimums[:-1]):

		nextpoint = local_minimums[i+1]
		if not window:
			window = low_pass_gyroForceCurve[ point:nextpoint ].tolist()
			head = point
		tail = nextpoint

		MAX = max(window)
		MIN = min(window)
		rise = MAX - window[0]
		fall = MAX - window[-1]
		height = MAX - MIN

		# print("fall = %s, rise = %s, height = %s" % (fall, rise, height))
		if height < entropy*threshold_coef:
			# print("window (%s,%s) is deprecated." % (head, tail))
			window = []
			continue

		if rise < entropy*threshold_coef or fall < entropy*threshold_coef:
			# print("point %s is deprecated." % (str(nextpoint)))
			if point in segmentation:
				del segmentation[-1]
				segmentation.append(nextpoint)
			if i+2 < len(local_minimums):
				window = window+low_pass_gyroForceCurve[ nextpoint:local_minimums[i+2] ].tolist()
			continue

		if not segmentation:
			# print("%s is added." % head)
			segmentation.append(head)
		# print("%s is added." % tail)
		segmentation.append(tail)
		windows.append((head, tail))
		window = []

	# # split each segment data from raw curves # # # # # #
	# # # # rawdata is stored after going through a order=1 low-pass filter # # # #
	namespace = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

	# # # # rawdata is stored after going through a order=1 low-pass filter # # # #
	data = {}
	for key in namespace:
		data[key] = butter_lowpass_filter( jsdata[key], cutoff, fs, 1 )

	cnt = 0
	# print(pred_size, word)
	for i, chosen in enumerate(windows):
		chosen_head, chosen_tail = chosen

		d = {}
		d[ 'rgs' ] = normalize(gf[ chosen_head:chosen_tail ], nor_len=100)
		d[ 'gs' ] = normalize(lpgf[ chosen_head:chosen_tail ], nor_len=100)
		for key in namespace:
			d[ key ] = normalize(data[ key ][ chosen_head:chosen_tail ], nor_len=100)
			d[ 'r'+key ] = normalize(jsdata[ key ][ chosen_head:chosen_tail ], nor_len=100)
		result.append(d)
		cnt += 1
	# print('collect %s characters.' % cnt)

	if not return_segmentation:
		return result
	else:
		return (result, windows)


if __name__ == "__main__":

	# parse parameters
	parser = argparse.ArgumentParser(description='process input parameters')
	parser.add_argument('fn', help='input the filename you want to show', type=str)
	args = parser.parse_args()

	filename = args.fn
	word = filename.split('.')[0].split('/')[-1]
	fp = open(filename, "r")
	jsdata = json.loads(fp.read())
	fp.close()

	# step 1 : segmentation # # # # # # #
	# # segment data according to total sum of gyroscope curve # # # # # #
	valid_curve_length = min( [ len(c) for c in [ jsdata["gx"], jsdata["gy"], jsdata["gz"] ] ] )

	# # # # low-pass filter gyroscope curve # #
	gyroForceCurve = [ math.sqrt( jsdata["gx"][i]**2 + jsdata["gy"][i]**2 + jsdata["gz"][i]**2 ) for i in range( valid_curve_length ) ]
	low_pass_gyroForceCurve = butter_lowpass_filter( gyroForceCurve, cutoff, fs, order )

	gf = list(gyroForceCurve)
	tmp_gf = list(gyroForceCurve)
	lpgf = butter_lowpass_filter( tmp_gf, cutoff, fs, 1 )

	# # # # find local minimum in the curve, as candidates of segmentation point # # # #
	local_minimums, = argrelextrema( np.array( low_pass_gyroForceCurve ), np.less )
	local_minimums = local_minimums.tolist()
	if low_pass_gyroForceCurve[0] < low_pass_gyroForceCurve[1]:
		local_minimums.insert(0, 0)
	if low_pass_gyroForceCurve[-1] < low_pass_gyroForceCurve[-2]:
		local_minimums.append(len(low_pass_gyroForceCurve)-1)

	# # # # judge segmentation candidates' point, segmentation finish # # # #
	segmentation = []
	threshold = ( max( low_pass_gyroForceCurve ) + min( low_pass_gyroForceCurve ) ) / 2
	entropy = max( low_pass_gyroForceCurve ) - min( low_pass_gyroForceCurve )
	threshold_coef = float(1)/3.5
	# print(local_minimums)

	window = []
	windows = []
	head = 0
	tail = 0
	for i, point in enumerate(local_minimums[:-1]):

		nextpoint = local_minimums[i+1]
		if not window:
			window = low_pass_gyroForceCurve[ point:nextpoint ].tolist()
			head = point
		tail = nextpoint

		MAX = max(window)
		MIN = min(window)
		rise = MAX - window[0]
		fall = MAX - window[-1]
		height = MAX - MIN

		# print("fall = %s, rise = %s, height = %s" % (fall, rise, height))
		if height < entropy*threshold_coef:
			# print("window (%s,%s) is deprecated." % (head, tail))
			window = []
			continue

		if rise < entropy*threshold_coef or fall < entropy*threshold_coef:
			# print("point %s is deprecated." % (str(nextpoint)))
			if point in segmentation:
				del segmentation[-1]
				segmentation.append(nextpoint)
			if i+2 < len(local_minimums):
				window = window+low_pass_gyroForceCurve[ nextpoint:local_minimums[i+2] ].tolist()
			continue

		if not segmentation:
			# print("%s is added." % head)
			segmentation.append(head)
		# print("%s is added." % tail)
		segmentation.append(tail)
		windows.append((head, tail))
		window = []

	# print(segmentation)


	# # split each segment data from raw curves # # # # # #

	# # # # dataset structure # # # #
	# # # # | 'a' : [{'ax':[], 'ay': [], ...}, {}, {}, ...] | # # # #
	# # # # | 'b' : [{}, {}, {}, ...] 						| # # # #
	# # # # | 'c' : [{}, {}, {}, ...] 						| # # # #
	# # # # |           :             						| # # # #
	# # # # |           :             						| # # # #
	# # # # | 'y' : [{}, {}, {}, ...] 						| # # # #
	# # # # | 'z' : [{}, {}, {}, ...] 						| # # # #
	try:
		client = pymongo.MongoClient("localhost", 27017)
	except Exception as e:
		raise

	db = client.test_set

	namespace = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

	# # # # rawdata is stored after going through a order=1 low-pass filter # # # #
	data = {}
	for key in namespace:
		data[key] = butter_lowpass_filter( jsdata[key], cutoff, fs, 1 )

	cnt = 0
	pred_size = (segmentation[-1]-segmentation[0])/len(word)
	# print(pred_size, word)
	for i, ch in enumerate(word):
		pred_head = segmentation[0]+pred_size*i
		pred_tail = segmentation[0]+pred_size*(i+1)

		chosen = None
		min_diff = (segmentation[-1]-segmentation[0])
		for head, tail in windows:
			diff = abs(head-pred_head)+abs(tail-pred_tail)
			if diff < min_diff:
				chosen = (head, tail)
				min_diff = diff

		# print(chosen)
		chosen_head, chosen_tail = chosen

		d = {}
		d[ 'rgs' ] = normalize(gf[ chosen_head:chosen_tail ], nor_len=100)
		d[ 'gs' ] = normalize(lpgf[ chosen_head:chosen_tail ], nor_len=100)
		for key in namespace:
			d[ key ] = normalize(data[ key ][ chosen_head:chosen_tail ], nor_len=100)
			d[ 'r'+key ] = normalize(jsdata[ key ][ chosen_head:chosen_tail ], nor_len=100)
		db[toClass(ch)].insert_one(d)
		cnt += 1