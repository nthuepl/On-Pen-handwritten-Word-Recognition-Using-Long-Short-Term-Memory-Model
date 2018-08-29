import pickle
import json
import os

axis = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
for fn in os.listdir(os.getcwd()):
	if ".json" in fn:
		jsdata = json.loads(open(fn).read())
		print(jsdata)
		break
