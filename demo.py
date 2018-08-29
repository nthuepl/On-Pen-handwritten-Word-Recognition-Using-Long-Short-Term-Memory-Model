# from bluepy.btle import Scanner, DefaultDelegate
from bluepy.btle import *
# import numpy as np
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import threading
import Queue
from time import sleep
from QTGraph import QTGraph
import random
import json
import time
import math
import os
import socket

# Flag for stopping the program
RUNNING = True

# Devices' name

# The size of window
windowSize = 2000
x_lim = 0

# Global variables
acc_dividor = 4096
gyro_dividor = 65.5
accBias = [0,0,0]
gyroBias = [0,0,0]
DEG2RAD = 0.01745329251
alpha = 0.8

# numpy parameters
order = 1
fs = 30.0
cutoff = 2.0

targetDeviceName = "WinstonScale"
deviceTargets = []

updateLock = threading.Lock()
dataStreamList = [[0] for i in range(6)]
gravity = [float(0) for i in range(4)]
isRemoveNoise = [False, False, False, False, False, False]
acceleration = []
gyroscope = []

# plt.ion()
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3)
ax2 = plt.subplot2grid((3,3),(1,0),colspan=3)
ax3 = plt.subplot2grid((3,3),(2,0),colspan=3)

def eliminate_gravity(value, axis):
	a_policy = ['ax', 'ay', 'az']
	g_policy = ['gx', 'gy', 'gz']
	if axis in a_policy:
		index = a_policy.index(axis)
		gravity[index] = alpha*gravity[index]+(1-alpha)*value
		return value-gravity[index]
	elif axis in g_policy:
		return value
	else:
		return None

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def hexString2Int(string):
	num = 0
	buf_ = string.lower()
	for i in range(0, len(buf_), 1):
		ch = buf_[i]
		num = num*16+int(ch, 16)
	if num >= 32768:
		num = num-65536	
	return num

def hexString2float(string):
	singed = ''
	degree = ''
	tail = ''
	stream = ''

	for ch in string:
		stream = stream+format(int(ch, 16), 'b').zfill(4)

	signed = stream[0]
	degree = stream[1:9]
	tail = stream[9:]
	power = 0
	for i in range(0, len(degree), 1):
		if degree[7-i] == '1':
			power = power+pow(2, i)
	power = power-127
	
	base = 1.0
	for i in range(0, len(tail), 1):
		if (tail[i] == '1'):
			base = base+pow( 2, (-1)*(i+1) )

	if signed == '1':
		return (-1)*base*pow(2, power)
	elif signed == '0':
		return base*pow(2, power)

def callLSTM(filename):
	HOST = '127.0.0.1'
	PORT = 8001

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((HOST, PORT))

	s.send(filename)
	data = s.recv(1024)
	s.close()
	return data

class handleNotificationDelegate(DefaultDelegate):
	def __init__(self, scanEntry, lock=None):
		DefaultDelegate.__init__(self)
		self.scanEntry = scanEntry
		self.lock = lock

	def handleNotification(self, cHandle, data):
		reading = str(binascii.hexlify(data))
		global counter
		global windowSize
		global dataStreamQueue
		axis = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

		rawdata = [reading[4*i:4*i+4] for i in range(6)]
		rawvalue = [hexString2Int(data) for data in rawdata]
		rawvalue.append(int(reading[24:28], 16))
		rawvalue[0] = float(rawvalue[0]) / acc_dividor - accBias[0]
		rawvalue[1] = -( float(rawvalue[1]) / acc_dividor) + accBias[1]
		rawvalue[2] = -( float(rawvalue[2]) / acc_dividor ) + accBias[2]
		rawvalue[3] = ( float(rawvalue[3]) / gyro_dividor - gyroBias[0]) * DEG2RAD
		rawvalue[4] = -( float(rawvalue[4]) / gyro_dividor - gyroBias[1]) * DEG2RAD
		rawvalue[5] = -( float(rawvalue[5]) / gyro_dividor - gyroBias[2]) * DEG2RAD
		# for v in rawvalue: print(v)
		# print(" ")
		if self.lock != None:
			self.lock.acquire()

		# critical section
		for i in range(6): 
			dataStreamList[i].append(eliminate_gravity(rawvalue[i], axis[i]))
			# dataStreamList[i].append(rawvalue[i])
			if len(dataStreamList[i]) > windowSize:
				del dataStreamList[i][0]
			if len(dataStreamList[i]) == 20 and isRemoveNoise[i] == False:
				dataStreamList[i] = []
				isRemoveNoise[i] = True

		if dataStreamList[0]:
			acceleration.append(math.sqrt(dataStreamList[0][-1]**2 + dataStreamList[1][-1]**2 + dataStreamList[2][-1]**2))
			gyroscope.append(math.sqrt(dataStreamList[3][-1]**2 + dataStreamList[4][-1]**2 + dataStreamList[5][-1]**2))
			if len(acceleration) > windowSize:
				del acceleration[0]
				del gyroscope[0]

		# critical section
		if self.lock != None:
			self.lock.release()

class handleDiscoveryDelegate(DefaultDelegate):
	def __init__(self):
		DefaultDelegate.__init__(self)

	def handleDiscovery(self, scanEntry, isNewDev, isNewData):
		if isNewDev:
			print "Discovered device", scanEntry.addr
		elif isNewData:
			print "Received new data from", scanEntry.addr


def scanFunction():
	scanner = Scanner(1).withDelegate(handleDiscoveryDelegate())
	scanResults = scanner.scan(10)
	global deviceTargets

	for scanEntry in scanResults:
		"""
		print "Device %s (%s), RSSI=%d dB" % (scanEntry.addr, scanEntry.addrType, scanEntry.rssi)
		for (adtype, desc, value) in scanEntry.getScanData():
		    print "  %s = %s" % (desc, value)
		"""

		for (adtype, desc, value) in scanEntry.getScanData():
			if value == targetDeviceName:
				deviceTargets.append(scanEntry)
				print scanEntry.addr
				continue

	if (len(deviceTargets) > 1):
		print "< %s devices were found >" % len(deviceTargets)
	elif (len(deviceTargets) == 1):
		print "< %s device was found >" % len(deviceTargets)
	else:
		print "< No target device was found >"

class PeripheralThread(threading.Thread):
	def __init__(self, device):
		threading.Thread.__init__(self)
		self.device = device

	def run(self):
		# print "PeripheralThread : Run()"
		try:
			global updateLock
			connection = Peripheral(self.device)
			connection.setDelegate(handleNotificationDelegate(self.device, lock=updateLock))
			connection.getServices()
			Service = connection.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
			accBiasChar = Service.getCharacteristics()[0]
			gyroBiasChar = Service.getCharacteristics()[1]
			accBiasString = str(binascii.hexlify(accBiasChar.read()))
			gyroBiasString = str(binascii.hexlify(gyroBiasChar.read()))
			accBias = [accBiasString[4*i:4*i+4] for i in range(3)]
			gyroBias = [gyroBiasString[4*i:4*i+4] for i in range(3)]

			# accBias = [ float.fromhex(accBias[i]) for i in range(3) ]
			# gyroBias = [ float.fromhex(gyroBias[i]) for i in range(3) ]
			accBias = [ hexString2float(accBias[i]) for i in range(3) ]
			gyroBias = [ hexString2float(gyroBias[i]) for i in range(3) ]

			# print(accBias)
			# print(gyroBias)
			Characteristic_4 = Service.getCharacteristics()[3]
			connection.writeCharacteristic(Characteristic_4.handle + 2, struct.pack('<h', 0x0001), True)

			waitCycle = 0

			while RUNNING:
				if connection.waitForNotifications(0.5):
					continue

				print "Waiting..."
				waitCycle += 1
				if (waitCycle >= 3):
					raise ValueError('Timeout')
		except Exception as e:
			print "PeripheralThread Exception"
			if 'connection' in locals():
				connection.disconnect()
				del connection
			print(e)
			self.run()

def main():
	global deviceTargets
	global RUNNING
	global updateLock

	newScanThread = threading.Thread(target = scanFunction, name = "scanFunction") 
	newScanThread.setDaemon(True)
	newScanThread.start()
	newScanThread.join()    # Wait until it's done

	
	if (len(deviceTargets) != 0):
		# PeripheralThreads = []
		#for deviceTarget in deviceTargets:
		for idx, deviceTarget in enumerate(deviceTargets):
			print deviceTarget.addr
			try:
				print idx #############0 1 2 3
				print deviceTarget
				newPeripheralThread = PeripheralThread(deviceTarget)
				newPeripheralThread.setDaemon(True)
				newPeripheralThread.start()
				# PeripheralThreads.append(newPeripheralThread)
			except (KeyboardInterrupt, SystemExit):
				print "!!!!!!!!!!!!!!!"
				cleanup_stop_thread()
				sys.exit()


		while isRemoveNoise[0] == False:
			pass

		dataName = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
		# graph = QTGraph(dataStreamList, dataName, sourcesNum=6, x_range=windowSize, lock=updateLock, lowpass=True)
		graph = QTGraph([acceleration, gyroscope], ['acc', 'gyro'], sourcesNum=2, x_range=windowSize, lock=updateLock, lowpass=True)
		def cut(g, evt):
			global dataStreamList, RUNNING
			# generate English word
			wordset = set()
			fp = open('20k.txt', 'r')
			for line in fp.readlines():
				wordset.add(line[:-1])
			fp.close()
			sampleNum = 5
			menu = random.sample(wordset, sampleNum)
			print('Training set collection program start. There are %d words to write...' % sampleNum)
			for i, word in enumerate(menu):
				while True:
					updateLock.acquire()
					for s in dataStreamList: 
						del s[:-1]
					updateLock.release()
					print('[sample %s] Please write %s' % (i+1, word))
					evt.wait()
					break
				if RUNNING == False: 
					print('sampling thread termitating.')
					break
				data = {
					dataName[0] : list(dataStreamList[0]),
					dataName[1] : list(dataStreamList[1]),
					dataName[2] : list(dataStreamList[2]), 
					dataName[3] : list(dataStreamList[3]), 
					dataName[4] : list(dataStreamList[4]),
					dataName[5] : list(dataStreamList[5])
				}
				print("now recording...")
				time.sleep(3)

				# Write raw data to a plain text file
				filename = word+'.json'
				json_str = json.dumps(data)
				output = open(filename, 'w')
				output.write(json_str)
				output.close()
				print('identify: %s' % (callLSTM(filename)))
				evt.clear()

			if RUNNING:
				g.close()
				RUNNING = False
			print('##########training is done.##########')


		evt = threading.Event()
		cutting = threading.Thread(target=cut, args=(graph, evt,))
		graph.setTask(cutting, evt)
		graph.show(animate=True)
		if RUNNING:
			evt.set()
			graph.close()
			RUNNING = False
		cutting.join()

		print "< Program Terminated >"
	else:
		pass


if __name__ == '__main__':
	main()


