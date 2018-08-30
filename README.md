![Alt Image Text](https://epl.tw/wp-content/uploads/2018/08/hardware.png "Hardware setup")
**On-Pen handwritten Word Recognition Using Long Short-Term Memory Model** describes a system for text input from handwriting using a conventional pen with a clip-on sensing unit called [EcoMini](https://epl.tw/ecomini/). <br><br>
The clip-on unit is a wireless sensor node that collects data from a triaxial accelerometer and a triaxial gyroscope and transmits it to a conventional personal computer. <br><br>
The host computer then performs segmentation to handle continuous handwriting, followed by LSTM-based classification. Moreover, we use a lexicon-based corrector to increase the accuracy. <br><br>
Experimental results show our proposed system to achieve good accuracy and reasonable latency for interactive use.
	
## Pre-installation
* Environment: [Ubuntu16.04](http://releases.ubuntu.com/16.04/)

* [Mongodb](https://docs.mongodb.com/manual/administration/install-community/)

* [bluepy](https://github.com/IanHarvey/bluepy)

* Install_packages.sh
	* This script has all packages that we needed.

## System overview
![Alt Image Text](https://epl.tw/wp-content/uploads/2018/08/System-overview.png "System Overview")

There are four steps to implement the system:

1. Collected raw data from sensor and stored with json type.
	* train.py
	
2. Preprocess (low-pass fillter and segmentation) and upload to mongodb.
	* preprocessing.py
	
3. Get data from mongobd and stored to train.dat .
	* test.py
	
4. Using train.data to train LSTM model. 
	* LSTM.py (LSTM model)
	
5. Lexicon calibration
	* LSTMsocket.py
		* corrector.py (weighted Levenshtein distance)
		* spell.py (Bayes corrector)

## Demo [Video link](https://youtu.be/ZACSAVZMsMM)
![Alt Image Text](https://epl.tw/wp-content/uploads/2018/08/demo.png "Demo")

There are two steps:

1. As long as the name of the data file needed to be identified is passed via socket, it will return the recognized result.
	* LSTMsocket.py
2. Now demo.py is only for the fixed address of device (it will not be connected to other devices), so it be needed to change the address. Finally, following the prompts and then the screen will show the recognized result.
	* demo.py

