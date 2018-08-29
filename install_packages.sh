#!/bin/sh
apt_get_packages = "python-qt4 mongodb-clients"
sudo apt-get update
sudo pip install tensorflow
sudo pip install pymongo
sudo pip install progressbar
sudo pip install pyqtgraph
sudo apt-get install  -y $apt_get_packages
