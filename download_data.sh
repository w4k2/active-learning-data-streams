#! /bin/bash

# accelerometer dataset, 3 labels, well ballanced dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00611/accelerometer.csv -O ./data/accelerometer.csv

# adult dataset - binary labels, imbalance ration 1:3
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O ./data/adult.data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names -O ./data/adult.names
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -O ./data/adult.test
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names -O ./data/old.adult.names

