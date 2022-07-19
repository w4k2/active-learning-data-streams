#! /bin/bash

# accelerometer dataset, 3 labels, well ballanced dataset
mkdir data/accelerometer
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00611/accelerometer.csv -O ./data/accelerometer/accelerometer.csv

# adult dataset - binary labels, imbalance ratio 1:3
mkdir data/adult
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O ./data/adult/adult.data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names -O ./data/adult/adult.names
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -O ./data/adult/adult.test
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names -O ./data/adult/old.adult.names

# bank marketing - binary lables, imbalance ratio 1:8
mkdir data/bank_marketing
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip -O ./data/bank_marketing/bank.zip
unzip data/bank_marketing/bank.zip -d data/bank_marketing/
