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

# internet firewall - 4 classes, class counts: 37640, 14987, 12851, 54
mkdir data/firewall
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00542/log2.csv -O ./data/firewall/log2.csv

# chess - 18 classes, class counts: 2796, 1433, 2854, 2166,  471,  198, 4553, 1712,   78,  683,  592, 390, 1985, 4194,   81, 3597,  246,   27
mkdir data/chess
wget https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data -O ./data/chess/krkopt.data

# nursery - 4 classes, class counts: 4320, 328, 4266, 4044
mkdir data/nursery
wget https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data -O data/nursery/nursery.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.names -O data/nursery/nursery.names

# poker - 10 classes, class counts: 
mkdir data/poker
wget https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data -O data/poker/poker-hand-testing.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data -O data/poker/poker-hand-training-true.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand.names -O data/poker/poker-hand.names

# mushroom - 2 classes, class counts: 3156, 2937
mkdir data/mushroom
wget https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data -O data/mushroom/agaricus-lepiota.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names -O data/mushroom/agaricus-lepiota.names

# wine quiaty - 5 classes
mkdir data/wine
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv -O data/wine/winequality-red.csv
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv -O data/wine/winequality-white.csv
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names -O data/wine/winequality.names 

# abalone - 16 classes
mkdir data/abalone
wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data -O data/abalone/abalone.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names -O data/abalone/abalone.names
