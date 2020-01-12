import random
train_data_path = './data/KDDTrain_20Percent.txt'
train_data = []

with open(train_data_path) as f:
    for line in f:
        train_data.append(line.strip().split(','))

random.shuffle(train_data)

print(len(train_data[0]))