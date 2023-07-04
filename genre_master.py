from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator


def calculate_distance(instance1, instance2, k):
    distance = np.sum(np.abs(instance1[:k] - instance2[:k]))
    return distance


def get_neighbors(training_set, instance, k):
    distances = []
    for x in range(len(training_set)):
        dist = calculate_distance(training_set[x], instance, k) + calculate_distance(instance, training_set[x], k)
        distances.append((training_set[x][-1], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_majority_class(neighbors):
    class_vote = {}
    for neighbor in neighbors:
        if neighbor in class_vote:
            class_vote[neighbor] += 1
        else:
            class_vote[neighbor] = 1
    sorted_votes = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def calculate_accuracy(test_set, predictions):
    correct = sum(1 for x in range(len(test_set)) if test_set[x][-1] == predictions[x])
    return correct / len(test_set)


directory = "__path_to_dataset__"
output_file = "my.dat"

features = []
class_index = 0

for folder in os.listdir(directory):
    class_index += 1
    if class_index == 11:
        break
    for file in os.listdir(directory + folder):
        (rate, sig) = wav.read(directory + folder + "/" + file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, class_index)
        features.append(feature)

with open(output_file, 'wb') as f:
    pickle.dump(features, f)

dataset = []


def load_dataset(filename, split, training_set, test_set):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                break

    for data in dataset:
        if random.random() < split:
            training_set.append(data)
        else:
            test_set.append(data)


training_set = []
test_set = []
load_dataset(output_file, 0.66, training_set, test_set)

test_set_length = len(test_set)
predictions = []
for x in range(test_set_length):
    neighbors = get_neighbors(training_set, test_set[x], 5)
    predictions.append(get_majority_class(neighbors))

accuracy = calculate_accuracy(test_set, predictions)
print("Accuracy:", accuracy)
