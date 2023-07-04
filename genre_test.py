from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator
from collections import defaultdict

dataset = []

def load_dataset(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                break

load_dataset("my.dat")

def calculate_distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1)
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def get_neighbors(training_set, instance, k):
    distances = []
    for x in range(len(training_set)):
        dist = calculate_distance(training_set[x], instance, k) + calculate_distance(instance, training_set[x], k)
        distances.append((training_set[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def get_majority_class(neighbors):
    class_vote = defaultdict(int)
    for neighbor in neighbors:
        class_vote[neighbor] += 1
    sorted_votes = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

results = defaultdict(str)

i = 1
for folder in os.listdir("./musics/wav_genres/"):
    results[i] = folder
    i += 1

(rate, sig) = wav.read("__path_to_new_audio_file_")
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, 0)

pred = get_majority_class(get_neighbors(dataset, feature, 5))

print(results[pred])
