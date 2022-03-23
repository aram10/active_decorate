import numpy as np
import math
import numbers
import random
import json
import time
import arff
import pickle

import impyute as impy
from enum import Enum
from sklearn import svm, datasets
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import list_datasets, load_dataset, list_metrics, load_metric

#GLOBALS
dataset = 'SOYBEAN'


class Logger:

    def __init__(self):
        self.experiment = 1
        self.history = []
        self.curr = []
        self.filepath = 'decoratelog.txt'

    def log(self, active_learner, iterations, acc, labeled_sizes, run_time, parameters):
        stuff = {
            'dataset': dataset,
            'active_learner': active_learner,
            'experiment_num': self.experiment,
            'al_cycles': iterations,
            'acc': acc,
            'num_examples': labeled_sizes,
            'runtime': run_time,
            'parameters': parameters,
        }
        self.curr.append(stuff)
        self.experiment += 1

    def write(self):
        f = open(self.filepath, "a")
        f.write(json.dumps(self.curr))
        f.close()
        self.history = self.curr
        self.curr = []


logger = Logger()

class FeatureType(Enum):
    CONTINUOUS = 0
    NOMINAL = 1


def feature_helper(X):
    types = []
    for x in X.T:
        is_cont = False
        for i in range(len(x)):
            if isinstance(x[0], numbers.Number) and not x[0].is_integer():
                is_cont = True
                break
        types.append(FeatureType.CONTINUOUS) if is_cont else types.append(FeatureType.NOMINAL)
    return types


def ensemble_predict(X, ensemble, get_probs=False):
    if len(ensemble) == 0:
        raise ValueError("Ensemble must be nonempty")
    # shape: (num_models_in_ensemble, num_datapoints, num_classes)
    probabilities = np.zeros(shape=(len(ensemble), X.shape[0], len(ensemble[0].__dict__['classes_'])))
    idx = 0
    for model in ensemble:
        prob = model.predict_proba(X)
        probabilities[idx] = prob
        idx = idx + 1
    avg_probabilities = probabilities.mean(0)
    return np.array([np.argmax(x) for x in avg_probabilities]) if not get_probs else avg_probabilities


def ensemble_error(y, y_pred):
    return np.count_nonzero(y - y_pred) / y.size


def adversarially_label(X, num_classes, ensemble, epsilon):
    # get the current predictions of the ensemble
    probs = ensemble_predict(X, ensemble, get_probs=True)

    # replace 0 probabilities with epsilon
    f = lambda x: x + epsilon if x == 0 else x
    f = np.vectorize(f)
    probs = f(probs)

    # normalize probabilities
    probs = probs / np.linalg.norm(probs)

    # adversarial probabilities
    adv_probs = 1 / probs / np.sum(1 / probs, axis=1)[:, None]
    y = np.empty(shape=(X.shape[0],))
    for i in range(len(adv_probs)):
        y[i] = np.random.choice(num_classes, 1, p=adv_probs[i])
    return y


def discretize_continuous_data(X, feat_types, num_bins):
    """
    feat_types: 0=nominal, 1=continuous
    """
    cont_feat_types = np.argwhere(feat_types == 1).flatten()
    for i in cont_feat_types:
        max_value = np.max(X[:, i])
        min_value = np.min(X[:, i])
        interval = (max_value - min_value) / num_bins
        X[:, i] = [np.ceil((x - min_value) / interval) if x > min_value else 1 for x in X[:, i]]
    return X


def generate_and_label_data(X, y, num_classes, ensemble, feat_types, r, epsilon):
    num_data = math.floor(X.shape[0] * r)
    R = np.empty(shape=(num_data, X.shape[1]))
    r_pred = np.empty(shape=(num_data,))
    i = 0
    for t in feat_types:
        if t == FeatureType.CONTINUOUS:
            mean = np.mean(X[:, i])
            std = np.sqrt(np.var(X[:, i]))
            for j in range(R.shape[0]):
                R[j, i] = np.random.normal(mean, std)
        else:
            values, counts = np.unique(X[:, i], return_counts=True)
            probs = counts / X[:, i].shape[0]
            R[:, i] = np.random.choice(values, R.shape[0], p=probs)
        i = i + 1
    adv_y = adversarially_label(R, num_classes, ensemble, epsilon)
    return R, adv_y


def decorate(base_learner_class, X, y, X_test, y_test, c_size, i_max, r_size, epsilon, silenced=False):
    parameters = {
        'committee size': c_size,
        'max iterations': i_max,
        'sampling size': r_size,
        'epsilon': epsilon
    }
    acc = []
    num_labeled = []
    start_time = time.time()
    i = 0
    trials = 0
    learner = base_learner_class()
    learner.fit(X, y)
    ensemble = [learner]
    err = ensemble_error(y, ensemble_predict(X, ensemble))
    acc.append(accuracy_score(y, ensemble_predict(X, ensemble)))
    # assuming all classes present in data
    num_classes = len(np.unique(y))
    feat_types = feature_helper(X)
    T = np.array(X, copy=True)
    t = np.array(y, copy=True)
    while i < c_size and trials < i_max:
        num_labeled.append(math.floor(r_size * T.shape[0] * (trials + 1)))
        # generate data, adversarially label
        R, adv_y = generate_and_label_data(T, t, num_classes, ensemble, feat_types, r_size, epsilon)
        # wrap new data into existing data
        T_new = np.concatenate((T, R))
        t_new = np.concatenate((t, adv_y))
        order = np.random.permutation(range(T_new.shape[0]))
        T_new = T_new[order]
        t_new = t_new[order]
        # train learner on data amalgamation
        C_new = base_learner_class()
        C_new.fit(T_new, t_new)
        # create new ensemble with learner added in
        ensemble_new = ensemble + [C_new]
        # evaluate new ensemble on dataset w/o synthetic examples
        err_new = ensemble_error(y_test, ensemble_predict(X_test, ensemble_new))
        if err_new < err:
            i = i + 1
            err = err_new
            ensemble = ensemble_new
            acc.append(accuracy_score(y_test, ensemble_predict(X_test, ensemble)))
        trials = trials + 1
    run_time = time.time() - start_time
    if not silenced:
        logger.log('DECORATE', trials, acc, num_labeled, run_time, parameters)
    return ensemble


def top_2_values(array):
    indexes = array.argsort()[-2:][::-1]
    A = set(indexes)
    B = set(list(range(array.shape[0])))
    array[list(B.difference(A))] = 0
    return array


def utility(ensemble, X):
    """
    :param ensemble: ensemble of classifiers (list)
    :param X: data
    :return: numpy 1D array of size X.shape[0], where utility[i] is
    the margin between the ensemble and the ith training example (margin = diff.
    between highest and second-highest class probabilities predicted by ensemble)
    """
    utility = np.zeros(shape=X.shape[0])
    probs = ensemble_predict(X, ensemble, get_probs=True)
    top_2_probs = np.sort(np.apply_along_axis(top_2_values, 1, probs))[:, probs.shape[1] - 2:]
    return np.abs(top_2_probs[:, 0] - top_2_probs[:, 1])


def vote_entropy(ensemble, X, y):
    """
    :param ensemble: ensemble of classifiers (list)
    :param X: data
    :param y: labels
    :return: numpy 1D array of size X.shape[0], where utility[i] is
    """
    labels = np.unique(y)
    C = len(ensemble)
    utility = np.zeros(shape=X.shape[0])
    predictions = []
    for model in ensemble:
        predictions.append(model.predict(X))
    predictions = np.vstack([el for el in predictions])
    votes = add_votes(predictions, labels)
    log = lambda x: math.log(x) if x != 0 else 0
    return np.array(
        [sum([(votes[i][j] / C) * log(votes[i][j] / C) for j in range(len(labels))]) for i in range(X.shape[0])])


def add_votes(predictions, labels):
    """
    :param predictions: an array of shape (ensemble_size, num_samples) where predictions[i][j] is the
        ith model's prediction of the class label of the jth example
    :param labels: 1D array of all unique (numerical) labels
    :return: array votes of shape (num_samples, num_labels) where votes[i][j] is the number of models
        that predicted the jth class label for the ith example
    """
    return np.array([[np.count_nonzero(example == label) for label in labels] for example in predictions.transpose()])


def active_decorate(X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test, base_learn, k, m, c_size, i_max,
                    r_size, epsilon, margin_utility=True):
    trials = k
    parameters = {
        'iterations': k,
        'm': m,
        'committee size': c_size,
        'max DECORATE iterations': i_max,
        'sampling size': r_size,
        'epsilon': epsilon
    }
    committee = []
    acc = []
    num_labeled = []
    start_time = time.time()
    while (k > 0):
        committee = decorate(base_learn, X_labeled, y_labeled, X_test, y_test, c_size, i_max, r_size, epsilon,
                             silenced=True)
        acc.append(accuracy_score(y_unlabeled, ensemble_predict(X_unlabeled, committee)))
        exp_utility = utility(committee, X_unlabeled) if margin_utility else vote_entropy(committee, X_unlabeled,
                                                                                          y_unlabeled)
        idx = np.argpartition(exp_utility, -m)[-m:]
        S, s = X_unlabeled[idx], y_unlabeled[idx]
        X_labeled = np.concatenate((X_labeled, S), axis=0)
        y_labeled = np.concatenate((y_labeled, s))
        num_labeled.append(X_labeled.shape[0])
        X_unlabeled = np.delete(X_unlabeled, idx, axis=0)
        y_unlabeled = np.delete(y_unlabeled, idx)
        k = k - 1
    logger.log('ACTIVE-DECORATE', trials, acc, num_labeled, time.time() - start_time, parameters)
    return decorate(base_learn, X_labeled, y_labeled, X_test, y_test, c_size, i_max, r_size, epsilon, silenced=True)


def load_soybean():
    # preprocess soybean dataset
    with open('data/soybean.arff', 'r') as f:
        data = arff.load(f)
        f.close()
    X = np.array(data['data'])
    y = X[:, -1]
    X = X[:, :-1]

    class_label_encoder = preprocessing.LabelEncoder()
    class_label_encoder.fit(y)
    y = class_label_encoder.transform(y)
    X[X == None] = 'None'
    feature_label_encoder = preprocessing.LabelEncoder()
    X = np.array([feature_label_encoder.fit(datapoint).transform(datapoint) for datapoint in X.transpose()]).transpose()
    X = X.astype('O')
    X[X == 0] = np.nan
    X = X.astype('float')
    X = impy.mode(X)
    return X, y
