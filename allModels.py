import pandas as pd 
import numpy as np
import math
import scipy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPClassifier
from concurrent.futures import ThreadPoolExecutor
import pickle as pickle
import csv as csv

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from warnings import filterwarnings

def normalize_column(X):
    """
    You will get overflow problems when calculating exponentials if 
    your feature values are too large.  This function adjusts all values to be
    in the range of 0 to 1 for each column.
    """         
    X = X - X.min() # shift range to start at 0
    normalizedX = X/X.max() # divide by possible range of values so max is now 1
    return normalizedX

def normalize_data(X):
    columns = X.columns
    new = []
    for column in columns:
        new.append(normalize_column(X[column]))
    return pd.DataFrame(new).transpose()


def runTuneTest_multithreaded(learner, parameters, X, y, continuous=False):
    """
    Uses Stratified K Fold with 5 splits on an Exhaustive Grid Search to tune
        hyperparameters on given learner. Finds best hyperparameters and score 
        for each fold
    Params:
        learner (SKLearn Model): The learner model to be evaluated
        parameters (dict): The hyperparameters to tune
        X (data): The feature values of the dataset to train/test on
        y (data): The label values of the dataset to train/test on
    Returns:
        scores (list): A list of the best score for each fold
    """
    splits = 5
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    if continuous:
        skf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    
    scores = []
    with ThreadPoolExecutor(max_workers=splits) as executor:
        i=1
        futures = []
        for train, test in skf.split(X, y):
            futures.append(executor.submit(__do_single_split, train, test, i, X, y, learner, parameters))
            i+=1

        for future in futures:
            j, score, best_params = future.result()
            scores.append(score)
            print(f"\tFold {j}:\n\tBest parameters: {best_params}\n\tTuning Set Score: {score}\n")
        
    return scores

def runTuneTest_singlethread(learner, parameters, X, y, continuous=False):
    """
    Uses Stratified K Fold with 5 splits on an Exhaustive Grid Search to tune
        hyperparameters on given learner. Finds best hyperparameters and score 
        for each fold
    Params:
        learner (SKLearn Model): The learner model to be evaluated
        parameters (dict): The hyperparameters to tune
        X (data): The feature values of the dataset to train/test on
        y (data): The label values of the dataset to train/test on
    Returns:
        scores (list): A list of the best score for each fold
    """
    splits = 5
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    if continuous:
        skf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    
    scores = []
    models = []
    i = 1
    for train, test in skf.split(X, y):
        j, score, best_params = __do_single_split(train, test, i, X, y, learner, parameters)
        scores.append(score)
        print(f"\tFold {i}:\n\tBest parameters: {best_params}\n\tTuning Set Score: {score['accuracy']}\n")
        i+=1
        
    return scores

def __do_single_split(train, test, i, X, y, learner, parameters):
    """
    Helper Function for RunTuneTest. Allows for easy parallelization of stratified folds
    """
    print(f"Executing fold {i}")
    clf = GridSearchCV(learner, parameters, cv=3)
    #print("did_grid_search")
    trainX = X.iloc[train]
    trainY = y.iloc[train]
    if isinstance(learner, SGDClassifier) and len(np.unique(trainY)) < 2:
        print("Skipping uneven test labels for SGD Classifier on fold", str(i))
        return i, {"accuracy": 0.0, "no_precision": 0.0, "no_recall": 0.0, "yes_precision": 0.0, "yes_recall": 0.0}, {}
    clf.fit(trainX, trainY)
    #print("did_fit")
    testX = X.iloc[test]
    testY = y.iloc[test]
    score = clf.score(testX, testY)
    y_predicted = clf.predict(testX)
    report = classification_report(testY, y_predicted, output_dict=True)
    # print((report))
    scores = {"accuracy": score, "no_precision": report['No']['precision'], "no_recall": report['No']['recall'], "yes_precision": report['Yes']['precision'], "yes_recall": report['Yes']['recall']}
    best_params = clf.best_params_
    return i, scores, best_params

def runPipeline(X, y, identifier=""):
    """
    PipeLine function that identifies the best parameters for each model.
        Prints the accuracy scores for each model, across 5 Stratified K Folds. 
        Runs pipeline for a Random Forest, K Nearest Neighbors, Decision Tree, and Stochastic Gradient Descent Classifiers
    Params:
        X (pd.Dataframe) : Examples to train/test on
        y (pd.Dataframe) : Example labels to train/test on
        identifier (str) : Optional name of data
    Returns:
        dictionary of accuracy scores for each model
    """

    print("Running pipeline for", f"'{identifier}'")
    
    rf_classifier = RandomForestClassifier(n_estimators=100)  # Fewer trees
    rf_parameters = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10, 50],
        }
    rf_results = runTuneTest_multithreaded(rf_classifier, rf_parameters, X, y)

    knn_classifier = KNeighborsClassifier()
    knn_parameters = {
        'n_neighbors': [5, 10, 20],
        'weights': ['uniform', 'distance'],
        }
    knn_results = runTuneTest_multithreaded(knn_classifier, knn_parameters, X, y)

    dt_classifier = DecisionTreeClassifier()
    dt_parameters = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10, 50],
        'criterion': ['gini', 'entropy'],
        }
    dt_results = runTuneTest_multithreaded(dt_classifier, dt_parameters, X, y)

    sgd_classifier = SGDClassifier(loss='log_loss', max_iter=50, tol=None, penalty=None, eta0=0.1)
    sgd_parameters = {
        'loss': ['hinge', 'log_loss'],
        'penalty': ['l2', 'elasticnet'],
        'alpha': [1e-4, 1e-2],
        }
    sgd_results = runTuneTest_multithreaded(sgd_classifier, sgd_parameters, X, y)

    final_res = [rf_results, knn_results, dt_results, sgd_results] # , mlp_results]
    classifiers = ["RandomForest", "KNN", "DecisionTree", "SGD"] # , "MLP"]

    values = dict(zip(classifiers, final_res))

    print(f"Identifier: {identifier}")
    for name, results in zip(classifiers, final_res):
        for fold, acc in enumerate(results, 1):
            print(f"{name}, Fold {fold}: {acc['accuracy'] * 100:.2f}%")

    return values

def new_runPipeline(data, identifier=""):
    return runPipeline(data.drop(columns=['RainTomorrow']), data['RainTomorrow'], identifier=identifier)


imputations = ['reg-class', 'mean-mode', 'hybrid']
variations = ['drop-loc+date', 'drop-loc-(month-disc)', 'drop-loc-(month-circ)', 'drop-date-(lat-long)', 
              'disc-month-lat-long', 'disc-month+loc', 'month-circ-lat-long', 'month-circ-lat-long-wind-circ']
'''imputations = ['remove', 'reg-class', 'mean-mode', 'hybrid']
variations = ['drop-loc+date', 'drop-loc-(month-disc)', 'drop-loc-(month-circ)', 'drop-date-(lat-long)', 
              'disc-month-lat-long', 'disc-month+loc', 'month-circ-lat-long', 'month-circ-lat-long-wind-circ', 
              'split-loc-(drop-date)', 'split-season-(drop-loc)', 'split-loc+season',
              'split-loc-(month-disc)', 'split-loc-(month-circ)', 'split-season-(lat-long)',
              'rm-temp1', 'rm-temp2', 'rm-pres1', 'rm-pres2', 'rm-hum1', 'rm-hum2', 'rm-1', 'rm-2', 'rm-3', 'rm-4',
               'rm-rain', 'rm-all-but-rain', 'rm-all-but-rain+temp']'''

print("STARTING FOR LOOP")
dictionary = {}
for imp in imputations:
    dictionary[imp] = {}
    for var in variations:
        print("\n\n")
        data = pd.read_csv(f'/scratch/srebarb1/NewNewCSVs/{imp}_{var}.csv')
        if 'split-loc' in var:
            new_dict = {}
            locs = data.Location.unique()
            for loc in locs:
                new_data = data[data.Location == loc]
                new_data = new_data.drop(columns=['Location'])
                new_dict[loc] = new_runPipeline(new_data, identifier=f'{imp}_{var}_{loc}')
        elif 'split-season' in var:
            new_dict = {}
            seasons = data.Season.unique()
            for season in seasons:
                new_data = data[data.Season == season]
                new_dict[season] = new_runPipeline(new_data, identifier=f'{imp}_{var}_{season}')
        elif 'split-loc+season' in var:
            new_dict = {}
            locs = data.Location.unique()
            seasons = data.Season.unique()
            for loc in locs:
                new_dict[loc] = {}
                for season in seasons:
                    new_data = data[(data.Location == loc) and (data.Season == season)]
                    new_dict[loc][season] = new_runPipeline(new_data, identifier=f'{imp}_{var}_{loc}_{season}')
        else:
            new_dict = new_runPipeline(data, identifier=f'{imp}_{var}')

        with open(f'/scratch/srebarb1/MLproject_dictionaries/{imp}_{var}.pkl', "wb") as f:
            pickle.dump(new_dict, f)
        dictionary[imp][var] = new_dict

with open(f'/scratch/srebarb1/MLproject_dictionaries/all.pkl', "wb") as f:
    pickle.dump(dictionary, f)


