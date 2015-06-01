__author__ = 'minhtule'

import logging
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

# Data attributes
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
#    -- Iris-setosa (0)
#    -- Iris-versicolour (1)
#    -- Iris-virginica (2)

MAP_FLOWER_NAME_TO_CODE = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

def parse_point(line):
    line = line.split(',')
    flower_code = MAP_FLOWER_NAME_TO_CODE[line[4]]
    features = [float(x) for x in line[:2]]
    return LabeledPoint(flower_code, features)

def predictionError(model, data):
    actualsAndPredictions = data.map(lambda p: (p.label, model.predict(p.features)))
    error = actualsAndPredictions.filter(lambda (actual, prediction): actual != prediction).count() / float(data.count())
    return error

def train(trainingData, validationData, **kwargs):
    iterationsValues = kwargs['iterations'] if 'iterations' in kwargs else [10, 1e2, 1e4]
    regTypeValues = kwargs['regType'] if 'regType' in kwargs else [None, 'l1', 'l2']
    interceptValues = kwargs['intercept'] if 'intercept' in kwargs else [True, False]
    correctionsValues = kwargs['corrections'] if 'corrections' in kwargs else [10, 1e2]
    bestModel = None
    bestParams = None
    bestError = float('inf')

    for (iterations, regType, intercept, corrections) in itertools.product(iterationsValues, regTypeValues,
                                                                           interceptValues, correctionsValues):
        model = LogisticRegressionWithLBFGS.train(trainingData, numClasses=3, iterations=iterations, regType=regType,
                                                  intercept=intercept, corrections=corrections)
        validationError = predictionError(model, validationData)
        logging.info('Validation error = %.3f with iterations = %d, regType = %s, intercept = %s, corrections = %d' %
                     (validationError, iterations, regType, intercept, corrections))

        if validationError < bestError:
            bestError = validationError
            bestModel = model
            bestParams = (iterations, regType, intercept, corrections)

    return bestModel, bestError, bestParams

def visualize(data, model, figure_name, error, params):
    X = np.array(data.map(lambda p: p.features).collect())
    Y = np.array(data.map(lambda p: p.label).collect())

    MESH_STEP_SIZE = 0.02

    # Plot the decision boundary

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP_SIZE), np.arange(y_min, y_max, MESH_STEP_SIZE))
    Z = np.array(map(lambda point: model.predict(point), np.c_[xx.ravel(), yy.ravel()]))

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.pcolormesh(xx, yy, Z)

    # Plot data points

    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    model_weights = map(lambda x: round(x, 2), model.weights)
    plt.suptitle('error = %.3f with\n'
                 'model=(weights = %s, intercept = %s)\n'
                 'iterations = %d, regType = %s, intercept = %s, corrections = %d'
                 % ((error, model_weights, model.intercept) + params), fontsize=11, fontweight='bold')
    plt.savefig('%s.png' % figure_name)
    plt.close()

def main():
    data = sc.textFile('iris.dat').map(parse_point)

    training_random_seed = random.randint(0, 1e5)
    trainingData, restData = data.randomSplit([6, 4], seed=training_random_seed)
    validationData, testData = restData.randomSplit([2, 2])

    model, validationError, params = train(trainingData, validationData)
    logging.info('Params used for the best model: iterations = %d, regType = %s, intercept %s, corrections = %d' %
                 params)
    logging.info('Validation error = %.3f' % validationError)

    testError = predictionError(model, testData)
    logging.info('Test error = %.3f' % testError)

    visualize(data, model, 'result-%d' % training_random_seed, testError, params)


logging.getLogger().setLevel(logging.INFO)
conf = SparkConf().setMaster('local').setAppName('Iris')
sc = SparkContext(conf=conf)

for i in range(10):
    logging.info('Iteration %d' %i)
    main()

