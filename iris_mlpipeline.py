__author__ = 'minhtule'

import logging
import matplotlib.pyplot as plt
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

logging.getLogger().setLevel(logging.INFO)


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
    'Iris-virginica': 1
}

X_MIN = 3.8
X_MAX = 8.4
Y_MIN = 1.5
Y_MAX = 4.9
MESH_STEP_SIZE = 0.02


def parse_point(line):
    line = line.split(',')
    flower_code = MAP_FLOWER_NAME_TO_CODE[line[4]]
    features = [float(x) for x in line[:2]]
    return LabeledPoint(flower_code, features)


def visualize(data, model):

    # Plot the decision boundary
    Data = Row('features')

    xx, yy = np.meshgrid(np.arange(X_MIN, X_MAX, MESH_STEP_SIZE), np.arange(Y_MIN, Y_MAX, MESH_STEP_SIZE))
    grid_points = sc.parallelize(np.c_[xx.ravel(), yy.ravel()].tolist()).map(lambda x: Data(Vectors.dense(x))).toDF()
    predictions = model.transform(grid_points).collect()
    Z = np.array(map(lambda row: row.prediction, predictions))

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.pcolormesh(xx, yy, Z)

    # Plot data points

    X = np.array(data.map(lambda p: p.features).collect())
    Y = np.array(data.map(lambda p: p.label).collect())

    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.show()
    plt.close()


data = sc.textFile('iris.dat').map(parse_point)
trainingData, testData = data.randomSplit([7, 3], seed=10)
trainingData = trainingData.toDF()
testData = testData.toDF()

lr = LogisticRegression()
pipeline = Pipeline(stages=[lr])

paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [100, 10000])\
    .addGrid(lr.elasticNetParam, [0.0, 1.0])\
    .addGrid(lr.fitIntercept, [True, False])\
    .build()
crossValidator = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,
                                evaluator=BinaryClassificationEvaluator())
model = crossValidator.fit(trainingData)

