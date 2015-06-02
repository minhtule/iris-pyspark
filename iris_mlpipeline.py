__author__ = 'minhtule'

import logging
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


logging.getLogger().setLevel(logging.INFO)
conf = SparkConf().setMaster('local').setAppName('Iris')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

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
    'Iris-setosa': 2,
    'Iris-versicolor': 1,
    'Iris-virginica': 0
}

MAP_CODE_TO_FLOW_NAME = {v: k for (k, v) in MAP_FLOWER_NAME_TO_CODE.iteritems()}

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


def visualize(data, models, name, testError):
    # Plot the decision boundary
    xx, yy = np.meshgrid(np.arange(X_MIN, X_MAX, MESH_STEP_SIZE), np.arange(Y_MIN, Y_MAX, MESH_STEP_SIZE))
    grid_points = sc.parallelize(np.c_[xx.ravel(), yy.ravel()].tolist())
    Z = np.array(predict(grid_points, models))

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.pcolormesh(xx, yy, Z)

    # Plot data points

    X = np.array(data.map(lambda p: p.features).collect())
    Y = np.array(data.map(lambda p: p.label).collect())

    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Test error = %.3f' % testError)

    plt.savefig(name)
    plt.close()



def buildModel(data, label):
    """
    Build a pipeline to classify `label` against the rest of classes using Binary Regression Classification

    :param data: the training data as RDD
    :param label: 0..C-1 where C is the number of classes
    :param shouldDisplayGraph: True to plot the graph illustrating the classification
    :return: the model as a Transformer
    """
    logging.info('building model for label = %d, type = %s' % (label, type(label)))
    lr = LogisticRegression()
    pipeline = Pipeline(stages=[lr])

    paramGrid = ParamGridBuilder()\
        .addGrid(lr.maxIter, [100])\
        .addGrid(lr.elasticNetParam, [0.0, 1.0])\
        .addGrid(lr.fitIntercept, [True, False])\
        .build()
    crossValidator = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,
                                    evaluator=BinaryClassificationEvaluator(), numFolds=15)

    dataDF = data.map(lambda point: LabeledPoint(0 if point.label == label else 1, point.features)).toDF()
    model = crossValidator.fit(dataDF)

    return model


def selectBestPrediction(predictions):
    bestPrediction = 0
    bestProbability = predictions[0].probability[0]
    for index, prediction in enumerate(predictions):
        if prediction.probability[0] > bestProbability:
            bestPrediction = index
            bestProbability = prediction.probability[0]

    return bestPrediction


def predict(points, models):
    """
    Predict the label of points

    :param points: RDD of Vector of features
    :param models: Prediction models for all class
    :return: Return label predictions
    """
    points = points.zipWithIndex().map(lambda p: Row(id=p[1], features=Vectors.dense(p[0]))).toDF()
    multiclass_predictions = [model.transform(points).orderBy('id').rdd.collect() for model in models]

    return map(selectBestPrediction, zip(*multiclass_predictions))


def main():
    data = sc.textFile('iris.dat').map(parse_point)
    random_seed = randint(0, 1e5)
    trainingData, testData = data.randomSplit([8, 2], seed=random_seed)

    numClasses = 3
    models = [buildModel(trainingData, label) for label in range(numClasses)]

    testDataPredictionAndActual = zip(predict(testData.map(lambda x: x.features), models),
                                      map(lambda p: p.label, testData.collect()))
    numOfError = len(filter(lambda x: x[0] != x[1], testDataPredictionAndActual))
    testError = float(numOfError) / testData.count()

    visualize(data, models, 'result-%d' % random_seed, testError)


main()
