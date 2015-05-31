__author__ = 'minhtule'

import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setMaster('local').setAppName('Iris')
sc = SparkContext(conf=conf)

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

data = sc.textFile('iris.dat').map(parse_point)

# data = sc.parallelize([
#     LabeledPoint(0.0, [0.0, 10.0]),
#     LabeledPoint(1.0, [10.0, 0.0]),
#     LabeledPoint(2.0, [0.0, 0.0]),
#     LabeledPoint(2.0, [10.0, 10.0]),
#     LabeledPoint(2.0, [5.0, 5.0])
# ])

model = LogisticRegressionWithLBFGS.train(data, iterations=100, numClasses=3, corrections=100000, regType='l1')

labelsAndPreds = data.map(lambda p: (p.label, model.predict(p.features)))
trainingError = labelsAndPreds.filter(lambda (value, prediction): value != prediction).count() / float(data.count())
print('Training error = %s' % trainingError)

# Visualization

X = np.array(data.map(lambda p: p.features).collect())
Y = np.array(data.map(lambda p: p.label).collect())

MESH_STEP_SIZE = 0.2

# Plot the decision boundary

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP_SIZE), np.arange(y_min, y_max, MESH_STEP_SIZE))
Z = np.array(map(lambda point: model.predict(point), np.c_[xx.ravel(), yy.ravel()]))

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z)

# Plot training points

plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.savefig('result.png')
plt.close()

