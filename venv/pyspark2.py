# Databricks notebook source
# Import SQLContext and data types

from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, LinearRegression
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# COMMAND ----------

#There are 60 columns. Because I am a beginner, we are operating on only a limited set of columns.
#We will use more in the future when I am better.

#SP is a dataframe. Grab everything from the table.
sp = sqlContext.sql("SELECT * FROM russiantrain")
testTable = sqlContext.sql("SELECT * FROM russiantest")

#sp.printSchema()
#convert the data values from strings to integers/floats for future processing:
sp_num = sp.select(sp.id.cast("int"), sp.full_sq.cast("int"), sp.life_sq.cast("int"), sp.price_doc.cast("float"), \
                  sp.floor.cast("int"),sp.max_floor.cast("int"), sp.build_year.cast("int"), sp.kitch_sq.cast("int"),  \
                  sp.num_room.cast("int"), sp.state.cast("int"))


test_num = testTable.select(testTable.id.cast("int"), testTable.full_sq.cast("int"), testTable.life_sq.cast("int"), \
                  testTable.floor.cast("int"),testTable.max_floor.cast("int"), testTable.build_year.cast("int"), testTable.kitch_sq.cast("int"),  \
                  testTable.num_room.cast("int"), testTable.state.cast("int"))

#Data cleanup and imputation goes here.
# We should check percent of missing data, and either drop the data or do imputation.
# Currently we don't know what imputation is, so we just...
# ...Convert N/A values to 0. This assigns missing data to 0 which is not right.
sp_num = sp_num.fillna(0)
test_num = test_num.fillna(0)
#sp_num.printSchema()

#Split our train data set into 2 parts, one for training model, one for testing model.
(trainingData, testData) = sp_num.randomSplit([0.7, 0.3])
#sp_num.take(10)



# COMMAND ----------

# This identifies categorical features and indexes them into features.
featureAssembler = VectorAssembler().setInputCols([ \
   "full_sq", "life_sq", "floor", "max_floor", "build_year", "num_room", "kitch_sq", "state"]).setOutputCol("features")
pipeline = Pipeline().setStages([featureAssembler])
model=pipeline.fit(trainingData)

#Now we have two dataframes sorted as 'features' and 'label'.
trainLFDF = model.transform(trainingData).select("features","price_doc")
testLFDF = model.transform(testData).select("features","price_doc")
ttestLFDF = model.transform(test_num).select("id","features")
ttestLFDF.take(10)


# COMMAND ----------

#Creating an evaluator measuring our label vs our prediction using RMSE evaluation.
evaluator = RegressionEvaluator(metricName="rmse")\
  .setLabelCol("price_doc")\
  .setPredictionCol("prediction")

# COMMAND ----------

#Decision tree regression, testing on both train and test dataset.
dt = DecisionTreeRegressor(labelCol='price_doc')

#This builds the dt model using the train dataset
model = dt.fit(trainLFDF)
#This predicts dt model outcomes on train and test dataset
trainPredictions = model.transform(trainLFDF)
testPredictions = model.transform(testLFDF)

trainscore = evaluator.evaluate(trainPredictions)
testscore = evaluator.evaluate(testPredictions)
print(trainscore,testscore)

#DT 8 Vars RMSE 3493522, 3901961

# COMMAND ----------

#Gradient boosted tree regression 
gbt = GBTRegressor(labelCol='price_doc')
model = gbt.fit(trainLFDF)
gbt_trainP = model.transform(trainLFDF)
gbt_testP = model.transform(testLFDF)
gbt_tTestP = model.transform(ttestLFDF)

trainscore = evaluator.evaluate(gbt_trainP)
testscore = evaluator.evaluate(gbt_testP)
print(trainscore,testscore)

#DT 8 Vars RMSE 3078244, 3821738

# COMMAND ----------

#I'm trying to figure out how to save my output so we can submit it to the kaggle competition.
#How do it is to press display, followed by export (download CSV which is one of the bottom buttons).
gbt_submit = gbt_tTestP.select("id", "prediction")

#We clean it by changing negatives to zeros.
submitRDD = gbt_submit.rdd
submitRDD = submitRDD.map(lambda x: x if x[1] > 0 else [x[0],0] )


display(submitRDD.toDF())

# COMMAND ----------

#Testing linear regression model
lr = LinearRegression().setLabelCol("price_doc")

# Fit 2 models, using different regularization parameters
modelA = lr.fit(trainLFDF, {lr.regParam:100.0})
predictionsA = modelA.transform(trainLFDF)
testpredictions = modelA.transform(testLFDF)
predictionsA.take(2)

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse").setLabelCol("price_doc")
RMSE = evaluator.evaluate(predictionsA)
print("LinearRegression RMSE- train:" + str(RMSE) + " test:" + str(evaluator.evaluate(testpredictions)))


# ModelA 1 var: Root mean squared Error = 4543716.30277
# ModelA 2 vars: Root Mean Squared Error = 4461341.43412
# ModelA 4 vars: Root Mean Squared Error = 4501992.16466
# ModelA 6 vars: RMSE = 4311985
#LR 8 variables RMSE 4227835, 4263233

# COMMAND ----------

# Import numpy, pandas, and ggplot
import numpy as np
from pandas import *
from ggplot import *

# Create Python DataFrame
pop = trainLFDF.rdd.map(lambda p: (p.features[0])).collect()
price = trainLFDF.rdd.map(lambda p: (p.price_doc)).collect()
predA = predictionsA.select("prediction").rdd.map(lambda r: r[0]).collect()
predGBT = gbt_trainP.select("prediction").rdd.map(lambda r: r[0]).collect()

pydf = DataFrame({'pop':pop,'price':price,'predA':predA, 'predGBT':predGBT})

# COMMAND ----------

# Create scatter plot and two regression models (scaling exponential) using ggplot
p = ggplot(pydf, aes('pop','price')) +\
geom_point(color='blue') +\
geom_line(pydf, aes('pop','predA'), color='red') +\
geom_line(pydf, aes('pop','predGBT'), color='green') +\
scale_x_log10() + scale_y_log10()
display(p)

# COMMAND ----------

display(price)