import sys

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.sql.functions import *


def labeled_points(rdd,n,label):
	rdd_list=rdd.map(lambda x: x.split())
	hTF=HashingTF(n)
	rdd_h=rdd_list.map(lambda x: hTF.transform(x))
	rdd_h_l=rdd_h.map(lambda x: LabeledPoint(label,x))
	return rdd_h_l

def train_model(labeled_class_0,labeled_class_1):
	emails = sc.union([labeled_class_0,labeled_class_1])	
	trained_model = LogisticRegressionWithSGD.train(emails)
	return trained_model, emails

def predictions(query,trained_model):
	query_p = labeled_points(query,100,0)
  	test_data = query_p.map(lambda x: x.features)

  	pred_test = trained_model.predict(test_data)

  	from pyspark.sql import SQLContext
  	sqlContext = SQLContext(sc)
   
  	df=pred_test.zip(query).toDF().select(col("_1").alias("prediction"), col("_2").alias("email"))
  	df.registerTempTable("df")
  	return sqlContext.sql("select * from df").show()

def train_accuracy(training_set,trained_model):
	training_data=training_set.map(lambda x: x.features)

	pred_train = trained_model.predict(training_data)
	actual = training_set.map(lambda x: x.label)
	predict_label = pred_train.zip(actual)

	overall_accuracy = '{:.2%}'.format(predict_label.filter(lambda (v, p): v == p).count() / float(pred_train.count()))

	n_spam = predict_label.filter(lambda (v,p): p == 1).count()
	spam_accuracy = '{:.2%}'.format(predict_label.filter(lambda (v, p): (p == 1) & (v == p)).count() / n_spam)


	n_nospam =predict_label.filter(lambda (v,p): p == 0).count()
	no_spam_accuracy = '{:.2%}'.format(predict_label.filter(lambda (v, p): (p == 0) & (v == p)).count() / n_nospam)

	print("overall train accuracy: %s" %overall_accuracy)	
	print("train spam accuracy: %s" %spam_accuracy)	
	print("train no spam accuracy: %s" %no_spam_accuracy)


if __name__ == "__main__":

  # create Spark context with Spark configuration
  conf = SparkConf().setAppName("Spam Classifier")
  sc = SparkContext(conf=conf)

  # read the input files
  no_spam=sc.textFile(sys.argv[1])
  spam=sc.textFile(sys.argv[2])
  query=sc.textFile(sys.argv[3])
	
  # create labeled datasets
  no_spam_labeled=labeled_points(no_spam,100,0)
  spam_labeled=labeled_points(spam,100,1)

  # training the model
  lr_model, emails = train_model(no_spam_labeled,spam_labeled)  

  #Making predictions on test set

  print(predictions(query,lr_model))

  # Calculating trainning accuracy

  train_accuracy(emails,lr_model)
