from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
#for preprocess
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, HashingTF, NGram, Word2Vec
from pyspark.ml.feature import  MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
import joblib
import numpy as np
import csv
# end
import json


def preprocess_save(l,sc):
    spark = SparkSession(sc)
    df = spark.createDataFrame(l,schema='length long,subject_of_message string,content_of_message string,ham_spam string')

    # preprocessing part (can add/remove stuff) , right now taking the column subject_of_message for spam detection

    #preprocessing the content of message column
    tokenizer = Tokenizer(inputCol="content_of_message", outputCol="token_text")
    stopwords = StopWordsRemover().getStopWords() + ['-']
    stopremove = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text').setOutputCol('stop_tokens')
    bigram = NGram().setN(2).setInputCol('stop_tokens').setOutputCol('bigrams')
    word2Vec = Word2Vec(vectorSize=5, minCount=0, inputCol="bigrams", outputCol="feature2")
    mmscaler = MinMaxScaler(inputCol='feature2',outputCol='scaled_feature2')


    # pre-procesing the subject of message column
    tokenizer1 = Tokenizer(inputCol="subject_of_message", outputCol="token_text1")
    stopwords1 = StopWordsRemover().getStopWords() + ['-']
    stopremove1 = StopWordsRemover().setStopWords(stopwords).setInputCol('token_text1').setOutputCol('stop_tokens1')
    bigram1 = NGram().setN(2).setInputCol('stop_tokens1').setOutputCol('bigrams1')
    word2Vec1 = Word2Vec(vectorSize=5, minCount=0, inputCol="bigrams1", outputCol="feature1")
    mmscaler1 = MinMaxScaler(inputCol='feature1',outputCol='scaled_feature1')

    #ht = HashingTF(inputCol="bigrams", outputCol="ht",numFeatures=8000)
    ham_spam_to_num = StringIndexer(inputCol='ham_spam',outputCol='label')
    print("starting pre-processing of data (wait 2-3 min)")

    # applying the pre procesed pipeling model on the batches of data recieved
    data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,bigram,word2Vec,mmscaler,tokenizer1,stopremove1,bigram1,word2Vec1,mmscaler1])
    cleaner = data_prep_pipe.fit(df)
    clean_data = cleaner.transform(df)
    clean_data = clean_data.select(['label','stop_tokens','bigrams','feature1','feature2','scaled_feature2','scaled_feature1'])
    print("pre process of data completed")

    # show preprocess data
    clean_data.show()

    X = np.array(clean_data.select(['scaled_feature1','scaled_feature2']).collect())
    y = np.array(clean_data.select('label').collect())
    # reshaping the data
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    np.savetxt("feature_cols.csv", X, delimiter=',')
    np.savetxt("target_cols.csv", y, delimiter=',')
    


sc = SparkContext("local[2]","test")

id_count = 0
ssc = StreamingContext(sc,1)
spark = SparkSession(sc)
sql_context = SQLContext(sc)
lines = ssc.socketTextStream('localhost',6100)
batches = lines.flatMap(lambda line: line.split("\n"))
def process(rdd):
        if not rdd.isEmpty():

            json_strings = rdd.collect()
            for rows in json_strings:
                temp_obj = json.loads(rows,strict = False)
            rows_spam = []
            for i in temp_obj.keys():
                temp_l = []
                temp_l.append(len(str(temp_obj[i]['feature1'])))
                temp_l.append(str(temp_obj[i]['feature0']).strip(' '))
                temp_l.append(str(temp_obj[i]['feature1']).strip(' '))
                temp_l.append(str(temp_obj[i]['feature2']).strip(' '))
                rows_spam.append(tuple(temp_l))
            print("Recieved batch of data of length :",len(rows_spam))
            rdd2 = sc.parallelize(rows_spam)
            #calling preprocessor
            preprocess_save(rows_spam, sc)
            global id_count
            id_count+=1
            print(id_count, "batch completed\n\n")
batches.foreachRDD(lambda rdd : process(rdd))
ssc.start()
ssc.awaitTermination()
