from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, udtf
from pyspark.sql.types import StringType, IntegerType
import re


spark = SparkSession.builder \
    .appName('HashTaxextractor') \
    .getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

data = [( "Learning #AI with #ML",), ("Explore #DataScience",), ("No hashs here",)]
df = spark.createDataFrame(data, ["text"])  



@udf(returnType=StringType())
def count_hashtags(text: str):
    if text:
        return len(re.findall(r"#\w+", text))
    

@udtf(returnType="hashtag: string")
class HashtagExtractor:
    def eval(self, text: str):
        if text:
            hashtags = re.findall(r"#\w+", text)
            for hashtag in hashtags:
                yield (hashtag,) # 'Yield' over a database session, once a connection is done with it then it gives it back
                # Essesntially, take this db session, use it, then when you're done, give it back

spark.udf.register('count_hashtags', count_hashtags)
spark.udtf.register('HashtagExtractor', HashtagExtractor)


spark.sql("SELECT count_hashtags('Welcome to #ApacheSpark and #BigData') AS hashtag_count").show()
df.selectExpr('text', 'count_hashtags(text) as num_hashtags').show()

spark.sql('SELECT * FROM HashtagExtractor("Welcome to #apache and #bigdata!")').show()

df.createOrReplaceTempView('tweets')

# Lateral allows one to access the text from the function it seems
spark.sql("SELECT text, hashtag FROM tweets, LATERAL HashtagExtractor(text)").show()




spark.stop()