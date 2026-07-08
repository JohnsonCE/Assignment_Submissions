import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# MODE TOGGLE — change this line to switch datasets
# ============================================================
# "LOCAL_100K" -> small dataset, local files, for fast/free testing
# "AWS_1M"     -> full dataset, S3 files, for the real EMR run
MODE = "LOCAL_100K"
# MODE = "AWS_1M"


if MODE == "LOCAL_100K":
    MOVIES_PATH = "../u.item"    
    RATINGS_PATH = "../u.data"
    MOVIES_SEP = "|"
    RATINGS_SEP = "\t"
elif MODE == "AWS_1M":
    MOVIES_PATH = "s3a://revature-369113522467-us-east-2-an/ml-1m/movies.dat"
    RATINGS_PATH = "s3a://revature-369113522467-us-east-2-an/ml-1m/ratings.dat"
    MOVIES_SEP = "::"
    RATINGS_SEP = "::"
else:
    raise ValueError(f"Unknown MODE: {MODE}")



spark = SparkSession.builder.appName("MovieSimilarities").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# ----------------------------
# Load movie names
# ----------------------------
print(f"Loading movie names ({MODE})...")

if len(MOVIES_SEP) == 1:
    # 100k delimiter
    movies = spark.read.csv(MOVIES_PATH, sep=MOVIES_SEP, header=False).select(
        F.col("_c0").cast("int").alias("movieId"),
        F.col("_c1").alias("title"),
    )
else:
    # 1M delimiter
    movies = spark.read.text(MOVIES_PATH).select(
        F.split(F.col("value"), MOVIES_SEP).getItem(0).cast("int").alias("movieId"),
        F.split(F.col("value"), MOVIES_SEP).getItem(1).alias("title"),
    )


# ----------------------------
# Load ratings
# ----------------------------
print(f"Loading ratings ({MODE})...")

if len(RATINGS_SEP) == 1:
    ratings = spark.read.csv(RATINGS_PATH, sep=RATINGS_SEP, header=False).select(
        F.col("_c0").cast("int").alias("userId"),
        F.col("_c1").cast("int").alias("movieId"),
        F.col("_c2").cast("double").alias("rating"),
    )
else:
    ratings = spark.read.text(RATINGS_PATH).select(
        F.split(F.col("value"), RATINGS_SEP).getItem(0).cast("int").alias("userId"),
        F.split(F.col("value"), RATINGS_SEP).getItem(1).cast("int").alias("movieId"),
        F.split(F.col("value"), RATINGS_SEP).getItem(2).cast("double").alias("rating"),
    )

ratings.cache() # Store this for multiple uses

# ----------------------------
# Build movie pairs (self-joining on userId, keeping movie1 < movie2 to remove duplicates)
# ----------------------------
r1 = ratings.alias("r1")
r2 = ratings.alias("r2")

moviePairs = (
    r1.join(r2, on="userId")
    .filter(F.col("r1.movieId") < F.col("r2.movieId"))
    .select(
        F.col("r1.movieId").alias("movie1"),
        F.col("r2.movieId").alias("movie2"),
        F.col("r1.rating").alias("rating1"),
        F.col("r2.rating").alias("rating2"),
    )
)

# ----------------------------
# Compute cosine similarity 
# ----------------------------
moviePairSimilarities = (
    moviePairs.groupBy("movie1", "movie2")
    .agg(
        F.sum(F.col("rating1") * F.col("rating1")).alias("sum_xx"),
        F.sum(F.col("rating2") * F.col("rating2")).alias("sum_yy"),
        F.sum(F.col("rating1") * F.col("rating2")).alias("sum_xy"),
        F.count(F.lit(1)).alias("numPairs"),
    )
    .withColumn(
        "score",
        F.when(
            (F.sqrt("sum_xx") * F.sqrt("sum_yy")) != 0,
            F.col("sum_xy") / (F.sqrt("sum_xx") * F.sqrt("sum_yy")),
        ).otherwise(F.lit(0.0)),
    )
    .select("movie1", "movie2", "score", "numPairs")
    .cache()
)


if MODE == "AWS_1M":
    moviePairSimilarities.write.mode("overwrite").parquet(
        "s3a://revature-369113522467-us-east-2-an/output/movie-sims"
    )
else:
    moviePairSimilarities.write.mode("overwrite").parquet("output/movie-sims")


# ----------------------------
# Query similar movies
# ----------------------------
if len(sys.argv) > 1:
    movieID = int(sys.argv[1])
    scoreThreshold = 0.97
    coOccurrenceThreshold = 50

    filtered = moviePairSimilarities.filter(
        ((F.col("movie1") == movieID) | (F.col("movie2") == movieID))
        & (F.col("score") > scoreThreshold)
        & (F.col("numPairs") > coOccurrenceThreshold)
    )

    results = (
        filtered.withColumn(
            "similarMovieId",
            F.when(F.col("movie1") == movieID, F.col("movie2")).otherwise(
                F.col("movie1")
            ),
        )
        .join(movies, F.col("similarMovieId") == movies.movieId)
        .orderBy(F.col("score").desc())
        .select("title", "score", "numPairs")
        .limit(10)
        .collect()
    )

    movieTitle = movies.filter(F.col("movieId") == movieID).first()["title"]
    print(f"\nTop 10 similar movies for: {movieTitle}")

    for row in results:
        print(row["title"], "\tscore:", row["score"], "\tstrength:", row["numPairs"])

spark.stop()


# RDD Version of MovieSim
"""
import sys
from pyspark import SparkConf, SparkContext
from math import sqrt


# ----------------------------
# Compute cosine similarity
# ----------------------------
def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0

    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    score = (sum_xy / denominator) if denominator else 0.0

    return (score, numPairs)


# ----------------------------
# Filter duplicate movie pairs
# ----------------------------
def filterDuplicates(userRatings):
    ratings = userRatings[1]
    movie1 = ratings[0][0]
    movie2 = ratings[1][0]
    return movie1 < movie2


# ----------------------------
# Make ((movie1, movie2), (r1, r2))
# ----------------------------
def makePairs(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))


# ----------------------------
# Load movie names from S3
# ----------------------------
def loadMovieNames(sc, path):
    lines = sc.textFile(path)

    return lines \
        .map(lambda line: line.split("::")) \
        .map(lambda x: (int(x[0]), x[1])) \
        .collectAsMap()


# ----------------------------
# Main Spark setup
# ----------------------------
conf = SparkConf()
sc = SparkContext(conf=conf)

sc.setLogLevel("WARN")

# ----------------------------
# S3 paths (CHANGE THIS)
# ----------------------------
MOVIES_PATH = "s3a://revature-369113522467-us-east-2-an/ml-1m/movies.dat"
RATINGS_PATH = "s3a://revature-369113522467-us-east-2-an/ml-1m/ratings.dat"


# ----------------------------
# Load and broadcast movie names
# ----------------------------
print("Loading movie names from S3...")
nameDict = sc.broadcast(loadMovieNames(sc, MOVIES_PATH))


# ----------------------------
# Load ratings from S3
# ----------------------------
print("Loading ratings from S3...")
data = sc.textFile(RATINGS_PATH)


ratings = data \
    .map(lambda l: l.split("::")) \
    .map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))


# ----------------------------
# Build movie pairs
# ----------------------------
ratingsPartitioned = ratings.partitionBy(100)

joinedRatings = ratingsPartitioned.join(ratingsPartitioned)

uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

moviePairs = uniqueJoinedRatings.map(makePairs).partitionBy(100)

moviePairRatings = moviePairs.groupByKey()


# ----------------------------
# Compute similarities
# ----------------------------
moviePairSimilarities = moviePairRatings \
    .mapValues(computeCosineSimilarity) \
    .persist()


# Optional: save full results
moviePairSimilarities.saveAsTextFile("s3a://revature-369113522467-us-east-2-an/output/movie-sims")


# ----------------------------
# Query similar movies
# ----------------------------
if len(sys.argv) > 1:

    movieID = int(sys.argv[1])

    scoreThreshold = 0.97
    coOccurrenceThreshold = 50

    filteredResults = moviePairSimilarities.filter(
        lambda pairSim:
            (pairSim[0][0] == movieID or pairSim[0][1] == movieID)
            and pairSim[1][0] > scoreThreshold
            and pairSim[1][1] > coOccurrenceThreshold
    )

    results = filteredResults \
        .map(lambda pairSim: (pairSim[1], pairSim[0])) \
        .sortByKey(ascending=False) \
        .take(10)

    print("\nTop 10 similar movies for:",
          nameDict.value[movieID])

    for sim, pair in results:
        similarMovieID = pair[0] if pair[0] != movieID else pair[1]

        print(
            nameDict.value[similarMovieID],
            "\tscore:", sim[0],
            "\tstrength:", sim[1]
        )


"""