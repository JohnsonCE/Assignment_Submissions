# Import the SparkSession class, which is the entry point for working with Spark.
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    DecimalType, IntegerType, StructType, StructField, StringType, DateType, BooleanType, FloatType
)

# ---------------------------------------------------------------------------
# MODE toggle: "local" reads/writes files from this directory and skips all
# AWS/Glue/Iceberg setup. "aws" behaves like the original script.
# ---------------------------------------------------------------------------
MODE = "local"  # "local" or "aws"

# Local directory where orders.csv, products.csv, customers.csv live.
# Change this if your CSVs are somewhere other than the script's folder.
LOCAL_DATA_DIR = "."

S3_BUCKET = "s3://project1-369113522467-us-east-2-an"

# Create and configure a Spark session.
if MODE == "local":
    spark = (
        SparkSession.builder
        .appName("Ingestion-Local")
        .getOrCreate()
    )
else:
    spark = (
        SparkSession.builder

        # Set a name for the Spark application (shows up in Spark UI/logs).
        .appName("Ingestion")

        # Enable Apache Iceberg SQL extensions so Spark understands
        # Iceberg-specific SQL commands and table operations.
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
        )

        # Register a catalog named "glue_catalog".
        # Spark will use this catalog whenever tables are referenced with
        # the prefix "glue_catalog".
        .config(
            "spark.sql.catalog.glue_catalog",
            "org.apache.iceberg.spark.SparkCatalog"
        )

        # Tell Spark that this catalog should use AWS Glue
        # as the metadata store for Iceberg tables.
        .config(
            "spark.sql.catalog.glue_catalog.catalog-impl",
            "org.apache.iceberg.aws.glue.GlueCatalog"
        )

        # Specify the S3 warehouse location where Iceberg table data
        # and metadata files will be stored.
        .config(
            "spark.sql.catalog.glue_catalog.warehouse",
            f"{S3_BUCKET}/"
        )

        # Configure Iceberg to use the S3FileIO implementation
        # for reading and writing data in Amazon S3.
        .config(
            "spark.sql.catalog.glue_catalog.io-impl",
            "org.apache.iceberg.aws.s3.S3FileIO"
        )

        # Create the Spark session with all of the above settings.
        .getOrCreate()
    )

orders_schema = StructType([
    StructField('order_id', StringType()),
    StructField('customer_id', StringType()),
    StructField('product_id', StringType()),
    StructField('order_date', StringType()),
    StructField('ship_date', StringType()),
    StructField('quantity', StringType()),
    StructField('unit_price', StringType()),
    StructField('discount_pct', StringType()),
    StructField('total_amount', StringType()),
    StructField('payment_method', StringType()),
    StructField('order_status', StringType())
])

products_schema = StructType([
    StructField('product_id', StringType()),
    StructField('product_name', StringType()),
    StructField('category', StringType()),
    StructField('brand', StringType()),
    StructField('price', StringType()),
    StructField('cost', StringType()),
    StructField('stock_quantity', StringType()),
    StructField('weight_kg', StringType()),
    StructField('created_date', StringType()),
    StructField('is_active', StringType())
])

customers_schema = StructType([
    StructField('customer_id', StringType()),
    StructField('first_name', StringType()),
    StructField('last_name', StringType()),
    StructField('email', StringType()),
    StructField('phone', StringType()),
    StructField('signup_date', StringType()),
    StructField('country', StringType()),
    StructField('state', StringType()),
    StructField('postal_code', StringType()),
    StructField('is_active', StringType()),
    StructField('loyalty_points', StringType()),
])

# Resolve source paths based on MODE.
if MODE == "local":
    orders_path = f"{LOCAL_DATA_DIR}/orders.csv"
    products_path = f"{LOCAL_DATA_DIR}/products.csv"
    customers_path = f"{LOCAL_DATA_DIR}/customers.csv"
else:
    orders_path = f"{S3_BUCKET}/orders.csv"
    products_path = f"{S3_BUCKET}/products.csv"
    customers_path = f"{S3_BUCKET}/customers.csv"

orders_df = spark.read.options(
    header=True
).schema(orders_schema).csv(orders_path)

products_df = spark.read.options(
    header=True
).schema(products_schema).csv(products_path)

customers_df = spark.read.options(
    header=True,
).schema(customers_schema).csv(customers_path)

# Display the DataFrame's schema (column names and data types)
# to verify the data was loaded correctly.
orders_df.printSchema()
products_df.printSchema()
customers_df.printSchema()

if MODE == "aws":
    # Create an Iceberg database (namespace) in AWS Glue if it
    # doesn't already exist.
    spark.sql("""
    CREATE DATABASE IF NOT EXISTS glue_catalog.iceberg_catalog_db
    """)



# ----------------------------------------------------
# Cleaning customers





# Drop duplicates
cleaned_cx = customers_df.dropDuplicates()

# Using filter and rlike to find the records that match numbers, 
# the use of filter means that things that don't match the regex will be filtered out
cleaned_cx = customers_df.filter(F.col("customer_id").rlike("^[0-9]+$"))

# Using withColumn to access the column and then create a new column (maybe?), casting the old column data to be integers
# This will then have values that were not integers become null 
cleaned_cx = customers_df.withColumn("customer_id", F.col("customer_id").cast(IntegerType())) 

#Removing whitespace
cleaned_cx = customers_df.withColumn("first_name", F.trim(F.col("first_name")))
cleaned_cx = customers_df.withColumn("last_name", F.trim(F.col("last_name"))) 

# Formatting the email with regex
cleaned_cx = customers_df.withColumn("email", 
        F.when(F.lower(F.col("email")).rlike("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"), 
               F.lower(F.trim(F.col("email"))))
         .otherwise(None)
    ) 

# Checking that the phone attribute/column values contains numbers
cleaned_cx = customers_df.withColumn("phone_clean", F.regexp_replace(F.col("phone"), "[^0-9]", "")) 
# Formatting all the phone number records
cleaned_cx = cleaned_cx.withColumn("phone", 
        F.when(F.length(F.col("phone_clean")) == 10, 
               F.concat_ws("-", 
                   F.substring(F.col("phone_clean"), 1, 3), 
                   F.substring(F.col("phone_clean"), 4, 3), 
                   F.substring(F.col("phone_clean"), 7, 4)))
         .otherwise(None)
    ).drop("phone_clean") 

# Converting to date value
cleaned_cx = customers_df.withColumn("signup_date", F.to_date(F.col("signup_date"), "yyyy-MM-dd")) 

# If the value is any of the items in the list, change it to "US"
cleaned_cx = customers_df.withColumn("country", 
        F.when(F.upper(F.col("country")).isin("USA", "UNITED STATES", "US", "U.S.A."), "US")
         .otherwise(F.upper(F.trim(F.col("country"))))
    ) 

# Trim state column
cleaned_cx = customers_df.withColumn("state", F.upper(F.trim(F.col("state")))) 

# If values in "is_active" are like yes, 1, y, then becomes True, similar to False
cleaned_cx = customers_df.withColumn("is_active", 
        F.when(F.lower(F.col("is_active")).isin("true", "yes", "1", "y"), True)
         .when(F.lower(F.col("is_active")).isin("false", "no", "0", "n"), False)
         .otherwise(None)
    ) 

# Convert loyalty points to int, create loyalty points raw
cleaned_cx = customers_df.withColumn("loyalty_points_raw", F.col("loyalty_points").cast(IntegerType())) 

# Remove outliers 
cleaned_cx = cleaned_cx.withColumn("loyalty_points", 
        F.when((F.col("loyalty_points_raw") >= 0) & (F.col("loyalty_points_raw") < 100000), 
               F.col("loyalty_points_raw"))
         .otherwise(None)
    ).drop("loyalty_points_raw")




# ----------------------------------------------------
# Cleaning products



# Drop duplicates
cleaned_prod = products_df.dropDuplicates() 

# Check if matches numbers
cleaned_prod = products_df.filter(F.col("product_id").rlike("^P[0-9]+$")) 

# Trim qutoes away
cleaned_prod = products_df.withColumn("product_name", F.regexp_replace(F.trim(F.col("product_name")), '""', '')) 

# Trim whitespace
cleaned_prod = products_df.withColumn("category", F.initcap(F.trim(F.col("category")))) 

# Trim whitespace
cleaned_prod = products_df.withColumn("brand", F.initcap(F.trim(F.col("brand")))) 

# Cleaning price, removing $, ',', '.', created temporary column: price_clean
cleaned_prod = products_df.withColumn("price_clean", 
        F.regexp_replace(F.regexp_replace(F.col("price"), "\\$", ""), ",", ".")
    ) 
# Convert to decimal, use and drop price_clean
cleaned_prod = cleaned_prod.withColumn("price", 
        F.when(F.col("price_clean").cast(DecimalType(10, 2)) >= 0, 
               F.col("price_clean").cast(DecimalType(10, 2)))
         .otherwise(None)
    ).drop("price_clean") 

# Cast to decimal
cleaned_prod = products_df.withColumn("cost", F.col("cost").cast(DecimalType(10, 2))) 

# Convert to int
cleaned_prod = products_df.withColumn("stock_quantity", 
        F.when(F.col("stock_quantity").cast(IntegerType()) >= 0, 
               F.col("stock_quantity").cast(IntegerType()))
         .otherwise(0)
    ) 

# Convert to decimal
cleaned_prod = products_df.withColumn("weight_kg", F.col("weight_kg").cast(DecimalType(8, 2))) 

# Convert to date, follow the format
cleaned_prod = products_df.withColumn("created_date", F.to_date(F.col("created_date"), "yyyy-MM-dd")) 

# If values in "is_active" are like yes, 1, y, then becomes True, similar to False
cleaned_prod = products_df.withColumn("is_active", 
        F.when(F.lower(F.col("is_active")).isin("true", "yes", "1", "y"), True)
         .when(F.lower(F.col("is_active")).isin("false", "no", "0", "n"), False)
         .otherwise(None)
    )


# ----------------------------------------------------
# Cleaning orders, enforcing referential integrity


# ~~~
# First pass for cleaning orders
# ~~~


# Drop duplicates
clean_orders_1 = orders_df.dropDuplicates() 

# Match that numbers are in the order_id
clean_orders_1 = orders_df.filter(F.col("order_id").rlike("^[0-9]+$")) 

# Cast to int
clean_orders_1 = orders_df.withColumn("order_id", F.col("order_id").cast(IntegerType())) 

# Cast to int
clean_orders_1 = orders_df.withColumn("customer_id", F.col("customer_id").cast(IntegerType())) 

# Cast to int
clean_orders_1 = orders_df.withColumn("quantity", F.col("quantity").cast(IntegerType())) 

# Check that quantity is positive
clean_orders_1 = orders_df.filter(F.col("quantity") > 0) 

# Format date, cast to date
clean_orders_1 = orders_df.withColumn("order_date", F.to_date(F.col("order_date"), "yyyy-MM-dd")) 

# Format date, cast to date
clean_orders_1 = orders_df.withColumn("ship_date", F.to_date(F.col("ship_date"), "yyyy-MM-dd")) 

# Check that ship date is not before order date, follows valid inputs
clean_orders_1 = orders_df.filter(F.col("ship_date").isNull() | (F.col("ship_date") >= F.col("order_date"))) 

# Cast to decimal, create discount_pct_clean
clean_orders_1 = orders_df.withColumn("discount_pct_clean", F.col("discount_pct").cast(DecimalType(5, 2))) 
# Ensure discounts are between 0 and 100, use and drop discount_pct_Clean 
clean_orders_1 = clean_orders_1.withColumn("discount_pct", 
        F.when((F.col("discount_pct_clean") >= 0) & (F.col("discount_pct_clean") <= 100), 
               F.col("discount_pct_clean"))
         .otherwise(0)
    ).drop("discount_pct_clean") 

# Make sure payment method is within the list, otherwise change to credit card
clean_orders_1 = orders_df.withColumn("payment_method", 
        F.when(F.lower(F.col("payment_method")).isin("visa", "mastercard", "credit card"), "Credit Card")
         .otherwise(F.initcap(F.trim(F.col("payment_method"))))
    ) 

# Trim column
clean_orders_1 = orders_df.withColumn("order_status", F.initcap(F.trim(F.col("order_status"))))


# ~~~
# Enforce Foreign Key Integrity (Keep only valid customer_ids and product_ids)
# ~~~


# Grab the cx ids, prod ids
valid_customer_ids = cleaned_cx.select("customer_id").distinct()
valid_product_ids = cleaned_prod.select("product_id").distinct()


# ~~~
# Final pass for cleaning orders
# ~~~


# Join on valid customer ids
clean_orders_final = clean_orders_1.join(valid_customer_ids, on="customer_id", how="inner") 

# Join on valid product ids
clean_orders_final = clean_orders_1.join(valid_product_ids, on="product_id", how="inner") 

# Join on corret price
clean_orders_final = clean_orders_1.join(cleaned_prod.select("product_id", "price"), on="product_id", how="left") 

# Unit price
clean_orders_final = clean_orders_final.withColumn("unit_price", F.col("price")) 

# Create total_amount from quantity, unit price, including the discount
# clean_orders_final = clean_orders_1.withColumn("total_amount", 
#         F.round(
#             F.col("quantity") * F.col("unit_price") * (1 - (F.col("discount_pct") / 100)), 2
#         )
#     ).drop("price")

clean_orders_final = clean_orders_1.withColumn(
    "total_amount", 
    F.round(
        F.col("quantity").cast("double") * 
        F.col("unit_price").cast("double") * 
        (F.lit(1.0) - (F.col("discount_pct").cast("double") / F.lit(100.0))), 
        2
    )
).drop("price")



if MODE == "local":
    # Just show the cleaned data locally so you can verify the transformations.
    print(f"Cleaned customer row count: {cleaned_cx.count()}")
    cleaned_cx.show(truncate=False)

    print(f"Cleaned product row count: {cleaned_prod.count()}")
    cleaned_prod.show(truncate=False)

    print(f"Cleaned orders row count: {clean_orders_final.count()}")
    clean_orders_final.show(truncate=False)
else:
    # Write the DataFrame as an Iceberg table.
    (
        cleaned_cx.writeTo(
            # Fully qualified table name:
            # catalog.database.table
            "glue_catalog.iceberg_catalog_db"
        )

        # Specify that the table format should be Apache Iceberg.
        .using("iceberg")

        # Create the table if it doesn't exist.
        # If it already exists, replace it with the new data.
        .createOrReplace()
    )

    # Query the newly created Iceberg table to verify that the
    # data was written successfully.
    spark.sql("""
    SELECT *
    FROM glue_catalog.iceberg_catalog_db
    -- LIMIT 10
    """).show()

# Stop the Spark session and release cluster resources.
spark.stop()
