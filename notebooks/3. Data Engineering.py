# Databricks notebook source
# MAGIC %md 
# MAGIC # 3. Implementation: Exploration and DE

# COMMAND ----------

# MAGIC %md-sandbox ## 3.1 Data Engineering
# MAGIC <div style="float:right">
# MAGIC   <img src="https://github.com/alexxx-db/gnn-lvdr-pytorch/blob/main/images/step_1-2.png?raw=True" alt="graph-training" width="840px", />
# MAGIC </div>
# MAGIC
# MAGIC We begin by ingesting our streaming data using Autoloader and saving as a delta table. Additionally we read in CSV files from our internal teams and convert them to delta tables for more efficient querying.
# MAGIC

# COMMAND ----------

# MAGIC %run ./setup

# COMMAND ----------

# DBTITLE 1,Create notebook widgets for database name and dataset paths
dbutils.widgets.text(name="database_name", defaultValue="gnn_hls_graphsage_db", label="Database Name")
dbutils.widgets.text(name="data_path", defaultValue="gnn_hls_graphsage_data_path", label="FileStore Path")
dbutils.widgets.text(name="catalog_name", defaultValue="gnn_hls_graphsage", label="Catalog Name")

# COMMAND ----------

# DBTITLE 1,Unzip data and choose a database for analysis
# Get widget values
data_path = dbutils.widgets.get("data_path")
database_name = dbutils.widgets.get("database_name")
catalog_name = dbutils.widgets.get("catalog_name")

# Extract the zip files in the data directory into dbfs
get_datasets_from_git(data_path=dbutils.widgets.get("data_path"))

# Create a database and use that for our downstream analysis
_ = spark.sql(f"create catalog if not exists {catalog_name};")
_ = spark.sql(f"create database if not exists {catalog_name}.{database_name};")
_ = spark.sql(f"use {catalog_name}.{database_name};")

# COMMAND ----------

full_data_path = f"dbfs:/FileStore/{data_path}/"

# COMMAND ----------

# DBTITLE 1,Defining as streaming source (our streaming landing zone) and destination, a delta table called bronze_patient_provider_data
bronze_relation_data = spark.readStream\
                         .format("cloudFiles")\
                         .option("cloudFiles.format", "json")\
                         .option("cloudFiles.schemaLocation", full_data_path+"_bronze_schema_loc")\
                         .option("cloudFiles.inferColumnTypes", "true")\
                         .load(full_data_path + "stream_landing_location")

bronze_relation_data.writeStream\
                    .format("delta")\
                    .option("mergeSchema", "true")\
                    .option("checkpointLocation", full_data_path + "_checkpoint_bronze_stream")\
                    .trigger(once=True)\
                    .table("bronze_relation_data")\
                    .awaitTermination()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 3.1.1 Quickly viewing our Data
# MAGIC Let's print out the tables that we are going to be using throughout this blog. We will use these tables to generate our dashboard to view our provider network. The only table used during training of the GNN will be the streamed procurement information, the remaining tables will be used to generate the recommendations dashboard. Also note that edges in the provider network graph from _bronze_relation_data_ would be Patient (node) \\( \rightarrow  \\) Provider (node). Iteratively building these edge bunches reveals the entire provider network graph.

# COMMAND ----------

# DBTITLE 1,Bronze collected procurement data 
bronze_patient_data = spark.read.table("bronze_relation_data")
display(bronze_patient_data)

# COMMAND ----------

# DBTITLE 1,Read our tables and register them into our Database
# Read from CSV file and save as Delta
def read_and_write_to_db(table_name: str) -> None:
  (spark.read\
        .option("inferSchema", "true")\
        .option("header", "true")\
        .option("delimiter", ",")\
        .csv(f"dbfs:/FileStore/{data_path}/finance_data/{table_name}.csv")\
        .write.format("delta").mode("overwrite").saveAsTable(table_name))

read_and_write_to_db("patient_score_data")
read_and_write_to_db("provider_locations")
read_and_write_to_db("patient_score_frame")

# COMMAND ----------

# DBTITLE 1,Patient score premiums (please note this data is randomised for demo purposes)
display(spark.table("patient_score_data"))

# COMMAND ----------

# DBTITLE 1,Locations of patients
display(spark.table("patient_locations"))

# COMMAND ----------

# DBTITLE 1,Score factors associated to patients (again, these are randomised for demo purposes)
display(spark.table("patient_score_frame"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1.2 Some Quick Data Engineering
# MAGIC There are two pieces of data engineering required for this raw Bronze data:
# MAGIC 1. Our ingestion scheme provides a probability and we do not want to train our GNN on low probability links so we will remove below a threshold.
# MAGIC 1. We notice that raw provider names have postfixes denoting different legal entities but pointing to the same physical provider. 

# COMMAND ----------

# DBTITLE 1,1. We note that the data set includes low likelihood pairs which should not be used during training
draw_probability_distribution(bronze_company_data, probability_col='probability')

# COMMAND ----------

# DBTITLE 1,2. We can use the cleanco package to remove legal entity postscripts within our raw data
print(f"Cleaned example 1: {cleanco.basename('018f2b10979746da820c5269e4d87bb2 LLC')}")
print(f"Cleaned example 2: {cleanco.basename('018f2b10979746da820c5269e4d87bb2 Ltd.')}")

# COMMAND ----------

# DBTITLE 1,We register this logic as a UDF and apply both DE tasks to our collected bronze table
from pyspark.sql.types import StringType
import pyspark.sql.functions as F

clean_company_name = udf(lambda x: cleanco.basename(x), StringType())

silver_relation_data = spark.readStream\
                      .format("cloudFiles")\
                      .option("cloudFiles.inferColumnTypes", "true")\
                      .table("bronze_relation_data")\
                      .filter(F.col("probability") >= 0.55)\
                      .withColumn("Purchaser", clean_company_name(F.col("Purchaser")))\
                      .withColumn("Seller", clean_company_name(F.col("Purchaser")))

(silver_relation_data.writeStream\
 .format('delta')\
 .option("checkpointLocation", full_data_path+"_checkpoint_silver_relations")\
 .option('mergeSchema', 'true')\
 .trigger(once=True)\
 .table('silver_relation_data'))\
 .awaitTermination()

# COMMAND ----------

display(spark.table("silver_relation_data"))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will use this Silver data to train our GNN and refine the low likelihood links that we omitted going from Bronze to Silver. The GNN will be trained on relatively confident links in the next notebook.

# COMMAND ----------


