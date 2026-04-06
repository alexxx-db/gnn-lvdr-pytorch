# Databricks notebook source
# MAGIC %md-sandbox 
# MAGIC
# MAGIC <div style="float:right">
# MAGIC   <img src="https://github.com/alexxx-db/gnn-lvdr-pytorch/blob/main/images/step_3-3.png?raw=True" alt="graph-training" width="700px", />
# MAGIC </div>
# MAGIC
# MAGIC ## 5 Deploying our GNN to continuously refine our candidate edges
# MAGIC Now that we have tuned our GNN and graph design choices, we can deploy our GNN as a UDF for distributed inference using mlflow. During inference we expect inputs in the form of the silver table. We will query the GNN as to whether or not a low confidence link should indeed be added to the overall network. If the GNN provides a high likelihood for the link, then we will add the link to silver table thereby creating a gold table.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:right">
# MAGIC   <img src="https://github.com/grandintegrator/gnn-db-blogpost/blob/main/media/tsne_plot.png?raw=True" alt="graph-training" width="700px" height="400px", />
# MAGIC </div>
# MAGIC
# MAGIC The model registry allows us to version and stage models depending on their suitability for production. We can then inspect the model for the accuracy, loss, and other artifacts that will help decide whether or not to transition the model into production. For this model, we assess it's capability to distinguish between real and negatively sampled edges from our validation and testing graphs. Since this is a binary classification task, we can inspect the AUC values that we logged with the latest mlflow run. We see that the model has a **training AUC of 0.84, validation AUC of 0.93, and testing AUC of 0.95** which shoes good generalisation. We are also happy with the t-SNE plot and will therefore deploy the model to production.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %run ./setup

# COMMAND ----------

dbutils.widgets.text(name="catalog_name", defaultValue="gnn_hls_graphsage", label="Catalog Name")
dbutils.widgets.text(name="database_name", defaultValue="gnn_hls_graphsage_db", label="Database Name")

catalog_name = dbutils.widgets.get("catalog_name")
database_name = dbutils.widgets.get("database_name")

_ = spark.sql(f"use {catalog_name}.{database_name};")

# COMMAND ----------

# DBTITLE 1,For the sake of simplicity, we promote latest version to champion
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
model_name = f"{catalog_name}.{database_name}.patient_recommendations_gnn_model"
model_version = client.get_model_version_by_alias(model_name, "champion").version if False else \
    max(mv.version for mv in client.search_model_versions(f"name='{model_name}'"))

# Latest registered model
registered_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

# DBTITLE 1,Set champion alias on our registered model
#                                                    Model name       Alias         Version
#                                                         |              |              |
client.set_registered_model_alias(model_name, "champion", model_version)

# COMMAND ----------

# DBTITLE 1,Create a UDF from the champion version of our GNN model
get_gnn_prediction = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}@champion", env_manager="local")

# COMMAND ----------

# DBTITLE 1,Generate our gold table with inference included
from pyspark.sql.functions import col, struct

# Read in our silver table
silver_relation_table = spark.table('silver_relation_data')

# Create our gold table based on the GNN predictions
gold_table_with_pred = silver_relation_table.withColumn("gnn_prediction", get_gnn_prediction(struct(*silver_relation_table.columns)))

gold_table_with_pred.write.format("delta").mode("overwrite").saveAsTable('gold_relations_table_with_predictions')

# COMMAND ----------

# DBTITLE 1,Now we can see that there are indeed links that the GNN has not seen which had low confidence but the GNN scores highly!
# MAGIC %sql
# MAGIC select Purchaser, Seller, probability, gnn_prediction from gold_relations_table_with_predictions
# MAGIC where 1=1
# MAGIC   and gnn_prediction > probability
# MAGIC   and gnn_prediction > 0.8
# MAGIC   and probability < 0.6

# COMMAND ----------

# DBTITLE 1,Finally, update the gold table with the outputs from our GNN model - more links!
from pyspark.sql.functions import when, col

gold_with_predictions = spark.table("gold_relations_table_with_predictions")

# Change the probability column accoridng to the gnn outputs
gold_relations = gold_with_predictions\
                  .withColumn("probability",
                              when(col("gnn_prediction") >= col("probability"),
                                   col("gnn_prediction"))\
                              .otherwise(col("probability")))\
                  .drop(col("gnn_prediction"))

# Register as the final table for BI
gold_relations.write.format("delta").mode("overwrite").saveAsTable("gold_relations_table_refined")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 4.2 Now we can go ahead and create our patient recommendations Dashboard!
# MAGIC
# MAGIC <div style="float:right">
# MAGIC   <img src="https://github.com/alexxx-db/gnn-lvdr-pytorch/blob/main/images/dashboard-1.png?raw=True" alt="graph-training" width="1000px"", />
# MAGIC </div>
# MAGIC

# COMMAND ----------


