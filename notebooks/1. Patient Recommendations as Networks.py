# Databricks notebook source
# MAGIC %md
# MAGIC # Deploying Graph Neural Networks for Patient Recommendations with Databricks

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Patient Recommendations as Networks
# MAGIC We live in an ever interconnected world. Nowhere is this more evident than our healthcare provider networks. Healthcare systems have become intricately linked and weaved together due to the macroeconomic environment. Healthcare providers, clinics, and physicians rely on each other through effective communication, resource sharing, referral networks, coordinated care, integrated health records, shared ethical standards, continuing education, quality control, efficient administrative handling, community outreach, and integrated mental health services, all to ensure timely, comprehensive, and ethical care for their patients. The relationships between these players form a dynamic complex network.  
# MAGIC
# MAGIC To provide relevant recommendations to patients, healthcare providers should model richly contextual networks considering multi-modal data integration; multi-relational graphs; time-dependencies; geospatial contexts; network analysis of patient pathways, service networks, physician collaboration networks; patient subgroups; patient feedback; cost and resource utilization; and include ethical and regulatory considerations. 
# MAGIC     
# MAGIC This solution accelerator focuses on building rich patient recommendation networks using the power of data and machine learning by showing how healthcare providers can have a real-time view of their patient, physician, clinic and insurance networks after having collected some representative data. We will develop a dashboard that healthcare provider teams can use to monitor their patient recommendations. We will assume that a provider has developed a way to gather this dataset either by using available datasets and web information ([see here for automated method using natural language processing and deep learning using web data]()). 
# MAGIC
# MAGIC Since patient data is inevitably going to be incomplete and change over time, we will also demonstrate training and productionising Graph Neural Networks (GNNs) to continuously monitor and refine the collected network data. The GNN will learn from connectivity patterns that have been gathered to find missing information in the form of links. Formally, in the graph representation learning literature, this prediction task is described as link prediction.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="text-align:center">
# MAGIC   <img src="https://github.com/alexxx-db/gnn-lvdr-pytorch/blob/main/images/patient-recommendation-network-graph.png?raw=True"  alt="graph-structured-data" style="height:600px;">
# MAGIC </div>
# MAGIC
# MAGIC <figcaption align="center"><b>Fig.1 Arrows between providers represent provider network and insurance relationships. Patients/subscribers typically see one layer of this network whilst there a web of other providers (and some potentially better positioned to provide healthcare to the patient in question) that are not visible. </b></figcaption>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2. Building Relevant Patient Recommendations
# MAGIC The first step to building relevant patient recommendatios is to have visibility over the provider network and insurance relationships themselves. Physical proximity is but one dimension to this. There are often provider networks qualified with a subscriber's health insurance a short distance further from a patient/subscriber's home address which provide better or more optimal services for that patient. Much regarding a patient's needs are the product of multiple relationships and factors best represented as a graph.
# MAGIC
# MAGIC For this solution accelerator, we're going to assume that a healthcare provider network has set up the pre-requisite architecture to obtain clinician, facility, and insurance relationships with some coefficient, say, a probability score. This data will be streamed to cloud storage and will be incrementally ingested ([```autoloader```](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html)). We then also incorporate static data from teams that have structured information about patient and provider profiles as well as recommendation scores to feed our executive level dashboard.
# MAGIC
# MAGIC <br>
# MAGIC <div style="text-align:center">
# MAGIC     <img src="https://github.com/alexxx-db/gnn-lvdr-pytorch/blob/main/images/architecture_including_ml.png?raw=True" width=1200px alt="graph-structured-data">
# MAGIC </div>
# MAGIC
# MAGIC <figcaption align="center"><b>Fig.2 Adopted architecture for this solution accelerator </b></figcaption>
# MAGIC
# MAGIC <br> 

# COMMAND ----------

# MAGIC %md ## Using GraphSAGE to perform relationship embedding
# MAGIC
# MAGIC GraphSAGE (Graph Sample and AggregatE) is a powerful graph embedding method that learns representations of nodes in the graph. It does so by sampling and aggregating features from each node's local neighborhood. However, it's important to note that in its original form, GraphSAGE primarily focuses on node embeddings rather than relationship or edge embeddings.
# MAGIC
# MAGIC GraphSAGE generates embeddings for a node by aggregating features from its neighbors in the graph. The aggregated features are then concatenated with the node's own features and passed through a non-linear transformation (usually a fully connected layer with a non-linear activation function). This process can be repeated for multiple layers, each time aggregating features from a larger neighborhood.
# MAGIC
# MAGIC In terms of relationships, GraphSAGE implicitly takes into account the relationships between nodes through the graph's structure. That is, the connections between nodes (edges) in the graph determine which nodes are considered neighbors and thus influence the aggregation process. However, GraphSAGE does not explicitly generate separate embeddings for the relationships (edges) in the graph.
# MAGIC
# MAGIC If you need to generate explicit relationship or edge embeddings, you might need to consider other techniques. For example, knowledge graph embedding methods like TransE, TransR, or RotatE are specifically designed to generate embeddings for both entities (nodes) and relationships (edges) in a graph. Alternatively, you could extend GraphSAGE or other graph neural network models to handle edge features or to generate edge embeddings, although this would require significant modifications to the model architecture and training procedure.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## How to tune the sampling and aggregation of features for GraphSAGE? What options do we have with GraphSAGE for each of these parameters?
# MAGIC
# MAGIC The GraphSAGE (Graph Sample and AggregatE) model provides a few key parameters that you can tune to adjust the sampling and aggregation of features. Here's a breakdown of those parameters:
# MAGIC
# MAGIC * Neighbor Sampling: GraphSAGE employs a neighbor sampling strategy, where you specify how many neighbors to sample at each layer. This is an important hyperparameter to tune. Choosing a larger number means considering more neighbors, which could potentially capture more relevant information but at the cost of increased computational complexity. Conversely, selecting a smaller number reduces computational load but might miss out on some relevant information.
# MAGIC
# MAGIC * Aggregator Functions: GraphSAGE offers several aggregator functions that you can choose from, depending on your specific use case. These include:
# MAGIC
# MAGIC * Mean Aggregator: This is the simplest form of aggregation, where the features of the neighboring nodes are simply averaged.
# MAGIC
# MAGIC * LSTM Aggregator: This uses an LSTM (Long Short-Term Memory) to aggregate the features. This can potentially capture more complex patterns among the neighbors but is more computationally expensive.
# MAGIC
# MAGIC * MaxPool Aggregator: This employs a max-pooling strategy over the features of the neighboring nodes.
# MAGIC
# MAGIC * MeanPool Aggregator: This utilizes an average-pooling strategy over the features of the neighboring nodes.
# MAGIC
# MAGIC * Dimensionality of the Output Embeddings: This is another hyperparameter to tune. A larger dimension might capture more information but could also lead to overfitting and higher computational costs.
# MAGIC
# MAGIC * Number of Layers: The depth of the GraphSAGE model can also be tuned. More layers allow the model to capture a larger context around each node but increase the computational complexity.
# MAGIC
# MAGIC * Learning Rate: The learning rate for the optimization process is another crucial parameter to tune.
