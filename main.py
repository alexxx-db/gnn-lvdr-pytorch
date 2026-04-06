import argparse
import logging
import sys
import yaml
from pathlib import Path
from warnings import simplefilter

from pyspark.sql import SparkSession

import torch
import mlflow.pytorch

# Ensure src/ is importable when running outside Databricks notebooks
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataset.dataloader import DataLoader
from managers.trainer import Trainer
from managers.evaluator import Evaluator
from utils import create_model, plot_tsne_embeddings


def main() -> None:
    """
    Main entry point for the project
    """
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=FutureWarning)
    spark = SparkSession.builder.getOrCreate()

    # Load run configuration settings from config file
    with open('config/config.yaml', 'r') as config_file:
        params = yaml.safe_load(config_file)

    # --------------------------------------------------------------------------
    # Unity Catalog namespace
    # --------------------------------------------------------------------------
    spark.sql(f"USE CATALOG {params['catalog']}")
    spark.sql(f"USE SCHEMA {params['schema']}")
    logging.info(f"Using {params['catalog']}.{params['schema']}")

    # --------------------------------------------------------------------------
    # Create dataset as well as dataloaders for training, validation and testing
    # --------------------------------------------------------------------------
    loader = DataLoader(params=params, spark=spark)
    data_loaders, graph_partitions, _ = loader.get_edge_dataloaders()

    # --------------------------------------------------------------------------
    # Create a graph model for training
    # --------------------------------------------------------------------------
    graph_model = create_model(params=params)

    # --------------------------------------------------------------------------
    # Start mlflow training run
    # --------------------------------------------------------------------------
    mlflow.set_registry_uri("databricks-uc")

    with mlflow.start_run(run_name='GNN-BLOG-MODEL') as run:
        # Log the parameters of the model run
        mlflow.log_params(params)
        mlflow.set_tag("link-prediction", "graphSAGE")

        # Train the GNN encoder along with the MLP link predictor
        logging.info('Starting training....')
        trainer = Trainer(params=params,
                          model=graph_model,
                          train_data_loader=data_loaders['training'],
                          training_graph=graph_partitions['training'])
        trained_model = trainer.train()

        # Log the model artefacts and parameters for the run
        trained_model.eval()
        with torch.no_grad():
            training_graph = graph_partitions['training']
            training_graph_embeddings = (
                trained_model.get_embeddings(g=training_graph,
                                             x=training_graph.ndata['feature'],
                                             batch_size=params['batch_size'],
                                             device=params['device'])
            )
        fig = plot_tsne_embeddings(graph_embeddings=training_graph_embeddings,
                                   chart_name='training_embeddings',
                                   save_fig=True)
        mlflow.log_artifact('data/training_embeddings.png')

        # --------------------------------------------------------------------------
        # Evaluate model accuracy on the testing split
        # --------------------------------------------------------------------------
        evaluator = Evaluator(params=params,
                              model=trained_model,
                              testing_data_loader=data_loaders['testing'])
        auc_list, ap_list = evaluator.evaluate()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("py4j").setLevel(logging.INFO)
    main()
