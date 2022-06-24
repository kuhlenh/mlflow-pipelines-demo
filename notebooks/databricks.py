# Databricks notebook source
from mlflow.pipelines import Pipeline
p = Pipeline(profile="databricks")

# COMMAND ----------

p.run("evaluate")
