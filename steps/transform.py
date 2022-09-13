"""
This module defines the following routines used by the 'transform' step of the regression pipeline:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

import mlflow
import databricks.automl_runtime
from databricks import feature_store
from databricks.feature_store import FeatureLookup
from pandas import Timestamp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from databricks.automl_runtime.sklearn import DatetimeImputer
from databricks.automl_runtime.sklearn import TimestampTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

fs = feature_store.FeatureStoreClient()
target_col = "clicked"
feature_table="ad_kasey.online_session"


def calculate_features(df: DataFrame):
    online_feature_lookups = [
    FeatureLookup( 
      table_name = feature_table,
      feature_names = ["age", "num_purchases_last_6_months"],
      lookup_key = ["user_id" ],
      timestamp_lookup_key=["impression_timestamp"]
    )
    ]
    raw = spark.table("ad_kasey.inference_silver").select("user_id", "impression_timestamp", "session_id", "clicked")
    
    training_set = fs.create_training_set(raw,
    feature_lookups = online_feature_lookups,
    label = "clicked"
    )
    training_df = training_set.load_df()

    
    return training_df.toPandas()


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    imputers = {
      "impression_timestamp": DatetimeImputer(),
    }

    datetime_transformers = []

    for col in ["impression_timestamp"]:
        ohe_transformer = ColumnTransformer(
            [("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), [TimestampTransformer.HOUR_COLUMN_INDEX])],
            remainder="passthrough")
        timestamp_preprocessor = Pipeline([
            (f"impute_{col}", imputers[col]),
            (f"transform_{col}", TimestampTransformer()),
            (f"onehot_encode_{col}", ohe_transformer),
            (f"standardize_{col}", StandardScaler()),
        ])
        datetime_transformers.append((f"timestamp_{col}", timestamp_preprocessor, [col]))
        transformers = datetime_transformers

        preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)
        
        sklr_classifier = LogisticRegression(
          C=2.30556906175862,
          penalty="l1",
          solver="saga",
          random_state=523525670,
        )
    
    return Pipeline([
          ("column_selector", col_selector),
          ("preprocessor", preprocessor),
          ("classifier", sklr_classifier),
      ])
        
