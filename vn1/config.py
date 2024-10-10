from typing import Any, Literal

import mlflow
import yaml
from pydantic import BaseModel, ConfigDict


class FeatureEng(BaseModel):
    kind: str
    params: dict[str, Any]


class DataFeatureEng(BaseModel):
    source: Literal["sales", "price"]
    groupby: list[str] | None = None
    group_stat: Literal["sum", "mean", "std", "min", "max"] = "sum"
    feature_eng_list: list[FeatureEng]

    model_config = ConfigDict(extra="forbid")


class StaticFeature(BaseModel):
    name: str
    categorical: bool

    model_config = ConfigDict(extra="forbid")


class PreprocessingConfig(BaseModel):
    data_feature_eng_list: list[DataFeatureEng]
    static_feature_list: list[StaticFeature]
    date_features: list[Literal["year", "month", "day"]]
    normalize_price: bool = True
    normalize_sales: bool = True

    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    preprocessing_config: PreprocessingConfig
    engine_params: dict[str, Any]
    multi_horizon: bool = True
    include_horizon_feature: bool = True
    include_horizon_year: bool = False
    include_horizon_month: bool = False
    include_horizon_day: bool = False

    model_config = ConfigDict(extra="forbid")


def load_config(path):
    with open(path, "r") as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


def track_config_with_mlflow(config):
    # track the most important attributes of the config with mlfow
    mlflow.log_params(config.engine_params)
    mlflow.log_params(config.preprocessing_config.model_dump())
    mlflow.log_param("multi_horizon", config.multi_horizon)
    mlflow.log_param("include_horizon_feature", config.include_horizon_feature)
