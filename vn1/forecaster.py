import os

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

from vn1.preprocessing import HORIZONS, ID_COLS, Preprocessor
from vn1.score import compute_competition_score


class Forecaster:
    def __init__(self, config):
        self.config = config

        self._model = None
        self._price_scaler = None
        self._sales_norm_data = None

    def build_future_predictions(self, sales, price):
        print("preprocess")
        sales, price = sales.copy(), price.copy()
        sales, price = self._normalize_sales_and_price(sales, price)
        targets, features = self._prepare_targets_and_features(sales, price)

        if self.config.multi_horizon:
            future_predictions = self._predict_multi_horizon(targets, features)
        else:
            future_predictions = self._predict_by_horizon(targets, features)

        return self._inverse_normalize(future_predictions)

    def cross_validate(self, sales, price, n_folds=4, test_size=0.05):
        print("preprocess")
        sales, price = sales.copy(), price.copy()
        # slight leakage by using min/max of all data
        sales, price = self._normalize_sales_and_price(sales, price)
        targets, features = self._prepare_targets_and_features(sales, price)

        if self.config.multi_horizon:
            self._cross_validate_multi_horizon(features, targets, n_folds, test_size)
        else:
            self._cross_validate_by_horizon(features, targets, n_folds, test_size)

    def _predict_multi_horizon(
        self,
        targets: dict[int, pd.DataFrame],
        features: pd.DataFrame,
    ):
        print("Training multi horizon")
        # prepare data in x,y format
        multi_h_target, multi_h_features = self._prepare_data_multi_horizon(
            targets, features
        )

        # train model
        # Important: drop the target NA, otherwise treated as 0!
        print("Training model")
        mask_na = multi_h_target.value.isna()
        multi_h_target_train = multi_h_target[~mask_na].reset_index(drop=True)
        multi_h_features_train = multi_h_features[~mask_na].reset_index(drop=True)
        self._model = LGBMRegressor(**self.config.engine_params)
        self._model.fit(multi_h_features_train, multi_h_target_train.value)

        # inference
        print("Inference")
        last_timestep = multi_h_target.time.max()
        mask = multi_h_target.time == last_timestep
        inference_features = multi_h_features[mask].reset_index(drop=True)
        inference_predictions = (
            multi_h_target[mask].drop(columns=["value", "time"]).reset_index(drop=True)
        )

        inference_predictions["y_pred"] = self._predict_with_magic_multiplier(
            inference_features
        )

        # format output
        print("Formatting output")
        id_cols = list(ID_COLS)
        future_predictions = targets[1][targets[1].time == last_timestep][
            id_cols
        ].reset_index(drop=True)
        for horizon in HORIZONS:
            pred_horizon = inference_predictions[
                inference_predictions.horizon == horizon
            ].drop(columns=["horizon"])

            future_predictions = pd.merge(
                future_predictions,
                pred_horizon,
                on=id_cols,
                how="left",
            ).rename(columns={"y_pred": f"pred_{horizon}"})

        return self._format_inference(future_predictions, last_timestep)

    def _predict_with_magic_multiplier(self, features: pd.DataFrame):
        return self._model.predict(features) * self.config.magic_multiplier

    def _prepare_data_multi_horizon(self, targets, features):
        # prepare data in x,y format
        multi_h_features_list = []
        multi_h_target_list = []
        for horizon in HORIZONS:
            # build target
            targets_h = targets[horizon].copy()
            targets_h["horizon"] = horizon
            multi_h_target_list.append(targets_h)

            # build features
            features_h = features.copy()

            if self.config.include_horizon_feature:
                features_h["horizon"] = horizon

            horizon_date = targets_h.time + pd.offsets.Week(horizon)
            if self.config.include_horizon_year:
                features_h["horizon_year"] = horizon_date.dt.year
            if self.config.include_horizon_month:
                features_h["horizon_month"] = horizon_date.dt.month
            if self.config.include_horizon_day:
                features_h["horizon_day"] = horizon_date.dt.day

            multi_h_features_list.append(features_h)

        multi_h_target = pd.concat(multi_h_target_list).reset_index(drop=True)
        multi_h_features = pd.concat(multi_h_features_list).reset_index(drop=True)

        return multi_h_target, multi_h_features

    def _predict_by_horizon(
        self,
        targets: dict[int, pd.DataFrame],
        features: pd.DataFrame,
    ):
        print("Training by horizon")

        last_timestep = targets[1].time.max()
        mask = targets[1].time == last_timestep
        id_cols = list(ID_COLS)
        future_predictions = targets[1][mask][id_cols].reset_index(drop=True)

        for horizon in HORIZONS:
            self._model = LGBMRegressor(**self.config.engine_params)
            self._model.fit(features, targets[horizon].value)

            mask = targets[horizon].time == last_timestep
            inference_features = features[mask].reset_index(drop=True)
            inference_targets = (
                targets[horizon][mask]
                .reset_index(drop=True)
                .drop(columns=["value", "time"])
            )
            inference_targets["y_pred"] = self._predict_with_magic_multiplier(
                inference_features
            )
            future_predictions = pd.merge(
                future_predictions,
                inference_targets,
                on=id_cols,
                how="left",
            ).rename(columns={"y_pred": f"pred_{horizon}"})

        return self._format_inference(future_predictions, last_timestep)

    def _format_inference(self, future_predictions, last_timestep):
        return future_predictions.rename(
            columns={
                f"pred_{h}": (
                    pd.Timestamp(last_timestep) + pd.offsets.Week(h)
                ).strftime("%Y-%m-%d")
                for h in HORIZONS
            }
        ).reset_index(drop=True)

    def _cross_validate_multi_horizon(self, features, targets, n_folds, test_size):
        print("CV multi horizon")
        # prepare data in x,y format
        multi_h_target, multi_h_features = self._prepare_data_multi_horizon(
            targets, features
        )
        # Important: drop the target NA, otherwise treated as 0!
        mask_na = multi_h_target.value.isna()
        multi_h_target = multi_h_target[~mask_na].reset_index(drop=True)
        multi_h_features = multi_h_features[~mask_na].reset_index(drop=True)

        # Time-based cross-validation
        end_quantile = 1
        delta_quantile = test_size
        scores = []
        for fold in range(n_folds):
            # Split data by time.
            # Not exactly equivalent as by-horizon split, the most recent fold
            # will have more short and less far horizons.
            # Probably negligible.
            end_timestep = multi_h_target.time.quantile(end_quantile)
            start_timestep = multi_h_target.time.quantile(end_quantile - delta_quantile)
            print(f"Fold {fold} start: {start_timestep}, end: {end_timestep}")

            mask_train = multi_h_target.time <= start_timestep
            mask_test = (multi_h_target.time > start_timestep) & (
                multi_h_target.time <= end_timestep
            )
            x_train = multi_h_features[mask_train]
            y_train = multi_h_target[mask_train]
            x_test = multi_h_features[mask_test]
            y_test = multi_h_target[mask_test].reset_index(drop=True)

            # Train model.
            self._model = LGBMRegressor(**self.config.engine_params)
            self._model.fit(x_train, y_train.value)
            y_pred = self._predict_with_magic_multiplier(x_test)

            # Inverse normalization by time-series.
            y_pred, y_test = self._inverse_normalize_cv(y_pred, y_test)

            # Evaluate.
            score = compute_competition_score(y_pred, y_test)
            print(f"Fold {fold} score: {score}")
            mlflow.log_metric(f"score_fold_{fold}", score)
            log_cv_predictions(y_pred, y_test, fold)
            end_quantile -= delta_quantile
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        print(f"Avg score: {avg_score}")
        mlflow.log_metric("score_avg", avg_score)

    def _cross_validate_by_horizon(self, features, targets, n_folds, test_size):
        end_quantile = 1
        delta_quantile = test_size
        features = features.reset_index(drop=True)
        scores = []
        for fold in range(n_folds):
            fold_scores = []
            for horizon in HORIZONS:
                targets_h = targets[horizon].reset_index(drop=True)

                # split train/test by time
                timesteps_no_na = targets_h.dropna().time
                end_timestep = timesteps_no_na.quantile(end_quantile)
                start_timestep = timesteps_no_na.quantile(end_quantile - delta_quantile)

                mask_train = targets_h.time <= start_timestep
                mask_test = (targets_h.time > start_timestep) & (
                    targets_h.time <= end_timestep
                )
                x_train = features[mask_train]
                y_train = targets_h[mask_train]
                x_test = features[mask_test]
                y_test = targets_h[mask_test].reset_index(drop=True)

                # Train model.
                self._model = LGBMRegressor(**self.config.engine_params)
                self._model.fit(x_train, y_train.value)
                y_pred = self._predict_with_magic_multiplier(x_test)

                # Inverse normalization by time-series.
                y_pred, y_test = self._inverse_normalize_cv(y_pred, y_test)

                score = compute_competition_score(y_pred, y_test)
                mlflow.log_metric(f"score_horizon_{horizon}_fold_{fold}", score)
                scores.append(score)
                fold_scores.append(score)

            avg_fold_score = sum(fold_scores) / len(fold_scores)
            mlflow.log_metric(f"score_fold_{fold}", avg_fold_score)

            end_quantile -= delta_quantile
        avg_score = sum(scores) / len(scores)
        mlflow.log_metric("score_avg", avg_score)

    def _normalize_sales_and_price(self, sales, price):
        if self.config.preprocessing_config.normalize_price:
            self._price_scaler = MinMaxScaler()
            price[price.columns[3:]] = self._price_scaler.fit_transform(
                price[price.columns[3:]].T
            ).T

        if self.config.preprocessing_config.normalize_sales:
            # Manual normalization, makes it easier to use for cross-validation.
            sales_norm_data = sales[sales.columns[:3]].copy()
            sales_norm_data["min"] = sales[sales.columns[3:]].T.min()
            sales_norm_data["max"] = sales[sales.columns[3:]].T.max()
            sales_norm_data["delta"] = sales_norm_data["max"] - sales_norm_data["min"]

            sales[sales.columns[3:]] = (
                (sales[sales.columns[3:]].T - sales_norm_data["min"])
                / sales_norm_data["delta"]
            ).T.values

            self._sales_norm_data = sales_norm_data

        return sales, price

    def _inverse_normalize(self, predictions):
        if self.config.preprocessing_config.normalize_sales:
            pd.testing.assert_frame_equal(
                predictions[list(ID_COLS)], self._sales_norm_data[list(ID_COLS)]
            )
            predictions[predictions.columns[3:]] = (
                (
                    predictions[predictions.columns[3:]].T
                    * self._sales_norm_data["delta"]
                )
                + self._sales_norm_data["min"]
            ).T
        # clip negative values
        predictions[predictions < 0] = 0
        return predictions

    def _inverse_normalize_cv(self, y_pred, test_df):
        y_test = test_df["value"]
        if self.config.preprocessing_config.normalize_sales:
            test_norm_data = pd.merge(
                test_df,
                self._sales_norm_data,
                on=ID_COLS,
                how="left",
            )
            y_pred = y_pred * test_norm_data["delta"] + test_norm_data["min"]
            y_test = y_test * test_norm_data["delta"] + test_norm_data["min"]
        # clip negative values
        y_pred[y_pred < 0] = 0
        return y_pred, y_test

    def _prepare_targets_and_features(self, sales, price):
        preprocessor = Preprocessor(
            self.config.preprocessing_config.static_feature_list,
            self.config.preprocessing_config.data_feature_eng_list,
            self.config.preprocessing_config.date_features,
        )
        return preprocessor.prepare_data(sales, price)


def log_cv_predictions(y_pred, y_test, fold):
    run_id = mlflow.active_run().info.run_id
    path = f"/tmp/predictions_{run_id}_fold_{fold}.p"
    pd.to_pickle((y_pred, y_test), path)
    mlflow.log_artifact(path)
    os.remove(path)
