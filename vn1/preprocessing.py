import pandas as pd

ID_COLS = ("Client", "Warehouse", "Product")
HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)


class Preprocessor:
    def __init__(self, static_feature_list, data_feature_eng_list, date_features):
        for col in static_feature_list:
            assert col.name in ID_COLS
        self.static_feature_map = {feat.name: feat for feat in static_feature_list}
        self.data_feature_eng_list = data_feature_eng_list
        self.date_features = date_features

    def prepare_data(self, sales, price):
        targets = self.prepare_targets(sales)
        features = self.prepare_features(sales, price)

        return targets, features

    def prepare_targets(self, sales):
        sales_ts = sales.set_index(list(ID_COLS))
        all_targets = {}
        for horizon in HORIZONS:
            df = (
                build_lagged_df(sales_ts, -horizon)
                .melt(ignore_index=False)
                .reset_index()
                .rename(columns={"variable": "time"})
            )
            df["time"] = pd.to_datetime(df["time"])
            all_targets[horizon] = df

        return all_targets

    def prepare_features(self, sales, price):
        all_features = []
        for data_feature_eng in self.data_feature_eng_list:
            # set time-series data on which to apply feature engineering
            data_name = self._build_data_name(
                data_feature_eng.source, data_feature_eng.groupby
            )
            match data_feature_eng.source:
                case "sales":
                    df_input = sales
                case "price":
                    df_input = price

            df_ts = self._group_timeseries(
                df_input,
                data_feature_eng.groupby,
                data_feature_eng.group_stat,
            )

            # engineer features
            for feat_eng in data_feature_eng.feature_eng_list:
                feat = self._apply_feat_eng(
                    feat_eng.kind, feat_eng.params, df_ts, data_name
                )
                all_features.append(feat)

        features = self._merge_all_features(all_features, sales)

        # Static features
        for col in ID_COLS:
            # drop if not listed
            if col not in self.static_feature_map:
                features = features.drop(columns=col)
            # convert to category if specified
            elif self.static_feature_map[col].categorical:
                features[col] = features[col].astype("category")

        # Date features
        for date_feat in self.date_features:
            match date_feat:
                case "year":
                    features["year"] = features["time"].dt.year
                case "month":
                    features["month"] = features["time"].dt.month
                case "day":
                    features["day"] = features["time"].dt.day
                case _:
                    raise ValueError(f"Unknown date feature: {date_feat}")

        return features.drop(columns=["time"])

    def _apply_feat_eng(self, kind, params, df_ts, data_name):
        match kind:
            case "lag":
                lag = params["lag"]
                feature_name = f"{data_name}_lag-{lag}"
                feat = build_lagged_df(df_ts, lag)
            case "rolling":
                window = params["window"]
                statistic = params["statistic"]
                feature_name = f"{data_name}_rolling-{statistic}_w{window}"
                feat = build_rolling_statistic_df(df_ts, window, statistic)
            case _:
                raise ValueError(f"Unknown feature engineering kind: {kind}")

        return self._format_engineered_feature(feat, feature_name)

    def _merge_all_features(self, all_features, sales):
        join_df = (
            sales.set_index(list(ID_COLS))
            .melt(ignore_index=False)
            .reset_index()
            .rename(columns={"variable": "time"})[list(ID_COLS) + ["time"]]
        )
        join_df["time"] = pd.to_datetime(join_df["time"])

        # merge all with join_df
        features_df = join_df.copy()
        for feat in all_features:
            for_join, join_on = self._prepare_for_join(feat)
            features_df = features_df.merge(for_join, on=join_on, how="left")

        return features_df

    def _group_timeseries(
        self,
        df: pd.DataFrame,
        groupby: list[str] | None = None,
        group_stat: str = "sum",
    ) -> pd.DataFrame:
        ts = df.copy()

        index_cols = groupby or list(ID_COLS)

        if groupby is not None:
            for col in groupby:
                assert col in ID_COLS

            cols_to_drop = [c for c in ID_COLS if c not in groupby]
            ts = ts.drop(columns=cols_to_drop)

            # ts = ts.groupby(groupby).sum().reset_index()
            ts = ts.groupby(groupby).agg(group_stat).reset_index()

        return ts.set_index(index_cols)

    def _prepare_for_join(self, df):
        join_on = list(df.index.names) + ["time"]
        df_for_join = df.reset_index()
        return df_for_join, join_on

    def _build_data_name(self, source, groupby):
        return f"{source}_{'' if groupby is None else '-'.join(groupby)}"

    def _format_engineered_feature(self, feat, feature_name):
        feat = feat.melt(ignore_index=False)
        feat = feat.rename(columns={"value": feature_name, "variable": "time"})
        feat["time"] = pd.to_datetime(feat["time"])
        return feat


def build_lagged_df(df, lag=0):
    return df.shift(lag, axis=1)


def build_rolling_statistic_df(df, window=4, statistic="mean"):
    return df.T.rolling(window).agg(statistic).T
