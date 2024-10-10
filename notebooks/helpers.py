def build_data_feature_eng_list(
    sales_lags,
    sales_rolling_mean,
    sales_rolling_std,
    sales_rolling_max,
    sales_rolling_min,
    sales_by_warehouse_lags,
    sales_by_warehouse_rolling_mean,
    sales_by_warehouse_rolling_std,
    sales_by_warehouse_rolling_max,
    sales_by_warehouse_rolling_min,
    sales_by_client_lags,
    sales_by_client_rolling_mean,
    sales_by_client_rolling_std,
    sales_by_client_rolling_max,
    sales_by_client_rolling_min,
    sales_by_product_lags,
    sales_by_product_rolling_mean,
    sales_by_product_rolling_std,
    sales_by_product_rolling_max,
    sales_by_product_rolling_min,
):
    data_feature_eng_list = [
        {
            "source": "sales",
            "groupby": None,
            "feature_eng_list": build_feature_eng_list(
                lags=sales_lags,
                rolling_mean=sales_rolling_mean,
                rolling_std=sales_rolling_std,
                rolling_max=sales_rolling_max,
                rolling_min=sales_rolling_min,
            ),
        },
        {
            "source": "sales",
            "groupby": ["Warehouse"],
            "feature_eng_list": build_feature_eng_list(
                lags=sales_by_warehouse_lags,
                rolling_mean=sales_by_warehouse_rolling_mean,
                rolling_std=sales_by_warehouse_rolling_std,
                rolling_max=sales_by_warehouse_rolling_max,
                rolling_min=sales_by_warehouse_rolling_min,
            ),
        },
        {
            "source": "sales",
            "groupby": ["Client"],
            "feature_eng_list": build_feature_eng_list(
                lags=sales_by_client_lags,
                rolling_mean=sales_by_client_rolling_mean,
                rolling_std=sales_by_client_rolling_std,
                rolling_max=sales_by_client_rolling_max,
                rolling_min=sales_by_client_rolling_min,
            ),
        },
        {
            "source": "sales",
            "groupby": ["Product"],
            "feature_eng_list": build_feature_eng_list(
                lags=sales_by_product_lags,
                rolling_mean=sales_by_product_rolling_mean,
                rolling_std=sales_by_product_rolling_std,
                rolling_max=sales_by_product_rolling_max,
                rolling_min=sales_by_product_rolling_min,
            ),
        },
    ]
    return [v for v in data_feature_eng_list if v["feature_eng_list"]]


def build_feature_eng_list(
    lags,
    rolling_mean,
    rolling_std,
    rolling_max,
    rolling_min,
):
    return (
        [{"kind": "lag", "params": {"lag": l}} for l in lags]
        + [
            {"kind": "rolling", "params": {"window": w, "statistic": "mean"}}
            for w in rolling_mean
        ]
        + [
            {"kind": "rolling", "params": {"window": w, "statistic": "std"}}
            for w in rolling_std
        ]
        + [
            {"kind": "rolling", "params": {"window": w, "statistic": "max"}}
            for w in rolling_max
        ]
        + [
            {"kind": "rolling", "params": {"window": w, "statistic": "min"}}
            for w in rolling_min
        ]
    )
