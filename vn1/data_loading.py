from pathlib import Path

import pandas as pd


def load_data(phase, path):
    assert phase in {1, 2}
    path = Path(path)
    sales_0 = pd.read_csv(path / "Phase 0 - Sales.csv")
    price_0 = pd.read_csv(path / "Phase 0 - Price.csv")

    if phase == 1:
        return sales_0, price_0

    sales_1 = pd.read_csv(path / "Phase 1 - Sales.csv")
    price_1 = pd.read_csv(path / "Phase 1 - Price.csv")

    sales = pd.concat([sales_0, sales_1[sales_1.columns[3:]]], axis=1)
    price = pd.concat([price_0, price_1[price_1.columns[3:]]], axis=1)

    return sales, price
