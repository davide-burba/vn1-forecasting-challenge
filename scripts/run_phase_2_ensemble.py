import os
from pathlib import Path

import mlflow
from fire import Fire

from vn1.config import load_config, track_config_with_mlflow
from vn1.data_loading import load_data
from vn1.forecaster import Forecaster
from vn1.preprocessing import ID_COLS
from vn1.submission import track_submission_with_mlflow

PATH_RAW_DATA = Path("data/raw/")


def main(
    n_estimators: int = 10,
    config_path: str = "config.yaml",
    dump_submission_locally: bool = True,
):
    mlflow.set_experiment("phase_2_ensemble")

    with mlflow.start_run():
        print("load inputs")
        config = load_config(config_path)
        sales, price = load_data(phase=2, path=PATH_RAW_DATA)
        mlflow.log_artifact(config_path)
        track_config_with_mlflow(config)

        run_id = mlflow.active_run().info.run_id
        if dump_submission_locally:
            output_folder = Path(f"submissions/submission-ensemble-{run_id}")
            os.makedirs(output_folder, exist_ok=True)

        submissions = []
        for seed in range(n_estimators):
            config.engine_params["seed"] = seed
            forecaster = Forecaster(config)

            print("build predictions")
            submission = forecaster.build_future_predictions(sales, price)

            print("dump")
            if dump_submission_locally:
                submission.to_csv(
                    output_folder / f"submission_seed_{seed}.csv",
                    index=False,
                )
            submissions.append(submission)

        id_cols = list(ID_COLS)
        submission = sum([df.set_index(id_cols) for df in submissions]) / n_estimators
        submission = submission.reset_index()

        track_submission_with_mlflow(submission)
        if dump_submission_locally:
            submission.to_csv(
                output_folder / "submission_ensemble.csv",
                index=False,
            )


if __name__ == "__main__":
    Fire(main)
