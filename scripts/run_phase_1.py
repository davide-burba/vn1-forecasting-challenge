from pathlib import Path

import mlflow
from fire import Fire

from vn1.config import load_config, track_config_with_mlflow
from vn1.data_loading import load_data
from vn1.forecaster import Forecaster
from vn1.submission import track_submission_with_mlflow, validate_submission

PATH_RAW_DATA = Path("data/raw/")
PATH_EXAMPLE_SUBMISSION = PATH_RAW_DATA / "Submission Phase 1 - Random (3).csv"


def main(
    config_path: str = "config.yaml",
    skip_cross_validation: bool = False,
    dump_submission_locally: bool = False,
):
    mlflow.set_experiment("phase_1")
    with mlflow.start_run():
        print("load inputs")
        config = load_config(config_path)
        sales, price = load_data(phase=1, path=PATH_RAW_DATA)
        mlflow.log_artifact(config_path)
        track_config_with_mlflow(config)

        forecaster = Forecaster(config)

        if not skip_cross_validation:
            print("cross validate")
            forecaster.cross_validate(sales, price)

        print("build predictions")
        submission = forecaster.build_future_predictions(sales, price)

        print("sanity check and dump")
        validate_submission(submission, PATH_EXAMPLE_SUBMISSION)
        if dump_submission_locally:
            submission.to_csv("submission_phase_1.csv", index=False)
        track_submission_with_mlflow(submission)


if __name__ == "__main__":
    Fire(main)
