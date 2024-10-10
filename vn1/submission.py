import mlflow
import pandas as pd

from vn1.preprocessing import ID_COLS


def validate_submission(submission, example_sumbission_path):
    example_submission = pd.read_csv(example_sumbission_path)

    assert (submission.columns == example_submission.columns).all()

    base_cols = list(ID_COLS)
    pd.testing.assert_frame_equal(submission[base_cols], example_submission[base_cols])


def track_submission_with_mlflow(submission):
    run_id = mlflow.active_run().info.run_id
    path = f"/tmp/submission_{run_id}.csv"
    submission.to_csv(path, index=False)
    mlflow.log_artifact(path)
