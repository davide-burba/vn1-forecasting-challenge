import numpy as np


def compute_competition_score(submission, objective):
    abs_err = np.nansum(abs(submission - objective))
    err = np.nansum((submission - objective))
    score = abs_err + abs(err)
    score /= objective.sum().sum()
    return score
