import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

'''
log-loss
    objective function to minimize (and converge upon optimal Tau value for Glicko-2)
brier score as benchmark to validate model signaling over guessing 
    different objective function (which handles residual-lengths less harshly than log-loss and reports overfitting)
    non_arbitrary number serves as benchmark
AUC ranking quality
    quantifies if rankings are credible based on binary random sampling
ECE + reliability curve
    diagnostic tool for seeing at what confidence intervals the model is overconfident and underconfident
'''

def log_loss(outcomes, probabilities):
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    nudge = 1e-10
    probabilities = np.clip(probabilities, nudge, 1 - nudge)
    return -np.mean(outcomes * np.log(probabilities) + (1 - outcomes) * np.log(1 - probabilities))

def brier_skill_score(outcomes, probabilities):
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    baseline_pred_rate = 0.5
    naive_model = np.mean((baseline_pred_rate - outcomes) ** 2)
    brier_model = np.mean((probabilities - outcomes) ** 2)
    return 1 - (brier_model / naive_model)

def ranking_calibration(outcomes, probabilities):

    actual = np.array(outcomes).astype(bool) #boolean array
    probabilities = np.array(probabilities) # for elementwise operations
    tpr_points = []
    fpr_points = []

    for cutoff in np.linspace(1, 0, 101):

        pred = np.array(probabilities > cutoff)
        true_pos = np.sum(pred & actual) #bitwise operators of booleans, ~means flip bool value
        false_pos = np.sum(pred & ~actual)
        true_neg = np.sum(~pred & ~actual)
        false_neg = np.sum(~pred & actual)

        true_pos_rate  = true_pos  / (true_pos  + false_neg) if (true_pos  + false_neg) else 0.0
        false_pos_rate = false_pos / (false_pos + true_neg)  if (false_pos + true_neg)  else 0.0

        tpr_points.append(true_pos_rate)
        fpr_points.append(false_pos_rate)

    area_under_curve = np.trapezoid(tpr_points, fpr_points)
    #AI generated plotting code; could not be bothered
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_points, tpr_points, label=f'Model (AUC = {area_under_curve:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Guessing (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.gca().set_aspect('equal')
    plt.show()

    return area_under_curve

def confidence_calibration(outcomes, probabilities, step):
    probabilities = np.array(probabilities)
    outcomes = np.array(outcomes)

    edges = np.arange(0, 1 + step, step)   # probabilities live in [0, 1]
    pred_points = []
    actual_points = []
    weights = []

    for i in range(len(edges) - 1):
        in_bin = (probabilities >= edges[i]) & (probabilities < edges[i + 1])

        if in_bin.sum() == 0:
            continue

        pred_points.append(probabilities[in_bin].mean())  # avg predicted prob in bin (x)
        actual_points.append(outcomes[in_bin].mean())     # win rate in bin (y)
        weights.append(in_bin.sum())

    pred_points = np.array(pred_points)
    actual_points = np.array(actual_points)
    weights = np.array(weights)
    expected_calibration_error = np.sum(weights / weights.sum() * np.abs(actual_points - pred_points))

    #AI generated plotting code below; couldn't be bothered :)
    plt.figure(figsize=(6, 6))
    plt.plot(pred_points, actual_points, 'o-', label=f'Model (ECE = {expected_calibration_error:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual win rate')
    plt.title('Reliability Curve')
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal')
    plt.show()

    return expected_calibration_error

def full_benchmark(outcomes, probabilities, step):
    print(f'log_loss: {log_loss(outcomes, probabilities)}')
    print(f'brier_skill_score: {brier_skill_score(outcomes, probabilities)}')
    print(f'ranking_calibration: {ranking_calibration(outcomes, probabilities)}')
    print(f'confidence_calibration: {confidence_calibration(outcomes, probabilities, step)}')