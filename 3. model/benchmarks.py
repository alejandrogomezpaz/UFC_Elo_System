import numpy as np
from scipy.integrate import quad
import matplotlib as plt

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
    nudge = 1e-10
    probabilities = np.clip(probabilities, nudge, 1 - nudge) #prevents log converging to literal infinity by slight nudge away from 0
    return -1 * np.mean(outcomes * np.log(probabilities) + (1 - outcomes) * np.log(1 - probabilities))

def brier_skill_score(outcomes, probabilities):
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

        true_pos_rate = true_pos / (true_pos + false_pos)
        false_pos_rate = true_neg / (true_neg + false_neg)

        tpr_points.append(true_pos_rate)
        fpr_points.append(false_pos_rate)

    area_under_curve = np.trapezoid(tpr_points, fpr_points)
    #AI generated plotting code; could not be bothered
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_points, tpr_points, label=f'Model (AUC = {auc:.3f})')
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

    edges = np.arange(0 , 100 + step, step)
    pred_points = []
    actual_points = []
    weights = []

    for i in range(len(edges) - 1):
        interval_accuracy = np.array()
        in_bin = [edges[i] <=  probabilities < edges[i + 1]]

        if len(in_bin) == 0:
            continue
        else:
            interval_accuracy.append(np.mean(in_bin))

        pred_points.append(probabilities[in_bin].mean()) # avg predicted prob in bin (x)
        actual_points.append(outcomes[in_bin].mean()) # win rate of fights in bin (y)
        weights.append(in_bin.sum())
    
    pred_points = np.array(pred_points)
    actual_points = np.array(actual_points)
    weights = np.array(weights)
    expected_calibration_error = np.sum(weights / weights.sum() * np.abs(actual_points - pred_points))

    #AI generated code for graph; couldn't be bothered :)
    plt.figure(figsize=(6, 6))
    plt.plot(pred_points, actual_points, 'o-', label=f'Model (ECE = {ece:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual win rate')
    plt.title('Reliability Curve')
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal')
    plt.show()

    return expected_calibration_error

def full_benchmark(outcomes, probabilities, step):
    print(log_loss(outcomes, probabilities))
    print(brier_skill_score(outcomes, probabilities))
    print(ranking_calibration(outcomes, probabilities))
    print(confidence_calibration(outcomes, probabilities, step))