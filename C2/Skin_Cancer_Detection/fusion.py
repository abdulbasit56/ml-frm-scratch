import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def fuse(tree_probs, ann_probs, weight):
    return (weight * tree_probs) + ((1 - weight) * ann_probs)


if __name__ == "__main__":

    y_cv            = np.load("splits/y_cv.npy")
    y_test          = np.load("splits/y_test.npy")
    tree_probs_cv   = np.load("splits/tree_probs_cv.npy").flatten()
    tree_probs_test = np.load("splits/tree_probs_test.npy").flatten()
    ann_probs_cv    = np.load("splits/ann_probs_cv.npy").flatten()
    ann_probs_test  = np.load("splits/ann_probs_test.npy").flatten()


    # ---- Step 1: find the best weight + threshold using CV data only ----
    # tree is the stronger model, so it always gets the larger share
    # for each weight: take the highest threshold that still catches TARGET_RECALL of malignant cases,
    # then keep the weight that gives the fewest false alarms at that threshold
    TARGET_RECALL = 0.97
    thresholds = [round(t, 2) for t in np.arange(0.95, 0.04, -0.05)]

    best_weight    = None
    best_threshold = None
    best_fp        = np.inf

    for weight in [0.6, 0.7, 0.8, 0.9, 1.0]:
        fused_cv_probs = fuse(tree_probs_cv, ann_probs_cv, weight)
        score = roc_auc_score(y_cv, fused_cv_probs)

        for t in thresholds:
            cm = confusion_matrix(y_cv, np.where(fused_cv_probs > t, 1, 0))
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn)

            if recall >= TARGET_RECALL:
                break

        print(f"weight={weight}: AUC (CV) = {score:.4f}, threshold={t}, recall={recall:.3f}, missed cases={fn}, false alarms={fp}")

        if fp < best_fp:
            best_fp        = fp
            best_weight    = weight
            best_threshold = t

    print(f"chosen weight: {best_weight}, chosen threshold: {best_threshold}")

    # saved so app.py can reuse them later
    np.save("splits/fusion_weight.npy", np.array(best_weight))
    np.save("splits/fusion_threshold.npy", np.array(best_threshold))


    # ---- Step 2: apply that weight + threshold once, on test data, for the real result ----
    fused_test_probs = fuse(tree_probs_test, ann_probs_test, best_weight)
    fused_test_preds = np.where(fused_test_probs > best_threshold, 1, 0)

    # single models stay at 0.5 as the baseline to compare against
    tree_test_preds = np.where(tree_probs_test > 0.5, 1, 0)
    ann_test_preds  = np.where(ann_probs_test > 0.5, 1, 0)


    # ---- Step 3: the three-way comparison ----
    print(f"Tabular only : Accuracy = {accuracy_score(y_test, tree_test_preds):.4f}, AUC = {roc_auc_score(y_test, tree_probs_test):.4f}")
    print(f"Image only   : Accuracy = {accuracy_score(y_test, ann_test_preds):.4f}, AUC = {roc_auc_score(y_test, ann_probs_test):.4f}")
    print(f"Fused        : Accuracy = {accuracy_score(y_test, fused_test_preds):.4f}, AUC = {roc_auc_score(y_test, fused_test_probs):.4f}")


    # ---- Step 4: check missed malignant cases specifically ----
    for name, preds in [("Tabular", tree_test_preds), ("Image", ann_test_preds), ("Fused", fused_test_preds)]:
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        print(f"{name}: missed malignant cases = {fn}, false alarms = {fp}")
        print(cm)
