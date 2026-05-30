"""Train sparse baselines (4 feature types × LR and linear SVM) on shared splits."""

from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .evaluate import evaluate_labels, metrics_dict_to_jsonable
from .features import BASELINE_EXPERIMENT_METHODS, BaselineFeatureExtractor
from .models import Split

logger = logging.getLogger(__name__)


def nested_sparse_baseline_results(
    train: Split,
    val: Split,
    test: Split,
    *,
    seed: int,
) -> tuple[dict[str, dict[str, dict]], dict[str, object]]:
    """Fit sparse baselines on *train*, report metrics on train/val/test.

    Returns:
        (results, timing_meta)

        Each ``results[method][classifier]`` is either ``{"error": str}`` or::

            {
                "splits": {
                    "train": <metrics_dict>,
                    "validation": <metrics_dict>,
                    "test": <metrics_dict>,
                },
                "timing_seconds": {
                    "feature_extract_and_transform": float,
                    "classifier_fit_and_predict": float,
                    "row_total": float,
                },
            }

        ``timing_meta`` aggregates wall times for logging / ``metrics.json``::

            {"total_wall_seconds": float, "rows": { "bow/logreg": {...}, ... }}
    """
    y_train = np.asarray(train.labels)
    y_val = np.asarray(val.labels)
    y_test = np.asarray(test.labels)

    def _new_clf(kind: str):
        if kind == "logreg":
            return LogisticRegression(max_iter=1000, random_state=seed)
        return LinearSVC(max_iter=2000, random_state=seed)

    results: dict[str, dict[str, dict]] = {}
    row_meta: dict[str, dict[str, float]] = {}
    total_wall = 0.0

    for display_name, extractor_method in BASELINE_EXPERIMENT_METHODS:
        results[display_name] = {}
        try:
            t_feat0 = time.perf_counter()
            extractor = BaselineFeatureExtractor(
                method=extractor_method,
                random_seed=seed,
            )
            X_train = extractor.fit_transform(train.texts)
            X_val = extractor.transform(val.texts)
            X_test = extractor.transform(test.texts)
            feature_s = time.perf_counter() - t_feat0
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            logger.warning("Feature extraction failed for '%s': %s", display_name, exc)
            for clf_name in ("logreg", "svm"):
                results[display_name][clf_name] = {"error": err}
            continue

        method_wall = feature_s
        for clf_name in ("logreg", "svm"):
            key = f"{display_name}/{clf_name}"
            try:
                t_clf0 = time.perf_counter()
                clf = _new_clf(clf_name)
                clf.fit(X_train, train.labels)
                pred_tr = np.asarray(clf.predict(X_train))
                pred_va = np.asarray(clf.predict(X_val))
                pred_te = np.asarray(clf.predict(X_test))
                clf_s = time.perf_counter() - t_clf0

                m_tr = metrics_dict_to_jsonable(evaluate_labels(y_train, pred_tr))
                m_va = metrics_dict_to_jsonable(evaluate_labels(y_val, pred_va))
                m_te = metrics_dict_to_jsonable(evaluate_labels(y_test, pred_te))

                row_inc = feature_s + clf_s
                method_wall += clf_s
                row_meta[key] = {
                    "feature_extract_and_transform": feature_s,
                    "classifier_fit_and_predict": clf_s,
                    "row_total": row_inc,
                }

                results[display_name][clf_name] = {
                    "splits": {
                        "train": m_tr,
                        "validation": m_va,
                        "test": m_te,
                    },
                    "timing_seconds": row_meta[key],
                }
                logger.info(
                    "Baseline %-12s / %-6s — test acc=%.4f  f1_macro=%.4f  (row %.2fs)",
                    display_name,
                    clf_name,
                    m_te["accuracy"],
                    m_te["f1_macro"],
                    row_inc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Classifier '%s' failed for baseline '%s': %s",
                    clf_name,
                    display_name,
                    exc,
                )
                results[display_name][clf_name] = {"error": str(exc)}
        total_wall += method_wall

    timing_meta: dict[str, object] = {
        "total_wall_seconds": total_wall,
        "rows": row_meta,
    }
    return results, timing_meta
