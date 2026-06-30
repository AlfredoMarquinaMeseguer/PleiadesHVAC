"""baseline: A Flower Baseline."""
from flwr.app import ArrayRecord
from flwr.serverapp.strategy.result import Result
from flwr.common import MetricRecord
from flwr.common import ArrayRecord
import numpy as np
import json



def exportable_metrics (metrics: dict[int, MetricRecord]) -> dict[int, dict[str, list[float] | list[int] | float | int]]:
    exportable_metrics = { k : metricrecord_to_normal(v) for k, v in metrics.items() }

    return exportable_metrics

def metricrecord_to_normal(metric_record: MetricRecord) -> dict[str, list[float] | list[int] | float | int]:
    return { k: metricvalue_to_normal(v) for k,v in metric_record.items() }

def metricvalue_to_normal(metric_value) -> list[float] | list[int] | float | int:
    if isinstance(metric_value, list) and all(isinstance(x, float) for x in metric_value):
            return [float(m) for m in metric_value]
    if isinstance(metric_value, list) and all(isinstance(x, int) for x in metric_value):
            return [int(m) for m in metric_value]
    if isinstance(metric_value, float):
            return float(metric_value)
    if isinstance(metric_value, int):
            return int(metric_value)

    raise ValueError(f"Unsupported metric type: {type(metric_value)}")

def save_result_to_json(result: Result, weigthed_metric : float ,filename: str) -> None:
    """Save the result of a federated learning run to a JSON file."""
    exportable_array = [i.tolist() for i in result.arrays.to_numpy_ndarrays()]

    metrics = {
        "train_metrics_clientapp": exportable_metrics(result.train_metrics_clientapp),
        "evaluate_metrics_clientapp": exportable_metrics(result.evaluate_metrics_clientapp),
        "evaluate_metrics_serverapp": exportable_metrics(result.evaluate_metrics_serverapp),
        "arrays": exportable_array,
        "weight_metric": weigthed_metric
    }

    json_result = json.dumps(metrics, indent=4)
    with open(filename, "w") as f:
        f.write(json_result)

    # result.train_metrics_clientapp,
    # result.evaluate_metrics_clientapp,
    # result.evaluate_metrics_serverapp

def load_result_from_json(filename: str) -> tuple[Result, float]:
    """Load the result of a federated learning run from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    arrays = [np.array(arr) for arr in data["arrays"]]
    train_metrics_clientapp = { int(k): MetricRecord(v) for k, v in data["train_metrics_clientapp"].items() }
    evaluate_metrics_clientapp = { int(k): MetricRecord(v) for k, v in data["evaluate_metrics_clientapp"].items() }
    evaluate_metrics_serverapp = { int(k): MetricRecord(v) for k, v in data["evaluate_metrics_serverapp"].items() }
    weigth_metric = data["weight_metric"]

    return Result(
        arrays=ArrayRecord(arrays),
        train_metrics_clientapp=train_metrics_clientapp,
        evaluate_metrics_clientapp=evaluate_metrics_clientapp,
        evaluate_metrics_serverapp=evaluate_metrics_serverapp,
    ), weigth_metric
