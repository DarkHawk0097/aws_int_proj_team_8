import json
import boto3

def lambda_handler(event, context):
    s3 = boto3.client("s3")
    bucket = event["bucket"]
    prediction_length = event.get("prediction_length", 14)

    forecast_obj = s3.get_object(Bucket=bucket, Key=event["forecast_key"])
    forecast_lines = [json.loads(l) for l in forecast_obj["Body"].read().decode("utf-8").splitlines()]

    actual_obj = s3.get_object(Bucket=bucket, Key=event["actual_key"])
    actual_lines = [json.loads(l) for l in actual_obj["Body"].read().decode("utf-8").splitlines()]

    mape_scores = []
    for forecast, actual in zip(forecast_lines, actual_lines):
        predicted_median = forecast["quantiles"]["0.5"]
        actual_last = actual["target"][-prediction_length:]
        errors = [abs(a - p) / a for a, p in zip(actual_last, predicted_median) if a != 0]
        if errors:
            mape_scores.append(sum(errors) / len(errors) * 100)

    overall_mape = sum(mape_scores) / len(mape_scores) if mape_scores else 0.0

    boto3.client("cloudwatch").put_metric_data(
        Namespace="P3-DemandForecasting",
        MetricData=[{"MetricName": "MAPE", "Value": overall_mape, "Unit": "Percent"}]
    )
    return {"overall_mape": overall_mape, "series_evaluated": len(mape_scores)}
