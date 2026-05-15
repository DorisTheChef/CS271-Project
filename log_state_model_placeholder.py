import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("jurkat_perturbation_prediction")

with mlflow.start_run(run_name="state_model_placeholder"):
    mlflow.log_param("model_type", "state_model")
    mlflow.log_param("status", "future_work")
    mlflow.log_param("data_setup", "same_as_linear_hvg_3051")
    mlflow.log_param("input_dim", 3119)
    mlflow.log_param("output_dim", 3051)
    mlflow.log_param("num_genes", 3051)
    mlflow.log_param("num_perturbations", 68)
    mlflow.log_param("planned_evaluation_metric", "MSELoss")
    mlflow.log_param("planned_tracking", "MLflow")

    mlflow.set_tag("model_status", "not_yet_run")
    mlflow.set_tag("purpose", "placeholder_for_future_state_model_comparison")

print("State model placeholder logged to MLflow.")