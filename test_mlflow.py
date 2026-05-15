import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("test_experiment")

with mlflow.start_run(run_name="first_test_run"):
    mlflow.log_param("model_type", "test")
    mlflow.log_param("learning_rate", 0.001)

    mlflow.log_metric("train_loss", 0.50)
    mlflow.log_metric("val_loss", 0.42)
    mlflow.log_metric("test_loss", 0.45)

print("MLflow test finished.")