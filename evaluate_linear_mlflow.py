import torch
import torch.nn as nn
import mlflow


class LinearPerturbationModel(nn.Module):
    def __init__(self, input_dim=3119, output_dim=3051):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


device = "cpu"

model_path = "Linear Model/linear_baseline_final.pt"
x_test_path = "data/processed/X_test.pt"
y_test_path = "data/processed/y_test.pt"

model = LinearPerturbationModel(input_dim=3119, output_dim=3051).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

X_test = torch.load(x_test_path, map_location=device)
y_test = torch.load(y_test_path, map_location=device)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

loss_fn = nn.MSELoss()

with torch.no_grad():
    predictions = model(X_test)
    test_loss = loss_fn(predictions, y_test).item()

print("Recomputed test loss:", test_loss)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("jurkat_perturbation_prediction")

with mlflow.start_run(run_name="linear_baseline_recomputed_test_loss"):
    mlflow.log_param("model_type", "linear")
    mlflow.log_param("input_dim", 3119)
    mlflow.log_param("output_dim", 3051)
    mlflow.log_param("test_samples", X_test.shape[0])
    mlflow.log_param("loss_function", "MSE")
    mlflow.log_param("evaluation_type", "recomputed_from_saved_test_set")

    mlflow.log_metric("test_loss", test_loss)

    mlflow.log_artifact(model_path, artifact_path="model_checkpoint")
    mlflow.log_artifact(x_test_path, artifact_path="test_data")
    mlflow.log_artifact(y_test_path, artifact_path="test_data")

print("Logged recomputed linear test loss to MLflow.")