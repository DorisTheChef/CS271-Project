import torch

path = "Linear Model/linear_baseline_final.pt"

checkpoint = torch.load(path, map_location="cpu")

print("Type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys:")
    for key in checkpoint.keys():
        print("-", key)

    print("\nDetails:")
    for key, value in checkpoint.items():
        if hasattr(value, "shape"):
            print(key, value.shape)
        elif isinstance(value, dict):
            print(key, "is a dict with", len(value), "items")
            first_keys = list(value.keys())[:5]
            print("  first keys:", first_keys)
        else:
            print(key, type(value), value)
else:
    print(checkpoint)