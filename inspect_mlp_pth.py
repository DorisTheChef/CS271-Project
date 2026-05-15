import torch

path = "knockdown_mlp_model.pth"

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
            print("  first keys:", list(value.keys())[:10])
        else:
            print(key, type(value), value)
else:
    print(checkpoint)