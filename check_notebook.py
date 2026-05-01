import json
notebook_path = "notebooks/01_training_smoke_and_tensorboard.ipynb"
with open(notebook_path, "r") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "RUN_SMOKE_TRAINING = True" in "".join(cell["source"]):
        print("".join(cell["source"]))
