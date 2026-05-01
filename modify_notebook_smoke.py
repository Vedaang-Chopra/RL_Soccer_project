import json

notebook_path = "notebooks/01_training_smoke_and_tensorboard.ipynb"
with open(notebook_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "smoke=True," in source_str and "RUN_SMOKE_TRAINING" in source_str:
            new_source = []
            for line in cell["source"]:
                if "smoke=True," in line:
                    new_source.append(line.replace("smoke=True,", "smoke=False,"))
                else:
                    new_source.append(line)
            cell["source"] = new_source
            break

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated to use smoke=False.")
