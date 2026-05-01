import json

notebook_path = "notebooks/01_training_smoke_and_tensorboard.ipynb"
with open(notebook_path, "r") as f:
    nb = json.load(f)

# The cell to modify is the one starting with "smoke_checkpoint = None"
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "smoke_checkpoint = None\n" in "".join(cell["source"]):
        cell["source"] = [
            "smoke_checkpoints = {}\n",
            "if RUN_SMOKE_TRAINING:\n",
            "    for stage in [\"ppo_baseline\", \"ppo_shaped\", \"ppo_curriculum\", \"ppo_selfplay\"]:\n",
            "        try:\n",
            "            smoke_checkpoints[stage] = run_training(\n",
            "                ctx,\n",
            "                stage=stage,\n",
            "                profile_name=PROFILE_NAME,\n",
            "                timesteps=SMOKE_TIMESTEPS,\n",
            "                smoke=True,\n",
            "                checkpoint_freq=1,\n",
            "                verbose=1,\n",
            "            )\n",
            "            print(f\"\\nInline progress summary for {stage}\")\n",
            "            show_training_snapshot(\n",
            "                ctx,\n",
            "                stage,\n",
            "                rows=10,\n",
            "                title=f\"{stage} training diagnostics\",\n",
            "            )\n",
            "        except Exception as exc:\n",
            "            print(f\"Smoke training failed for {stage}:\", type(exc).__name__, exc)\n",
            "else:\n",
            "    print(\"RUN_SMOKE_TRAINING=False. Looking for an existing smoke checkpoint.\")\n",
            "\n",
            "smoke_checkpoint = smoke_checkpoints.get(\"ppo_baseline\")\n",
            "if smoke_checkpoint is None:\n",
            "    try:\n",
            "        smoke_checkpoint = best_checkpoint(ctx, \"ppo_baseline\")\n",
            "        print(\"Using existing checkpoint:\", smoke_checkpoint)\n",
            "    except Exception as exc:\n",
            "        print(\"No checkpoint available yet:\", type(exc).__name__, exc)\n",
            "\n",
            "smoke_checkpoints\n"
        ]
        break

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
