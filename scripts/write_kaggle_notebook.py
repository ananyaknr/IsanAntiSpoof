import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# IsanAntiSpoof Kaggle + DagsHub Experiment\n",
                "This notebook shows how to run the `IsanAntiSpoof` experiment on Kaggle and track it using DagsHub MLflow."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Required Libraries\n",
                "Install repository dependencies and import helper modules."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -r /kaggle/working/isan_antispoof/requirements.txt\n",
                "!pip install dagshub\n",
                "\n",
                "import os\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import json\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Dataset and Set Kaggle Paths\n",
                "Define the repository and Kaggle dataset paths. If you are using a Kaggle dataset, mount it here."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "root = Path('/kaggle/working/IsanAntiSpoof')\n",
                "print('Repo root:', root)\n",
                "print('Exists:', root.exists())\n",
                "print('Kaggle input directory: /kaggle/input')\n",
                "print('Contents of /kaggle/working:')\n",
                "print([p.name for p in Path('/kaggle/working').iterdir()])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Explore Dataset\n",
                "Inspect the repository and dataset layout before training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "if root.exists():\n",
                "    for path in sorted(root.glob('**/*'))[:50]:\n",
                "        print(path.relative_to(root))\n",
                "else:\n",
                "    print('Repo root not found. Clone the repo first.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Preprocess Audio and Labels\n",
                "Use the repository scripts to build protocol metadata and extract features from raw audio."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('If raw audio is available in data/raw, run these steps:')\n",
                "print('python src/data/build_protocol.py')\n",
                "print('python src/features/extract_all.py feature=lfcc')\n",
                "print('python src/features/extract_all.py feature=mfcc')\n",
                "print('python src/features/extract_all.py feature=cqcc')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Build the Anti-Spoofing Model\n",
                "This repository uses `src/training/train.py` and Hydra configuration to build GMM/LCNN/ResNet models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Training is launched with existing repo entrypoint:')\n",
                "print('python src/training/train.py experiment=e1_baseline')\n",
                "print('Or use model and feature overrides:')\n",
                "print('python src/training/train.py experiment=e4_isan_aware model=lcnn feature=mfcc')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Train the Model\n",
                "Run the experiment and send metrics to DagsHub MLflow. Replace the placeholders with your DagsHub org and token."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "os.environ['DAGSHUB_TOKEN'] = '<your-dagshub-token>'\n",
                "dagshub_org = '<your-org>'\n",
                "repo_name = 'IsanAntiSpoof'\n",
                "mlflow_uri = f'https://dagshub.com/{dagshub_org}/{repo_name}.mlflow'\n",
                "os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri\n",
                "print('MLflow URI:', mlflow_uri)\n",
                "\n",
                "%cd /kaggle/working/IsanAntiSpoof\n",
                "!python src/training/train.py experiment=e1_baseline mlflow.tracking_uri=$MLFLOW_TRACKING_URI mlflow.experiment_name=isan_antispoof\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Evaluate Model Performance\n",
                "Read the aggregated results file produced by the experiment logger."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "results_path = root / 'experiments' / 'results.csv'\n",
                "print('Results path:', results_path)\n",
                "if results_path.exists():\n",
                "    display(pd.read_csv(results_path).tail(10))\n",
                "else:\n",
                "    print('No results file found yet.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Save Model and Prepare Submission\n",
                "Locate trained checkpoints and save artifacts for download."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "checkpoints_dir = root / 'checkpoints'\n",
                "print('Checkpoint directory exists:', checkpoints_dir.exists())\n",
                "if checkpoints_dir.exists():\n",
                "    for path in sorted(checkpoints_dir.glob('**/*'))[:50]:\n",
                "        print(path.relative_to(root))\n",
                "else:\n",
                "    print('No checkpoints saved yet.')\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = Path('notebooks/kaggle_dagshub_experiment.ipynb')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)
print(f'Wrote notebook to {output_path.resolve()}')
