import json
import datetime
from pathlib import Path

notebook_path = Path("training_classic.ipynb")
export_dir = Path("docs/source/notebooks/")

# Load notebook
with open(notebook_path, "r") as stream_json:
    notebook = json.load(stream_json)

# Get original title
title = notebook['cells'][0]['source'][0]

# Add date to title
export_time = datetime.datetime.now()
notebook['cells'][0]['source'][0] = f'{title} - {export_time.strftime("%Y %B %d")}'

# Add date to export path
export_name = f'{notebook_path.stem}_{export_time.strftime("%Y_%m_%d__%H%M%S")}.ipynb'
export_path = export_dir.joinpath(export_name)

# Export
with open(export_path, "w") as stream_json:
    json.dump(notebook, stream_json)
