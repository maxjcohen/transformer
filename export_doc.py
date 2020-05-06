"""
Export doc
This script defines the export_notebook method.
It gets the original title of the notebook and adds date.
Finally it exports to export_path.
"""
import json
import datetime
from pathlib import Path
import argparse

def export_notebook(given_notebook_path: Path, export_dir: Path):
    # Load notebook
    with open(given_notebook_path, "r") as stream_json:
        notebook = json.load(stream_json)

    # Get original title
    title = notebook['cells'][0]['source'][0]

    # Add date to title
    export_time = datetime.datetime.now()
    notebook['cells'][0]['source'][0] = f'{title} - {export_time.strftime("%Y %B %d")}'

    # Add date to export path
    export_name = f'training_{export_time.strftime("%Y_%m_%d__%H%M%S")}.ipynb'
    the_export_path = export_dir.joinpath(export_name)

    # Export
    with open(the_export_path, "w") as stream_json:
        json.dump(notebook, stream_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export notebook to docs.')

    parser.add_argument('notebook')
    parser.add_argument('-o', '--output-dir', )

    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    export_path = Path(args.output_dir or "docs/source/notebooks/trainings/")

    export_notebook(notebook_path, export_path)
