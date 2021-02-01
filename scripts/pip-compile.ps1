$lock_file = "requirements-lock.txt"
pip-compile setup.py --find-links=https://download.pytorch.org/whl/torch_stable.html --upgrade --generate-hashes --output-file=$lock_file
# Make sure produced lock file stays hidden
(get-item $lock_file).Attributes += 'Hidden'