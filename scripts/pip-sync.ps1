# Sync with requirements
pip-sync requirements-lock.txt
pip install -e .[dev,test,docs]