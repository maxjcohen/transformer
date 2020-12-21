python setup.py sdist bdist_wheel
$version="0.4.1"
$files_to_handle_str="dist/time_series_transformer-$version*" 
twine check $files_to_handle_str
twine upload $files_to_handle_str