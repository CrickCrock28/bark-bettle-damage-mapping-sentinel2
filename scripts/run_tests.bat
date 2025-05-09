@echo off

echo Esecuzione con config: config\tests\filtered_128_05_05.yaml
python main.py --config config\tests\filtered_128_05_05.yaml --test

echo Esecuzione con config: config\tests\filtered_256_05_05.yaml
python main.py --config config\tests\filtered_256_05_05.yaml --test

echo Esecuzione con config: config\tests\all_data_128_05_05.yaml
python main.py --config config\tests\all_data_128_05_05.yaml --test

echo Esecuzione con config: config\tests\all_data_256_05_05.yaml
python main.py --config config\tests\all_data_256_05_05.yaml --test

pause
