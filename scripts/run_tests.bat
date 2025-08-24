@echo off

echo Esecuzione con config: config\tests\filtered_128_05_05.yaml
python main.py --config config\tests\filtered_128_05_05.yaml --test

echo Esecuzione con config: config\tests\filtered_256_05_05.yaml
python main.py --config config\tests\filtered_256_05_05.yaml --test

echo Esecuzione con config: config\tests\all_data_128_05_05.yaml
python main.py --config config\tests\all_data_128_05_05.yaml --test

echo Esecuzione con config: config\tests\all_data_256_05_05.yaml
python main.py --config config\tests\all_data_256_05_05.yaml --test

echo Esecuzione con config: config\experiments\filtered_128_01_09.yaml
python main.py --config config\experiments\filtered_128_01_09.yaml --test

echo Esecuzione con config: config\experiments\filtered_256_01_09.yaml
python main.py --config config\experiments\filtered_256_01_09.yaml --test

echo Esecuzione con config: config\experiments\all_data_128_02_08.yaml
python main.py --config config\experiments\all_data_128_02_08.yaml --test

echo Esecuzione con config: config\experiments\all_data_256_02_08.yaml
python main.py --config config\experiments\all_data_256_02_08.yaml --test

echo Esecuzione con config: config\experiments\filtered_128_03_07.yaml
python main.py --config config\experiments\filtered_128_03_07.yaml --test

echo Esecuzione con config: config\experiments\filtered_256_03_07.yaml
python main.py --config config\experiments\filtered_256_03_07.yaml --test

echo Esecuzione con config: config\experiments\all_data_128_04_06.yaml
python main.py --config config\experiments\all_data_128_04_06.yaml --test

echo Esecuzione con config: config\experiments\all_data_256_04_06.yaml
python main.py --config config\experiments\all_data_256_04_06.yaml --test

echo Esecuzione con config: config\experiments\filtered_128_05_05.yaml
python main.py --config config\experiments\filtered_128_05_05.yaml --test

echo Esecuzione con config: config\experiments\filtered_256_05_05.yaml
python main.py --config config\experiments\filtered_256_05_05.yaml --test

echo Esecuzione con config: config\experiments\all_data_128_01_09.yaml
python main.py --config config\experiments\all_data_128_01_09.yaml --test

echo Esecuzione con config: config\experiments\all_data_256_01_09.yaml
python main.py --config config\experiments\all_data_256_01_09.yaml --test

pause
