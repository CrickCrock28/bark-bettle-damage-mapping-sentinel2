@echo off

echo Esecuzione con config: config\experiments\filtered_128_01_09.yaml
python main.py --config config\experiments\filtered_128_01_09.yaml --train

echo Esecuzione con config: config\experiments\filtered_128_02_08.yaml
python main.py --config config\experiments\filtered_128_02_08.yaml --train

echo Esecuzione con config: config\experiments\filtered_128_03_07.yaml
python main.py --config config\experiments\filtered_128_03_07.yaml --train

echo Esecuzione con config: config\experiments\filtered_128_04_06.yaml
python main.py --config config\experiments\filtered_128_04_06.yaml --train

echo Esecuzione con config: config\experiments\filtered_128_05_05.yaml
python main.py --config config\experiments\filtered_128_05_05.yaml --train

echo Esecuzione con config: config\experiments\filtered_256_01_09.yaml
python main.py --config config\experiments\filtered_256_01_09.yaml --train

echo Esecuzione con config: config\experiments\filtered_256_02_08.yaml
python main.py --config config\experiments\filtered_256_02_08.yaml --train

echo Esecuzione con config: config\experiments\filtered_256_03_07.yaml
python main.py --config config\experiments\filtered_256_03_07.yaml --train

echo Esecuzione con config: config\experiments\filtered_256_04_06.yaml
python main.py --config config\experiments\filtered_256_04_06.yaml --train

echo Esecuzione con config: config\experiments\filtered_256_05_05.yaml
python main.py --config config\experiments\filtered_256_05_05.yaml --train

echo Esecuzione con config: config\experiments\all_data_256_05_05.yaml
python main.py --config config\experiments\all_data_256_05_05.yaml --train

echo Esecuzione con config: config\experiments\all_data_128_05_05.yaml
python main.py --config config\experiments\all_data_128_05_05.yaml --train

pause
