@echo off
echo Inizio esecuzione esperimenti...

echo Esecuzione con config: config\experiments\128_03_07.yaml
python main.py --config config\experiments\128_03_07.yaml

echo Esecuzione con config: config\experiments\256_03_07.yaml
python main.py --config config\experiments\256_03_07.yaml

echo Esecuzione con config: config\experiments\128_02_08.yaml
python main.py --config config\experiments\128_02_08.yaml

echo Esecuzione con config: config\experiments\256_02_08.yaml
python main.py --config config\experiments\256_02_08.yaml

echo Esecuzione con config: config\experiments\128_01_09.yaml
python main.py --config config\experiments\128_01_09.yaml

echo Esecuzione con config: config\experiments\256_01_09.yaml
python main.py --config config\experiments\256_01_09.yaml

echo Tutti gli esperimenti sono stati completati.
pause
