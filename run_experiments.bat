@echo off
echo Inizio esecuzione esperimenti...

echo Esecuzione con config: 128_03_07.yaml
python main.py --config config\128_03_07.yaml

echo Esecuzione con config: 128_04_06.yaml
python main.py --config config\128_04_06.yaml

echo Esecuzione con config: 256_03_07.yaml
python main.py --config config\256_03_07.yaml

echo Esecuzione con config: 256_04_06.yaml
python main.py --config config\256_04_06.yaml

echo Esecuzione con config: 512_03_07.yaml
python main.py --config config\512_03_07.yaml

echo Esecuzione con config: 512_04_06.yaml
python main.py --config config\512_04_06.yaml

echo Tutti gli esperimenti sono stati completati.
pause
