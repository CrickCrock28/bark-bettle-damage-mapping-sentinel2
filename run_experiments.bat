@echo off
echo Inizio esecuzione esperimenti...

echo Esecuzione con config: config\experiments\unfiltered\256_05_05.yaml
python main.py --config config\experiments\unfiltered\256_05_05.yaml

echo Esecuzione con config: config\experiments\unfiltered\128_05_05.yaml
python main.py --config config\experiments\unfiltered\128_05_05.yaml

echo Tutti gli esperimenti sono stati completati.
pause
