@echo off
echo Inizio esecuzione esperimenti...

echo Esecuzione con config: config\interpolate\32x32_256_03_07.yaml
python main.py --config config\interpolate\32x32_256_03_07.yaml

echo Esecuzione con config: config\interpolate\64x64_256_03_07.yaml
python main.py --config config\interpolate\64x64_256_03_07.yaml

echo Esecuzione con config: config\no_resize\5x5_256_02_08.yaml
python main.py --config config\no_resize\5x5_256_02_08.yaml

echo Esecuzione con config: config\interpolate\32x32_256_04_06.yaml
python main.py --config config\interpolate\32x32_256_04_06.yaml

echo Esecuzione con config: config\interpolate\64x64_256_04_06.yaml
python main.py --config config\interpolate\64x64_256_04_06.yaml

echo Tutti gli esperimenti sono stati completati.
pause
