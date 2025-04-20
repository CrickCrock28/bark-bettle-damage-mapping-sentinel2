@echo off

echo Esecuzione con config: config\preprocess\1.yaml
python main.py --config config\preprocess\1.yaml --preprocess

echo Esecuzione con config: config\preprocess\2.yaml
python main.py --config config\preprocess\2.yaml --preprocess

echo Esecuzione con config: config\preprocess\3.yaml
python main.py --config config\preprocess\3.yaml --preprocess

echo Esecuzione con config: config\preprocess\4.yaml
python main.py --config config\preprocess\4.yaml --preprocess

pause
