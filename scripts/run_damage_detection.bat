@echo off

echo Esecuzione con config: config\damage_detection\euclidean.yaml
python main.py --config config\damage_detection\euclidean.yaml --damage_detection

echo Esecuzione con config: config\damage_detection\sam.yaml
python main.py --config config\damage_detection\sam.yaml --damage_detection

pause
