@echo off
set PYTHONPATH=%CD%;%CD%\objectsharer\;%CD%\pulseseq;%PYTHONPATH%
cd dataserver
python.exe dataserver.py
