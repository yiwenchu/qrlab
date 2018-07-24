@echo off
set PYTHONPATH=%CD%;%CD%\objectsharer\;%PYTHONPATH%
cd instrumentserver
python.exe instrument_gui.py
cd ..
