@echo off
set PYTHONPATH=%CD%;%CD%\objectsharer\;%CD%\pulseseq;%PYTHONPATH%
start start_instrumentserver.bat
start start_dataserver.bat
start start_instrumentgui.bat
