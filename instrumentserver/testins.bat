@ECHO OFF
set PYTHONPATH=%CD%;%CD%\..;%CD%\..\objectsharer
python instrument_plugins/%1.py test %2 %3 %4 %5 %6 %7 %8 %9