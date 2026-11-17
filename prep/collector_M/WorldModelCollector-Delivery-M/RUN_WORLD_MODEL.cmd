@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
if defined AIRSIM_PYTHON (
  set "PYTHON=%AIRSIM_PYTHON%"
) else (
  set "PYTHON=%USERPROFILE%\.conda\envs\ue4-airsim\python.exe"
)
if not exist "%PYTHON%" (
  echo ERROR: AirSim Python was not found.
  echo Set AIRSIM_PYTHON to the Python executable in your AirSim environment.
  echo Expected: %PYTHON%
  goto :failed
)
"%PYTHON%" -c "import sys,airsim,numpy,msgpack; print('Python:',sys.version); print('AirSim:',airsim.__version__); print('msgpack:',msgpack.version)"
if errorlevel 1 goto :failed
"%PYTHON%" "%ROOT%scripts\collect_world_model.py" --launch %*
if errorlevel 1 goto :failed
echo.
echo Collection completed. Output path is printed above.
pause
exit /b 0

:failed
echo.
echo Collection did not complete. Read the error above and the episode logs.
pause
exit /b 1

