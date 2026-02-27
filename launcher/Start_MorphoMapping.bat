@echo off
setlocal EnableDelayedExpansion
title MorphoMapping GUI

set "INSTALL_DIR=%USERPROFILE%\MorphoMapping"
set "CONDA_ENV=morphomapping"

set "CONDA_ACTIVATE="
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 goto :run_gui

if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
  set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
  goto :run_gui
)
if exist "%USERPROFILE%\miniconda3\condabin\activate.bat" (
  set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\condabin\activate.bat"
  goto :run_gui
)
if exist "%LOCALAPPDATA%\Programs\Miniconda3\Scripts\activate.bat" (
  set "CONDA_ACTIVATE=%LOCALAPPDATA%\Programs\Miniconda3\Scripts\activate.bat"
  goto :run_gui
)

echo [ERROR] Conda/Miniconda not found.
echo Run Install_MorphoMapping.bat first.
pause
exit /b 1

:run_gui
if not defined CONDA_ACTIVATE (
  call conda activate %CONDA_ENV%
) else (
  call "%CONDA_ACTIVATE%" %CONDA_ENV%
)
if not exist "%INSTALL_DIR%\gui\morphomapping_gui.py" (
  echo [ERROR] MorphoMapping not found at %INSTALL_DIR%.
  echo Edit INSTALL_DIR in this file or run Install_MorphoMapping.bat first.
  pause
  exit /b 1
)

cd /d "%INSTALL_DIR%\gui"
python morphomapping_gui.py

if %ERRORLEVEL% neq 0 (
  echo.
  echo [ERROR] Failed to start GUI.
  echo Check: python --version (should be 3.10 or 3.11)
  pause
)
endlocal
