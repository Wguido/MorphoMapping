@echo off
setlocal EnableDelayedExpansion
title MorphoMapping - Installation
echo.
echo ============================================
echo   MorphoMapping Installer (Windows)
echo ============================================
echo.

set "INSTALL_DIR=%USERPROFILE%\MorphoMapping"
set "CONDA_ENV=morphomapping"

set "CONDA_ACTIVATE="
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
  for /f "tokens=*" %%i in ('where conda 2^>nul') do set "CONDA_EXE=%%i"
  echo [OK] Conda found.
  goto :have_conda
)

if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
  set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
  set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
  goto :have_conda
)
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
  set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\condabin\activate.bat"
  set "CONDA_EXE=%USERPROFILE%\miniconda3\condabin\conda.bat"
  goto :have_conda
)
if exist "%LOCALAPPDATA%\Programs\Miniconda3\Scripts\activate.bat" (
  set "CONDA_ACTIVATE=%LOCALAPPDATA%\Programs\Miniconda3\Scripts\activate.bat"
  set "CONDA_EXE=%LOCALAPPDATA%\Programs\Miniconda3\Scripts\conda.exe"
  goto :have_conda
)

echo [ERROR] Conda/Miniconda not found.
echo.
echo Install Miniconda first, e.g.:
echo   winget install -e --id Anaconda.Miniconda3 --accept-package-agreements
echo.
echo Or: https://docs.conda.io/en/latest/miniconda.html
echo Then run this script again.
pause
exit /b 1

:have_conda
if not defined CONDA_ACTIVATE (
  set "CONDA_ACTIVATE=%CONDA_EXE:\conda.exe=\activate.bat%"
  if not exist "!CONDA_ACTIVATE!" set "CONDA_ACTIVATE=%CONDA_EXE:\conda.bat=\activate.bat%"
)

if not exist "%INSTALL_DIR%" (
  mkdir "%INSTALL_DIR%" 2>nul
  where git >nul 2>&1
  if !ERRORLEVEL! equ 0 (
    echo Cloning MorphoMapping to %INSTALL_DIR% ...
    git clone https://github.com/Wguido/MorphoMapping.git "%INSTALL_DIR%"
    if !ERRORLEVEL! neq 0 (
      echo [ERROR] Git clone failed.
      pause
      exit /b 1
    )
  ) else (
    echo [NOTE] Git not found.
    echo Please install MorphoMapping manually:
    echo 1. Open https://github.com/Wguido/MorphoMapping
    echo 2. Code ^> Download ZIP
    echo 3. Extract to %INSTALL_DIR% (gui folder must be inside)
    echo 4. Run this script again.
    pause
    exit /b 1
  )
) else (
  echo Directory already exists: %INSTALL_DIR%
  where git >nul 2>&1
  if !ERRORLEVEL! equ 0 (
    echo Updating repository ...
    cd /d "%INSTALL_DIR%"
    git pull 2>nul
    cd /d "%~dp0"
  )
)

if not exist "%INSTALL_DIR%\gui\morphomapping_gui.py" (
  echo [ERROR] Expected file not found: %INSTALL_DIR%\gui\morphomapping_gui.py
  echo Ensure the repo is fully extracted under %INSTALL_DIR%.
  pause
  exit /b 1
)

echo.
echo Creating Conda environment "%CONDA_ENV%" (Python 3.11) ...
"%CONDA_EXE%" create -n %CONDA_ENV% python=3.11 -y
if %ERRORLEVEL% neq 0 (
  echo [ERROR] conda create failed.
  pause
  exit /b 1
)

echo.
echo Installing Python packages (may take a few minutes) ...
call "%CONDA_ACTIVATE%" %CONDA_ENV%

pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl scipy --quiet
if %ERRORLEVEL% neq 0 (
  echo [ERROR] pip install dependencies failed.
  pause
  exit /b 1
)

cd /d "%INSTALL_DIR%"
pip install -e . --quiet
if %ERRORLEVEL% neq 0 (
  echo [ERROR] pip install -e . failed.
  cd /d "%~dp0"
  pause
  exit /b 1
)

cd /d "%~dp0"
echo.
echo ============================================
echo   Installation complete.
echo ============================================
echo.
echo MorphoMapping is at: %INSTALL_DIR%
echo To start the GUI: run Start_MorphoMapping.bat
echo.
pause
