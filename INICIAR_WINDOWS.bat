@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title AGRO IA - Laboratorio Didatico de Inspecao Foliar

echo ==============================================================
echo    AGRO IA - Inspecao Foliar com Inteligencia Artificial
echo ==============================================================
echo.

set "PYCOMMAND="
where py >nul 2>&1
if not errorlevel 1 (
    for %%V in (3.14 3.13 3.12 3.11) do (
        if not defined PYCOMMAND (
            py -%%V -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
            if not errorlevel 1 set "PYCOMMAND=py -%%V"
        )
    )
)
if not defined PYCOMMAND (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,14) and sys.maxsize > 2**32" >nul 2>&1
        if not errorlevel 1 set "PYCOMMAND=python"
    )
)
if not defined PYCOMMAND (
    echo ERRO: Python 3.11 a 3.14 de 64 bits nao encontrado.
    echo Confira a instalacao do Python e o Python Launcher.
    pause
    exit /b 1
)

echo Interpretador encontrado:
%PYCOMMAND% --version
if errorlevel 1 goto falha

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,14) and sys.maxsize > 2**32" >nul 2>&1
    if errorlevel 1 (
        echo Ambiente virtual antigo ou incompativel.
        echo Renomeie a pasta .venv para .venv_antiga e execute novamente.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Criando ambiente virtual...
    %PYCOMMAND% -m venv .venv
    if errorlevel 1 goto falha
)

echo [2/3] Verificando dependencias...
.venv\Scripts\python.exe -c "import pygame, numpy, cv2, PIL, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo Instalando bibliotecas. Internet necessaria nesta etapa.
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto falha
)

echo [3/3] Iniciando laboratorio...
.venv\Scripts\python.exe agroia_folhas.py
if errorlevel 1 goto falha
exit /b 0

:falha
echo.
echo Ocorreu uma falha. Confira as mensagens acima.
echo Caso exista agroia_erros.log, envie esse arquivo para diagnostico.
pause
exit /b 1
