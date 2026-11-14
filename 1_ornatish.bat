@echo off
chcp 65001 >nul
echo === Matnchi o'rnatilmoqda ===
cd /d "%~dp0"
py -3.12 -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate gradio
echo.
echo === Tayyor! Endi 2_ishga_tushirish.bat ni bosing ===
pause
