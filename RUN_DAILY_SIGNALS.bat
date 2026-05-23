@echo off
title Stock Trading - Daily Signals
cd /d "C:\Users\studi\Desktop\Hritik\Data Analytics\StockTrading\Equity"
echo.
echo =========================================
echo   RUNNING TODAY'S TRADING SIGNALS...
echo =========================================
echo.
python daily_runner.py --quick
echo.
echo =========================================
echo   DONE! Check signals above.
echo   Signals saved to: paper_trading\records\
echo =========================================
echo.
pause
