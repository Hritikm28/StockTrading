@echo off
title Stock Trading - Weekly ML Retrain (Run on Sunday)
cd /d "C:\Users\studi\Desktop\Hritik\Data Analytics\StockTrading\Equity"

echo.
echo =====================================================
echo   WEEKLY ML MODEL RETRAINING
echo   Trains XGBoost + LightGBM + CatBoost + RF
echo   for all 70 stocks. Takes 2-3 hours.
echo   Run this on Sunday evening.
echo =====================================================
echo.

python daily_runner.py

echo.
echo =====================================================
echo   Retraining complete! ML models cached.
echo   Daily runs this week will use these models.
echo =====================================================
echo.
pause
