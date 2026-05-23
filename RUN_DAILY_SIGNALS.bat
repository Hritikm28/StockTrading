@echo off
title Stock Trading - Full Daily Analysis
cd /d "C:\Users\studi\Desktop\Hritik\Data Analytics\StockTrading\Equity"

echo.
echo =====================================================
echo   FULL TRADING ANALYSIS  -  %DATE%
echo   All 7 India alpha signals + Regime detection
echo   70 stocks (Nifty50 + NiftyNext50)
echo =====================================================
echo.

python daily_runner.py --quick

echo.
echo =====================================================
echo   Pushing signals to GitHub...
echo =====================================================
echo.

python publish_signals.py

git add signals/
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "signals: manual run"
    git push
    echo   Signals pushed to GitHub!
) else (
    echo   No new signals to push.
)

echo.
echo =====================================================
echo   DONE! Full analysis complete.
echo   Signals saved locally AND pushed to GitHub.
echo   View at: github.com/Hritikm28/StockTrading
echo =====================================================
echo.
pause
