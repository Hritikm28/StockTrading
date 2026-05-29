@echo off
title Stock Trading - Daily Analysis (Run after 6 PM)
cd /d "C:\Users\studi\Desktop\Hritik\Data Analytics\StockTrading\Equity"

echo.
echo =====================================================
echo   DAILY TRADING ANALYSIS  -  %DATE%
echo   Run this AFTER 6 PM for NEXT DAY signals
echo   All 10 India alpha signals + Regime detection
echo   70 stocks (Nifty50 + NiftyNext50)
echo =====================================================
echo.

echo [Step 1/5] Updating market data (bhav copy + yfinance)...
echo.

python quick_update_data.py

echo.
echo [Step 2/5] Running signal engine...
echo.

python daily_runner.py --quick

echo.
echo [Step 3/5] Publishing signals...
echo.

python publish_signals.py

echo.
echo [Step 4/5] Updating paper trade tracker...
echo.

python paper_trade_tracker.py --today

echo.
echo Checking signal health...
echo.

python signal_decay_detector.py --report

echo.
echo [Step 5/5] Pushing signals to GitHub...
echo.

git add signals/
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "signals: %DATE%"
    git pull origin master -X ours
    git push
    if errorlevel 1 (
        echo   WARNING: Push failed. Run manually: git push
    ) else (
        echo   Signals pushed to GitHub!
    )
) else (
    echo   No new signals to push.
)

echo.
echo =====================================================
echo   DONE! Signals ready for TOMORROW.
echo   Review before 9:15 AM market open.
echo   View at: github.com/Hritikm28/StockTrading
echo =====================================================
echo.
pause
