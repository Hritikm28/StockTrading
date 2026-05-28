@echo off
title Stock Trading - Full Daily Analysis
cd /d "C:\Users\studi\Desktop\Hritik\Data Analytics\StockTrading\Equity"

echo.
echo =====================================================
echo   FULL TRADING ANALYSIS  -  %DATE%
echo   All 10 India alpha signals + Regime detection
echo   70 stocks (Nifty50 + NiftyNext50)
echo =====================================================
echo.

echo =====================================================
echo   [Step 1/5] Refreshing market data (fresh prices)...
echo =====================================================
echo.

python quick_update_data.py

echo.
echo =====================================================
echo   [Step 2/5] Running signal engine...
echo =====================================================
echo.

python daily_runner.py --quick

echo.
echo =====================================================
echo   [Step 3/5] Publishing signals...
echo =====================================================
echo.

python publish_signals.py

echo.
echo =====================================================
echo   [Step 4/5] Updating paper trade tracker...
echo =====================================================
echo.

python paper_trade_tracker.py --today

echo.
echo =====================================================
echo   Checking signal health...
echo =====================================================
echo.

python signal_decay_detector.py --report

echo.
echo =====================================================
echo   [Step 5/5] Pushing signals to GitHub...
echo =====================================================
echo.

git add signals/
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "signals: daily run %DATE%"
    git pull --rebase origin master
    git push
    if errorlevel 1 (
        echo   WARNING: Push failed. Try manually: git push
    ) else (
        echo   Signals pushed to GitHub!
    )
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
