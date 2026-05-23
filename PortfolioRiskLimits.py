class PortfolioRiskLimits:
    """
    Portfolio Risk Limits - MUST be consistent with config.py

    These limits are used across the system for risk enforcement.
    The values here should match or be derived from config.py values.

    IMPORTANT: If you change MAX_POSITIONS or MAX_PORTFOLIO_HEAT here,
    also update config.py to maintain consistency.
    """

    # PORTFOLIO LIMITS (Conservative Professional Standards)

    # Maximum total portfolio risk exposure at any time
    MAX_PORTFOLIO_HEAT = 30.0  # 30% max total risk (MUST match config.py)

    # Maximum number of positions - Too few = concentration risk
    MAX_POSITIONS = 10  # FIXED: Was 20, now matches config.py

    # Minimum number of positions (for diversification)
    MIN_POSITIONS = 3  # FIXED: Was 5, now more realistic for small portfolios

    # Maximum correlation between any two positions
    MAX_CORRELATION = 0.7  # Reject if corr > 0.7

    # Maximum sector concentration
    MAX_SECTOR_WEIGHT = 30.0  # 30% max in any sector

    # Maximum single position size (backup to Kelly)
    # FIXED: Calculated to be consistent with MAX_PORTFOLIO_HEAT and MAX_POSITIONS
    # Formula: (MAX_PORTFOLIO_HEAT / MAX_POSITIONS) * safety_margin
    # (30 / 10) * 0.8 = 2.4%
    POSITION_SIZE_SAFETY_MARGIN = 0.8
    MAX_SINGLE_POSITION = (MAX_PORTFOLIO_HEAT / MAX_POSITIONS) * POSITION_SIZE_SAFETY_MARGIN  # ~2.4%

    # Minimum position size
    # Too small = transaction costs eat profits
    MIN_POSITION_SIZE = 0.5  # 0.5% minimum

    # Maximum exposure to single market cap category
    MAX_MCAP_CONCENTRATION = 50.0  # 50% max in Large/Mid/Small
    
    
    @staticmethod
    def check_correlation(
        new_symbol: str,
        current_symbols: list,
        returns_data: dict,
        lookback: int = 60
    ) -> tuple:
        """
        FIXED: MAX_CORRELATION was defined but never enforced.
        Returns (is_allowed: bool, max_corr: float, correlated_with: str)

        Args:
            new_symbol: Symbol being added
            current_symbols: List of symbols already in portfolio
            returns_data: Dict of {symbol: pd.Series of daily returns}
            lookback: Number of days to compute correlation over
        """
        import pandas as pd
        import numpy as np

        if not current_symbols or new_symbol not in returns_data:
            return True, 0.0, None

        new_returns = returns_data.get(new_symbol)
        if new_returns is None or len(new_returns) < lookback:
            return True, 0.0, None  # Not enough data — allow conservatively

        new_ret = new_returns.iloc[-lookback:]
        max_corr = 0.0
        worst_symbol = None

        for sym in current_symbols:
            sym_returns = returns_data.get(sym)
            if sym_returns is None or len(sym_returns) < lookback:
                continue
            sym_ret = sym_returns.iloc[-lookback:]
            # Align indices
            combined = pd.concat([new_ret, sym_ret], axis=1).dropna()
            if len(combined) < 20:
                continue
            corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                worst_symbol = sym

        is_allowed = max_corr <= PortfolioRiskLimits.MAX_CORRELATION

        if not is_allowed:
            print(f"   ❌ Correlation check failed for {new_symbol}")
            print(f"   → Correlation with {worst_symbol}: {max_corr:.2f} > {PortfolioRiskLimits.MAX_CORRELATION}")

        return is_allowed, max_corr, worst_symbol

    @staticmethod
    def check_portfolio_heat(current_positions, new_position_kelly):

        # Calculate current portfolio heat
        current_heat = sum([pos.get('kelly_pct', 0) for pos in current_positions])
        
        # Calculate what new heat would be
        new_heat = current_heat + new_position_kelly
        
        # Check if within limit
        if new_heat > PortfolioRiskLimits.MAX_PORTFOLIO_HEAT:
            # Calculate available heat
            available_heat = PortfolioRiskLimits.MAX_PORTFOLIO_HEAT - current_heat
            
            if available_heat <= 0:
                print(f"   ❌ Portfolio heat at limit ({current_heat:.1f}% / {PortfolioRiskLimits.MAX_PORTFOLIO_HEAT}%)")
                print(f"   → Cannot add new position")
                return 0.0
            
            # Scale down new position to fit
            scaled_kelly = min(new_position_kelly, available_heat)
            
            print(f"   ⚠️  Portfolio heat limit: {new_heat:.1f}% > {PortfolioRiskLimits.MAX_PORTFOLIO_HEAT}%")
            print(f"   → Current heat: {current_heat:.1f}%")
            print(f"   → Available: {available_heat:.1f}%")
            print(f"   → Scaled position: {new_position_kelly:.2f}% → {scaled_kelly:.2f}%")
            
            return scaled_kelly
        
        # Within limit - no scaling needed
        return new_position_kelly
    
    
    @staticmethod
    def check_position_count(current_positions, action='add'):

        current_count = len(current_positions)
        
        if action == 'add':
            new_count = current_count + 1
            
            if new_count > PortfolioRiskLimits.MAX_POSITIONS:
                print(f"   ❌ Portfolio already at max positions ({current_count}/{PortfolioRiskLimits.MAX_POSITIONS})")
                return False
            
        elif action == 'remove':
            new_count = current_count - 1
            
            if new_count < PortfolioRiskLimits.MIN_POSITIONS:
                print(f"   ⚠️  Warning: Portfolio below min positions ({new_count}/{PortfolioRiskLimits.MIN_POSITIONS})")
                print(f"   → Concentration risk increases")
        
        return True
    
    
    @staticmethod
    def check_sector_concentration(current_positions, new_symbol, new_kelly_pct, sector_map):

        # Get new position's sector
        new_sector = sector_map.get(new_symbol, 'Unknown')
        
        if new_sector == 'Unknown':
            print(f"   ⚠️  Warning: Unknown sector for {new_symbol}")
            return True  # Allow if sector unknown
        
        # Calculate sector exposures
        sector_exposure = {}
        for pos in current_positions:
            sector = sector_map.get(pos['symbol'], 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.get('kelly_pct', 0)
        
        # Add new position
        new_exposure = sector_exposure.get(new_sector, 0) + new_kelly_pct
        
        # Check limit
        if new_exposure > PortfolioRiskLimits.MAX_SECTOR_WEIGHT:
            print(f"   ❌ Sector concentration exceeded: {new_sector}")
            print(f"   → Current: {sector_exposure.get(new_sector, 0):.1f}%")
            print(f"   → New: {new_exposure:.1f}% > {PortfolioRiskLimits.MAX_SECTOR_WEIGHT}%")
            return False
        
        return True
    
    
    @staticmethod
    def validate_portfolio(positions, sector_map=None, return_issues=False):

        issues = []
        
        # Check 1: Portfolio heat
        total_heat = sum([pos.get('kelly_pct', 0) for pos in positions])
        if total_heat > PortfolioRiskLimits.MAX_PORTFOLIO_HEAT:
            issues.append(f"Portfolio heat {total_heat:.1f}% > {PortfolioRiskLimits.MAX_PORTFOLIO_HEAT}%")
        
        # Check 2: Position count
        if len(positions) > PortfolioRiskLimits.MAX_POSITIONS:
            issues.append(f"Too many positions: {len(positions)} > {PortfolioRiskLimits.MAX_POSITIONS}")
        
        if len(positions) < PortfolioRiskLimits.MIN_POSITIONS and len(positions) > 0:
            issues.append(f"Too few positions: {len(positions)} < {PortfolioRiskLimits.MIN_POSITIONS}")
        
        # Check 3: Individual position sizes
        for pos in positions:
            kelly = pos.get('kelly_pct', 0)
            symbol = pos.get('symbol', 'Unknown')
            
            if kelly > PortfolioRiskLimits.MAX_SINGLE_POSITION:
                issues.append(f"{symbol}: {kelly:.1f}% > {PortfolioRiskLimits.MAX_SINGLE_POSITION}% max")
            
            if kelly < PortfolioRiskLimits.MIN_POSITION_SIZE:
                issues.append(f"{symbol}: {kelly:.1f}% < {PortfolioRiskLimits.MIN_POSITION_SIZE}% min")
        
        # Check 4: Sector concentration (if sector map provided)
        if sector_map:
            sector_exposure = {}
            for pos in positions:
                sector = sector_map.get(pos['symbol'], 'Unknown')
                sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.get('kelly_pct', 0)
            
            for sector, exposure in sector_exposure.items():
                if exposure > PortfolioRiskLimits.MAX_SECTOR_WEIGHT:
                    issues.append(f"Sector {sector}: {exposure:.1f}% > {PortfolioRiskLimits.MAX_SECTOR_WEIGHT}%")
        
        # Return results
        if return_issues:
            return issues
        else:
            return len(issues) == 0
    
    
    @staticmethod
    def get_portfolio_summary(positions, sector_map=None):
        """Get portfolio risk summary
        
        Args:
            positions: List of position dicts
            sector_map: Optional sector mapping
        
        Returns:
            dict: Portfolio summary statistics
        """
        summary = {
            'total_positions': len(positions),
            'total_heat': sum([pos.get('kelly_pct', 0) for pos in positions]),
            'avg_position_size': sum([pos.get('kelly_pct', 0) for pos in positions]) / len(positions) if positions else 0,
            'max_position': max([pos.get('kelly_pct', 0) for pos in positions]) if positions else 0,
            'min_position': min([pos.get('kelly_pct', 0) for pos in positions]) if positions else 0,
            'within_limits': True,
            'issues': []
        }
        
        # Add sector breakdown if available
        if sector_map:
            sector_exposure = {}
            for pos in positions:
                sector = sector_map.get(pos['symbol'], 'Unknown')
                sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.get('kelly_pct', 0)
            
            summary['sector_exposure'] = sector_exposure
        
        # Check limits
        issues = PortfolioRiskLimits.validate_portfolio(positions, sector_map, return_issues=True)
        summary['issues'] = issues
        summary['within_limits'] = len(issues) == 0
        
        return summary


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == '__main__':
    # Example current portfolio
    current_positions = [
        {'symbol': 'RELIANCE', 'kelly_pct': 5.0, 'sector': 'Energy'},
        {'symbol': 'TCS', 'kelly_pct': 4.5, 'sector': 'IT'},
        {'symbol': 'HDFCBANK', 'kelly_pct': 4.0, 'sector': 'Banking'},
        {'symbol': 'INFY', 'kelly_pct': 3.5, 'sector': 'IT'},
    ]
    
    # Sector mapping
    sector_map = {
        'RELIANCE': 'Energy',
        'TCS': 'IT',
        'HDFCBANK': 'Banking',
        'INFY': 'IT',
        'WIPRO': 'IT'
    }
    
    print("=" * 70)
    print("PORTFOLIO RISK LIMITS - EXAMPLE USAGE")
    print("=" * 70)
    
    # Example 1: Check portfolio heat
    print("\n1. Check Portfolio Heat:")
    new_kelly = 6.0
    scaled = PortfolioRiskLimits.check_portfolio_heat(current_positions, new_kelly)
    print(f"   Original Kelly: {new_kelly:.2f}%")
    print(f"   Scaled Kelly: {scaled:.2f}%")
    
    # Example 2: Check position count
    print("\n2. Check Position Count:")
    can_add = PortfolioRiskLimits.check_position_count(current_positions, 'add')
    print(f"   Can add position: {can_add}")
    
    # Example 3: Check sector concentration
    print("\n3. Check Sector Concentration:")
    can_add_wipro = PortfolioRiskLimits.check_sector_concentration(
        current_positions, 'WIPRO', 5.0, sector_map
    )
    print(f"   Can add WIPRO (IT sector): {can_add_wipro}")
    
    # Example 4: Validate entire portfolio
    print("\n4. Portfolio Validation:")
    is_valid = PortfolioRiskLimits.validate_portfolio(current_positions, sector_map)
    print(f"   Portfolio valid: {is_valid}")
    
    # Example 5: Get portfolio summary
    print("\n5. Portfolio Summary:")
    summary = PortfolioRiskLimits.get_portfolio_summary(current_positions, sector_map)
    import json
    print(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 70)