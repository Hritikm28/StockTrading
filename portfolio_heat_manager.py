class PortfolioHeatManager:
    """Manage portfolio heat and position sizing dynamically"""
    
    def __init__(self, max_heat=0.25, max_positions=10):
        self.max_heat = max_heat
        self.max_positions = max_positions
        
        self.max_single_position = (max_heat / max_positions) * 0.8
        
        print(f"📊 Portfolio Heat Manager Initialized:")
        print(f"   Max Heat: {max_heat*100:.0f}%")
        print(f"   Max Positions: {max_positions}")
        print(f"   Max Single Position: {self.max_single_position*100:.1f}%")
    
    def calculate_safe_position_size(self, kelly_pct, current_heat, num_positions):

        kelly_fraction = kelly_pct / 100.0  # Convert to decimal
        
        # Available heat headroom
        available_heat = self.max_heat - current_heat
        
        if available_heat <= 0:
            return 0.0  # No room
        
        # Can't exceed available heat
        size = min(kelly_fraction, available_heat)
        
        # Can't exceed max single position
        size = min(size, self.max_single_position)
        
        if num_positions < self.max_positions / 2:
            # Reserve at least 50% of heat for future positions
            reserved_heat = self.max_heat * 0.5
            if current_heat + size > reserved_heat:
                size = max(0, reserved_heat - current_heat)
        
        return size * 100  # Return as percentage
    
    def validate_portfolio(self, positions):
        """
        Check if portfolio violates heat limits
        Returns: (is_valid, violations, corrections)
        """
        total_heat = sum(pos['size'] for pos in positions.values())
        violations = []
        corrections = {}
        
        # Check total heat
        if total_heat > self.max_heat:
            violations.append(f"Total heat {total_heat*100:.1f}% > {self.max_heat*100:.0f}%")
            
            # Calculate scale factor to bring within limits
            scale_factor = self.max_heat / total_heat
            
            for symbol, pos in positions.items():
                new_size = pos['size'] * scale_factor
                corrections[symbol] = {
                    'old_size': pos['size'],
                    'new_size': new_size,
                    'reason': 'portfolio_heat_exceeded'
                }
        
        # Check individual positions
        for symbol, pos in positions.items():
            if pos['size'] > self.max_single_position:
                violations.append(f"{symbol}: {pos['size']*100:.1f}% > {self.max_single_position*100:.1f}%")
                
                if symbol not in corrections:
                    corrections[symbol] = {
                        'old_size': pos['size'],
                        'new_size': self.max_single_position,
                        'reason': 'position_size_exceeded'
                    }
        
        is_valid = len(violations) == 0
        return is_valid, violations, corrections