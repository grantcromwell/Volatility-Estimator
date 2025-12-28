#!/usr/bin/env python3
"""
Production Script: Market Volatility Assessment

Assesses overall market volatility (0-100% scale) and volatility for favorite assets.
Uses volatility estimation with ensemble methods.

Usage:
    python assess_market_volatility.py
    
    # With custom favorite assets:
    python assess_market_volatility.py --favorites AAPL MSFT GOOGL NVDA
"""

import numpy as np
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from vol_estimator import (
    VolatilityEstimator,
    EWMAVolatilityEstimator,
    get_logger
)

logger = get_logger(__name__)


class MarketVolatilityAssessor:
    """
    Market volatility assessor.
    
    Calculates:
    1. Overall market volatility (0-100% scale)
    2. Volatility for favorite assets
    """
    
    # Market indices for overall market assessment
    MARKET_INDICES = ['SPY', 'QQQ', 'DIA', 'IWM']  # S&P 500, Nasdaq, Dow, Russell 2000
    
    # Sector ETFs for sector volatility analysis
    SECTOR_ETFS = {
        'Technology': 'XLK',      # Technology Select Sector SPDR
        'Financial': 'XLF',       # Financial Select Sector SPDR
        'Healthcare': 'XLV',      # Health Care Select Sector SPDR
        'Consumer Discretionary': 'XLY',  # Consumer Discretionary Select Sector SPDR
        'Consumer Staples': 'XLP',  # Consumer Staples Select Sector SPDR
        'Energy': 'XLE',          # Energy Select Sector SPDR
        'Industrials': 'XLI',     # Industrial Select Sector SPDR
        'Materials': 'XLB',       # Materials Select Sector SPDR
        'Real Estate': 'XLRE',    # Real Estate Select Sector SPDR
        'Utilities': 'XLU',       # Utilities Select Sector SPDR
        'Communication Services': 'XLC'  # Communication Services Select Sector SPDR
    }
    
    # Historical volatility ranges for normalization (daily volatility)
    # Based on typical market behavior
    VOL_MIN = 0.005   # 0.5% - Very calm market
    VOL_MAX = 0.08    # 8% - Extreme volatility (crisis)
    VOL_TYPICAL_LOW = 0.01   # 1% - Low volatility
    VOL_TYPICAL_HIGH = 0.03  # 3% - High volatility
    
    def __init__(
        self,
        favorites: List[str] = None,
        period: str = "1y",
        use_ensemble: bool = True
    ):
        """
        Initialize market volatility assessor.
        
        Args:
            favorites: List of favorite asset symbols
            period: Data period ('6mo', '1y', '2y')
            use_ensemble: Use ensemble estimation (default: True)
        """
        self.favorites = favorites or []
        self.period = period
        self.use_ensemble = use_ensemble
        
        # Initialize estimator with enterprise features
        self.estimator = VolatilityEstimator(
            mst_enabled=True,
            lorenz_enabled=True,
            hmm_enabled=True,
            ewma_enabled=True,
            garch_enabled=False,  # Optional, requires arch library
            ensemble_weights={
                'simple_std': 0.1,
                'ewma': 0.35,
                'hmm': 0.2,
                'mst': 0.2,
                'lorenz': 0.15
            }
        )
        
        logger.info(
            f"MarketVolatilityAssessor initialized | "
            f"favorites={len(self.favorites)} | period={period}"
        )
    
    def normalize_volatility_to_percent(self, volatility: float) -> float:
        """
        Normalize volatility to 0-100% scale.
        
        Uses historical ranges:
        - 0% = Very calm market (vol < 0.5%)
        - 50% = Typical volatility (1-3%)
        - 100% = Extreme volatility (vol > 8%)
        
        Args:
            volatility: Daily volatility (as decimal, e.g., 0.02 = 2%)
        
        Returns:
            Normalized volatility percentage (0-100)
        """
        # Clamp to reasonable range
        vol_clamped = np.clip(volatility, self.VOL_MIN, self.VOL_MAX)
        
        # Linear normalization: map [VOL_MIN, VOL_MAX] to [0, 100]
        normalized = ((vol_clamped - self.VOL_MIN) / (self.VOL_MAX - self.VOL_MIN)) * 100
        
        # Ensure result is in [0, 100]
        return float(np.clip(normalized, 0, 100))
    
    def assess_market_volatility(self) -> Dict:
        """
        Assess overall market volatility.
        
        Returns:
            Dictionary with market volatility assessment
        """
        logger.info("Assessing overall market volatility...")
        
        try:
            # Fetch market index data
            data = self.estimator.fetch_market_data(
                self.MARKET_INDICES,
                period=self.period
            )
            
            if not data:
                raise ValueError("Failed to fetch market data")
            
            # Prepare returns
            returns, symbols = self.estimator.prepare_returns(data, min_length=50)
            
            if returns.size == 0:
                raise ValueError("Insufficient data for market assessment")
            
            logger.info(f"Market data prepared | symbols={symbols} | shape={returns.shape}")
            
            # Calculate market-wide volatility using ensemble
            if self.use_ensemble:
                market_vols = self.estimator.estimate_ensemble(returns, symbols)
            else:
                market_vols = self.estimator.estimate_volatility(returns, symbols)
            
            # Calculate average market volatility
            avg_market_vol = np.mean(list(market_vols.values()))
            
            # Normalize to 0-100%
            market_vol_percent = self.normalize_volatility_to_percent(avg_market_vol)
            
            # Determine volatility level
            if market_vol_percent < 20:
                level = "Very Low"
                description = "Market is very calm"
            elif market_vol_percent < 40:
                level = "Low"
                description = "Market volatility is below average"
            elif market_vol_percent < 60:
                level = "Moderate"
                description = "Market volatility is typical"
            elif market_vol_percent < 80:
                level = "High"
                description = "Market volatility is elevated"
            else:
                level = "Extreme"
                description = "Market volatility is very high (crisis level)"
            
            result = {
                'market_volatility_percent': market_vol_percent,
                'market_volatility_decimal': avg_market_vol,
                'level': level,
                'description': description,
                'index_volatilities': market_vols,
                'timestamp': datetime.now().isoformat(),
                'period': self.period
            }
            
            # Add sector volatility analysis
            sector_analysis = self.assess_sector_volatility()
            result['sector_analysis'] = sector_analysis
            
            logger.info(
                f"Market volatility assessed | "
                f"percent={market_vol_percent:.1f}% | level={level}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Market volatility assessment failed | error={e}")
            raise
    
    def assess_sector_volatility(self) -> Dict:
        """
        Assess sector volatility for this week and project next 60 days.
        
        Returns:
            Dictionary with sector volatility analysis
        """
        logger.info("Assessing sector volatility...")
        
        try:
            # Get sector ETF symbols
            sector_symbols = list(self.SECTOR_ETFS.values())
            sector_names = list(self.SECTOR_ETFS.keys())
            
            # Fetch recent data (need enough for weekly + projection)
            data = self.estimator.fetch_market_data(
                sector_symbols,
                period="3mo"  # Need enough data for analysis
            )
            
            if not data:
                logger.warning("Failed to fetch sector data")
                return {}
            
            # Prepare returns
            returns_dict = self.estimator.data_fetcher.compute_returns(data, method="log")
            returns_matrix, symbols = self.estimator.data_fetcher.align_returns(
                returns_dict, min_length=20
            )
            
            if returns_matrix.size == 0:
                logger.warning("Insufficient sector data")
                return {}
            
            # Map symbols to sector names
            symbol_to_sector = {}
            for i, symbol in enumerate(symbols):
                for sector_name, sector_symbol in self.SECTOR_ETFS.items():
                    if symbol == sector_symbol:
                        symbol_to_sector[symbol] = sector_name
                        break
            
            # Calculate weekly volatility (last 5 trading days)
            weekly_vols = {}
            n_periods = returns_matrix.shape[1]
            week_window = min(5, n_periods)
            
            for i, symbol in enumerate(symbols):
                if symbol in symbol_to_sector:
                    sector_name = symbol_to_sector[symbol]
                    week_returns = returns_matrix[i, -week_window:]
                    daily_vol = np.std(week_returns)  # Daily volatility
                    annualized_vol = daily_vol * np.sqrt(252)  # Annualized for display
                    weekly_vols[sector_name] = {
                        'symbol': symbol,
                        'volatility': annualized_vol,  # Annualized for comparison
                        'daily_volatility': daily_vol,  # Daily for normalization
                        'normalized': self.normalize_volatility_to_percent(daily_vol)
                    }
            
            # Find most volatile sector this week
            most_volatile_week = None
            if weekly_vols:
                most_volatile_week = max(
                    weekly_vols.items(),
                    key=lambda x: x[1]['volatility']
                )
            
            # Project 60-day volatility using EWMA
            projected_vols = {}
            for i, symbol in enumerate(symbols):
                if symbol in symbol_to_sector:
                    sector_name = symbol_to_sector[symbol]
                    asset_returns = returns_matrix[i, :]
                    
                    # Fit EWMA model
                    try:
                        ewma = EWMAVolatilityEstimator(decay_factor=0.94)
                        ewma.fit(asset_returns)
                        
                        # Forecast 60-day volatility (using current volatility as proxy)
                        # EWMA produces constant forecasts, so we use current vol
                        current_vol = ewma.get_current_volatility()
                        
                        # Scale to 60-day (assuming sqrt of time scaling)
                        projected_60d = current_vol * np.sqrt(60/252)  # 60 trading days out of 252
                        annualized_projected = current_vol * np.sqrt(252)
                        
                        projected_vols[sector_name] = {
                            'symbol': symbol,
                            'projected_volatility': annualized_projected,  # Annualized
                            'daily_projected_volatility': current_vol,  # Daily for normalization
                            'normalized': self.normalize_volatility_to_percent(current_vol)
                        }
                    except Exception as e:
                        logger.warning(f"EWMA projection failed for {sector_name} | error={e}")
                        continue
            
            # Find most volatile sector projection
            most_volatile_projected = None
            if projected_vols:
                most_volatile_projected = max(
                    projected_vols.items(),
                    key=lambda x: x[1]['projected_volatility']
                )
            
            result = {
                'weekly_volatilities': weekly_vols,
                'most_volatile_week': {
                    'sector': most_volatile_week[0] if most_volatile_week else None,
                    'volatility': most_volatile_week[1]['volatility'] if most_volatile_week else None,
                    'normalized': most_volatile_week[1]['normalized'] if most_volatile_week else None,
                    'symbol': most_volatile_week[1]['symbol'] if most_volatile_week else None
                } if most_volatile_week else {},
                'projected_volatilities': projected_vols,
                'most_volatile_projected': {
                    'sector': most_volatile_projected[0] if most_volatile_projected else None,
                    'projected_volatility': most_volatile_projected[1]['projected_volatility'] if most_volatile_projected else None,
                    'normalized': most_volatile_projected[1]['normalized'] if most_volatile_projected else None,
                    'symbol': most_volatile_projected[1]['symbol'] if most_volatile_projected else None
                } if most_volatile_projected else {},
                'projection_horizon_days': 60
            }
            
            logger.info(
                f"Sector volatility assessed | "
                f"most_volatile_week={result['most_volatile_week'].get('sector')} | "
                f"most_volatile_projected={result['most_volatile_projected'].get('sector')}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sector volatility assessment failed | error={e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def assess_favorite_assets(self) -> Dict:
        """
        Assess volatility for favorite assets.
        
        Returns:
            Dictionary with favorite assets volatility assessment
        """
        if not self.favorites:
            logger.warning("No favorite assets specified")
            return {}
        
        logger.info(f"Assessing volatility for favorite assets: {self.favorites}")
        
        try:
            # Fetch data for favorite assets
            data = self.estimator.fetch_market_data(
                self.favorites,
                period=self.period
            )
            
            if not data:
                raise ValueError("Failed to fetch favorite assets data")
            
            # Prepare returns
            returns, symbols = self.estimator.prepare_returns(data, min_length=50)
            
            if returns.size == 0:
                raise ValueError("Insufficient data for favorite assets")
            
            # Calculate volatilities using ensemble
            if self.use_ensemble:
                asset_vols = self.estimator.estimate_ensemble(returns, symbols)
            else:
                asset_vols = self.estimator.estimate_volatility(returns, symbols)
            
            # Normalize to percentages
            asset_vols_percent = {
                symbol: self.normalize_volatility_to_percent(vol)
                for symbol, vol in asset_vols.items()
            }
            
            result = {
                'assets': asset_vols,
                'assets_percent': asset_vols_percent,
                'timestamp': datetime.now().isoformat(),
                'period': self.period
            }
            
            logger.info(
                f"Favorite assets assessed | "
                f"n_assets={len(asset_vols)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Favorite assets assessment failed | error={e}")
            raise
    
    def assess_all(self) -> Dict:
        """
        Assess both market volatility and favorite assets.
        
        Returns:
            Complete assessment dictionary
        """
        logger.info("Starting comprehensive volatility assessment...")
        
        market_assessment = self.assess_market_volatility()
        favorites_assessment = self.assess_favorite_assets()
        
        return {
            'market': market_assessment,
            'favorites': favorites_assessment,
            'timestamp': datetime.now().isoformat()
        }


def print_assessment_report(assessment: Dict):
    """
    Print formatted assessment report.
    
    Args:
        assessment: Assessment dictionary from assess_all()
    """
    print("\n" + "=" * 80)
    print("MARKET VOLATILITY ASSESSMENT REPORT")
    print("=" * 80)
    print(f"Generated: {assessment['timestamp']}")
    
    # Market volatility
    market = assessment['market']
    print("\n" + "-" * 80)
    print("OVERALL MARKET VOLATILITY")
    print("-" * 80)
    print(f"Volatility Level: {market['level']}")
    print(f"Market Volatility: {market['market_volatility_percent']:.1f}%")
    print(f"Description: {market['description']}")
    print(f"\nMarket Index Volatilities ({market['period']}):")
    for symbol, vol in sorted(market['index_volatilities'].items()):
        vol_percent = vol * 100
        normalized = assessment['assessor'].normalize_volatility_to_percent(vol)
        print(f"  {symbol:6s}: {vol_percent:6.2f}% (normalized: {normalized:5.1f}%)")
    
    # Sector volatility analysis
    if 'sector_analysis' in market and market['sector_analysis']:
        sector_analysis = market['sector_analysis']
        print("\n" + "-" * 80)
        print("SECTOR VOLATILITY ANALYSIS")
        print("-" * 80)
        
        # Most volatile sector this week
        if sector_analysis.get('most_volatile_week') and sector_analysis['most_volatile_week'].get('sector'):
            mv_week = sector_analysis['most_volatile_week']
            print(f"\nMost Volatile Sector This Week:")
            print(f"  Sector: {mv_week['sector']}")
            print(f"  ETF: {mv_week['symbol']}")
            vol_annualized = mv_week['volatility'] * 100
            print(f"  Weekly Volatility (annualized): {vol_annualized:.2f}%")
            print(f"  Normalized: {mv_week['normalized']:.1f}%")
        
        # Projected most volatile sector (60 days)
        if sector_analysis.get('most_volatile_projected') and sector_analysis['most_volatile_projected'].get('sector'):
            mv_proj = sector_analysis['most_volatile_projected']
            print(f"\nProjected Most Volatile Sector (Next 60 Days):")
            print(f"  Sector: {mv_proj['sector']}")
            print(f"  ETF: {mv_proj['symbol']}")
            proj_vol_annualized = mv_proj['projected_volatility'] * 100
            print(f"  Projected Volatility (annualized): {proj_vol_annualized:.2f}%")
            print(f"  Normalized: {mv_proj['normalized']:.1f}%")
        
        # Top 5 sectors by weekly volatility
        if sector_analysis.get('weekly_volatilities'):
            weekly_vols = sector_analysis['weekly_volatilities']
            sorted_sectors = sorted(
                weekly_vols.items(),
                key=lambda x: x[1]['volatility'],
                reverse=True
            )[:5]
            
            print(f"\nTop 5 Most Volatile Sectors This Week:")
            for sector_name, vol_data in sorted_sectors:
                vol_ann = vol_data['volatility'] * 100
                print(f"  {sector_name:30s}: {vol_ann:6.2f}% (normalized: {vol_data['normalized']:5.1f}%)")
    
    # Favorite assets
    if assessment['favorites']:
        favorites = assessment['favorites']
        print("\n" + "-" * 80)
        print("FAVORITE ASSETS VOLATILITY")
        print("-" * 80)
        print(f"Assessment Period: {favorites['period']}")
        print("\nAsset Volatilities:")
        
        # Sort by volatility (highest first)
        sorted_assets = sorted(
            favorites['assets'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for symbol, vol in sorted_assets:
            vol_percent = vol * 100
            normalized = favorites['assets_percent'][symbol]
            print(f"  {symbol:10s}: {vol_percent:6.2f}% (normalized: {normalized:5.1f}%)")
        
        # Summary statistics
        vols = list(favorites['assets'].values())
        print(f"\nSummary:")
        print(f"  Average Volatility: {np.mean(vols)*100:.2f}%")
        print(f"  Highest Volatility: {max(vols)*100:.2f}% ({max(favorites['assets'].items(), key=lambda x: x[1])[0]})")
        print(f"  Lowest Volatility:  {min(vols)*100:.2f}% ({min(favorites['assets'].items(), key=lambda x: x[1])[0]})")
    
    print("\n" + "=" * 80)
    print("Assessment Complete")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Assess market volatility and favorite assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default assessment (no favorites)
  python assess_market_volatility.py
  
  # With favorite assets
  python assess_market_volatility.py --favorites AAPL MSFT GOOGL NVDA AMZN
  
  # Custom period
  python assess_market_volatility.py --period 6mo --favorites TSLA NVDA
        """
    )
    
    parser.add_argument(
        '--favorites',
        nargs='+',
        default=[],
        help='Favorite asset symbols (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '--period',
        default='1y',
        choices=['6mo', '1y', '2y'],
        help='Data period for assessment (default: 1y)'
    )
    
    parser.add_argument(
        '--no-ensemble',
        action='store_true',
        help='Disable ensemble estimation (use simple method)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create assessor
        assessor = MarketVolatilityAssessor(
            favorites=args.favorites,
            period=args.period,
            use_ensemble=not args.no_ensemble
        )
        
        # Perform assessment
        assessment = assessor.assess_all()
        
        # Store assessor for normalization in print
        assessment['assessor'] = assessor
        
        # Print report
        print_assessment_report(assessment)
        
        return assessment
        
    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user.")
        return None
    except Exception as e:
        logger.error(f"Assessment failed | error={e}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

