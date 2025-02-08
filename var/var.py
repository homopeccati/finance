from pathlib import Path
import sys

# Set project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.stats as stats
import data_parser as data_parser
from scipy.optimize import brentq
from datetime import datetime, timedelta
from utils.curve import Curve

from config import simulation_config


class Option:
    """Option class for pricing and risk analysis of European options.
    """    
    def __init__(self, underlier: str, S0: float, K: float, T: float, r: float, sigma: float, option_type: str, quantity: int, market_price: float):
        """Constructor for Option class.

        Args:
            underlier (str): Ticker symbol of the underlying asset
            S0 (float): Current price of the underlying asset
            K (float): Strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate.
            sigma (float): Daily volatility of the underlying asset.
            option_type (str): Type of the option ('call' or 'put').
            quantity (int): Number of option contracts (negative for short positions).
            market_price (float): Observed market price of the option.
        """        
        self.underlier = underlier
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma * np.sqrt(252)
        self.option_type = option_type.lower()
        self.quantity = quantity
        self.market_price = market_price

    def black_scholes_price(self) -> float:
        """Calculate the Black-Scholes price of the option.

        Raises:
            ValueError: If the option type is not 'call' or 'put'.

        Returns:
            float: Option price based on the Black-Scholes model
        """        
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == "call":
            price = self.S0 * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        elif self.option_type == "put":
            price = self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S0 * stats.norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        
        return price

    def payoff(self, ST: float) -> float:
        """Calculate the payoff of the option at expiration.

        Args:
            ST (float): The (simulated) price of the underlying asset at expiration.

        Returns:
            float: Payoff of the option at expiration.
        """        
        if self.option_type == "call":
            return np.maximum(ST - self.K, 0) * self.quantity
        elif self.option_type == "put":
            return np.maximum(self.K - ST, 0) * self.quantity
        
    def implied_vol(self) -> float:
        """Calculate the implied volatility of the option based on the market price.
        
            Note: Implemented, but not used in the final version of the project. Can be used as the simulation parameter in MC instead of historical vol.
        """        
        def call_option_price(sigma):
            d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)
            return self.S0 * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)

        def put_option_price(sigma):
            d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)
            return self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S0 * stats.norm.cdf(-d1)

        if self.option_type == "call":
            target_price = self.market_price
            target_func = lambda sigma: call_option_price(sigma) - target_price
        elif self.option_type == "put":
            target_price = self.market_price
            target_func = lambda sigma: put_option_price(sigma) - target_price
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")

        return brentq(target_func, 0.01, 2)

class MonteCarloVaR:
    """Monte Carlo VaR class for estimating the Value at Risk (VaR) of an option portfolio using Monte Carlo simulation.
    """    
    def __init__(self, portfolio: list, underlier_params: pd.DataFrame, days: int, simulations: int, confidence_level=0.95):
        """Constructor for MonteCarloVaR class.

        Args:
            portfolio (list): List of Option objects representing the option portfolio.
            underlier_params (pd.DataFrame): DataFrame with parameters for the underliers.
            days (int): Number of days to simulate stock prices into the future.
            simulations (int): Number of Monte Carlo simulations to run.
            confidence_level (float, optional): Confidence level for VaR estimation. Defaults to 0.95.
        """        
        self.portfolio = portfolio
        self.underlier_params = underlier_params
        self.days = days
        self.simulations = simulations
        self.confidence_level = confidence_level

    def simulate_prices(self) -> dict:
        """Simulate future stock prices for the underliers in the portfolio.

        Returns:
            dict: Dictionary of simulated stock prices for each underlier.
        """        
        dt = 1
        underliers = self.underlier_params['underlier'].tolist()
        mu = self.underlier_params['mu'].values
        sigma = self.underlier_params['sigma'].values
        initial_prices = {option.underlier: option.S0 for option in self.portfolio}

        # Generate correlated random variables
        corr_matrix = np.vstack(self.underlier_params.set_index('underlier').loc[underliers, 'correlation'].values)
        L = np.linalg.cholesky(corr_matrix)
        Z = np.random.randn(self.simulations, len(underliers))
        correlated_Z = Z @ L.T

        simulated_prices = {}
        for i, underlier in enumerate(underliers):
            S0 = initial_prices[underlier]
            S_sim = S0 * np.exp(
                (mu[i] - 0.5 * sigma[i]**2) * dt * self.days +
                sigma[i] * np.sqrt(dt * self.days) * correlated_Z[:, i]
            )
            simulated_prices[underlier] = S_sim

        return simulated_prices

    def estimate_var(self) -> float:
        """Estimate the Value at Risk (VaR) of the option portfolio using Monte Carlo simulation.

        Returns:
            float: Estimated VaR of the option portfolio.
        """        
        simulated_prices = self.simulate_prices()
        portfolio_payoffs = np.zeros(self.simulations)
        initial_value = 0

        for option in self.portfolio:
            underlier_sim_prices = simulated_prices[option.underlier]
            option_payoffs = np.array([option.payoff(ST) for ST in underlier_sim_prices])
            portfolio_payoffs += option_payoffs
            initial_value += option.black_scholes_price() * option.quantity

        profit_losses = portfolio_payoffs - initial_value
        var = np.percentile(profit_losses, (1 - self.confidence_level) * 100)
        return var

def get_options_data(file_path: str, date: str) -> pd.DataFrame:
    """Fetch option data from the MOEX API based on the tickers in the input file and gets the options' quantity.

    Args:
        file_path (str): Path to the file containing the tickers.
        date (str): Date for which the option data is fetched.

    Returns:
        pd.DataFrame: DataFrame containing the option data with quantities.
    """    
    data = pd.read_csv(file_path)
    tickers = data['ticker'].tolist()
    api_option_data = data_parser.fetch_option_data(tickers, date)
    api_option_data = pd.merge(api_option_data, data, left_on='SECID', right_on='ticker').drop(columns='ticker')
    
    return api_option_data

def get_underlier_parameters(options_data: pd.DataFrame, date: str) -> pd.DataFrame:
    """Fetch underlier data for the options in the portfolio and calculate the parameters for the underliers.

    Args:
        options_data (pd.DataFrame): DataFrame containing the option data.
        date (str): Date for which the underlier data is fetched

    Returns:
        pd.DataFrame: DataFrame containing the parameters for the underliers.
    """    
    end_date = datetime.strptime(date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=365)
    tickers = options_data['UNDERLYINGASSET'].unique().tolist()
    
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    data = data_parser.fetch_underlier_data(tickers, start_date, end_date)
    data = data[['SECID', 'TRADEDATE', 'CLOSE']].drop_duplicates()
    
    data['returns'] = data.groupby('SECID', as_index=False)['CLOSE'].pct_change()
    
    params = data.groupby('SECID', as_index=False).agg(mu=('returns', 'mean'), sigma=('returns', 'std'))
    params.rename(columns = {'SECID': 'underlier'}, inplace=True)
    
    # Calculate correlation matrix
    returns_pivot = data.pivot(index='TRADEDATE', columns='SECID', values='returns')
    correlation_matrix = returns_pivot.corr()
    params['correlation'] = correlation_matrix.values.tolist()
    
    return params

def define_portfolio(options_data: pd.DataFrame, underlier_params: pd.DataFrame, date: str) -> list:
    """Define the option portfolio based on the option data and underlier parameters.

    Args:
        options_data (pd.DataFrame): DataFrame containing the option data.
        underlier_params (pd.DataFrame): DataFrame containing the parameters for the underliers.
        date (str): Date for which the portfolio is defined.

    Raises:
        ValueError: If the option type is not 'call' or 'put'.

    Returns:
        list: List of Option objects representing the option portfolio.
    """    
    rf = Curve(data_parser.fetch_risk_free_rate())
    portfolio = []
    
    for _, row in options_data.iterrows():
        underlier=row['UNDERLYINGASSET']
        S0=row['UNDERLYINGSETTLEPRICE']
        K=row['STRIKE']
        
        expiry_date = datetime.strptime(row['LASTTRADEDATE'], '%Y-%m-%d')
        current_date = datetime.strptime(date, '%Y-%m-%d')
        
        T = (expiry_date - current_date).days / 365 
        
        r= rf.get_discount_rate(T)
        
        sigma = float(underlier_params[underlier_params['underlier'] == underlier]['sigma'].iloc[0])
        
        if 'Call' in row ['SECNAME']:
            option_type='call'
        elif 'Put' in row ['SECNAME']:
            option_type='put'
        else: 
            raise ValueError(f"Option type not found for {row['SECID']}. The SECNAME column must contain either 'call' or 'put'.")
        
        quantity=row['quantity']
        market_price=row['PREVSETTLEPRICE']
            
        option = Option(underlier, S0, K, T, r, sigma, option_type, quantity, market_price)
        portfolio.append(option)
    
    return portfolio

if __name__ == "__main__":
    """Main function to estimate the Monte Carlo VaR for an equity options portfolio.
    """    
    #Estimation date
    date = str(input('Input the date in the YYYY-MM-DD format: '))
    
    print('Fetching the data from external soruces...\n')
    options_data = get_options_data("var/tickers.csv", date)
    params = get_underlier_parameters(options_data, date)
    portfolio = define_portfolio(options_data, params, date)
    print('\nOptions and underliers data sucessfully fetched!')

    # # Fetch parameters from config.py
    days = simulation_config["days"]
    simulations = simulation_config["simulations"]
    confidence_level = simulation_config["confidence_level"]

    # Compute Monte Carlo VaR
    print('\nCalculating VaR...')
    mc_var = MonteCarloVaR(portfolio, params, days, simulations, confidence_level)
    var_value = mc_var.estimate_var()

    print(f"Monte Carlo {int(confidence_level * 100)}% VaR for EQ Options Portfolio: {-var_value:,.2f} RUB")

