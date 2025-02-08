# Set project root dynamically
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from utils.curve import Curve #Imported from bond.py

class CrossCurrencySwap:
    """CrossCurrencySwap class for calculating fixed rate for a cross currency swap.
    """    
    def __init__(self, notional: float, term: int, base_currency: str, quote_currency: str, spot_rate: float, swap_curves: dict, payment_frequency: int, floating_rate: float) -> None:
        """Constructor for CrossCurrencySwap class. 

        Args:
            notional (float): Notional amount in base currency
            term (int): Term of the swap in years
            base_currency (str): Base currency. The floating leg of the swap is in this currency.
            quote_currency (str): Quote currency. The fixed leg of the swap is in this currency.
            spot_rate (float): Spot exchange rate (base currency per unit of quote currency). Assumed to be constant throughout the term of the swap (otherwise a forward curve would be required as an input).
            swap_curves (dict): Dictionary of swap curves for base and quote currencies.
            payment_frequency (int): Frequency of cash flow exchange per year.
            floating_rate(int): Floating rate for the base currency at the swap inception.
        """
        self.notional = notional
        self.term = term
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.spot_rate = spot_rate
        self.swap_curves = swap_curves
        self.payment_frequency = payment_frequency 
        self.floating_rate = floating_rate

    def get_discount_factor(self, currency: str, tenor: int) -> float:
        """Get discount factor for a given currency and tenor. 
        The discount factor is calculated using the swap curve.

        Args:
            currency (str): Currency
            tenor (int): Tenor

        Returns:
            float: Discount factor for the given currency and tenor
        """       
        try:
            curve = self.swap_curves[currency]
            rate = curve.get_discount_rate(tenor)
            return 1 / (1 + rate/self.payment_frequency)**(tenor * self.payment_frequency)
        except KeyError:
            raise ValueError(f"Swap curve for currency {currency} not found.")
        except Exception as e:
            raise RuntimeError(f"Error calculating discount factor: {e}")
    
    def calculate_fixed_rate(self) -> float:
        """Calculate fixed rate for the swap. 
        The fixed rate is calculated as the rate that makes the present value of the floating leg equal to the present value of the fixed leg.
        In order to calculate the fixed rate, the present value of the floating leg is calculated by discounting the cash flows in base currency using the discount factors for the same currency based on the swap curves and the floating rate at the inception of the swap.
        Fixed rate is determined as the ratio of the floating leg's present value to the product of notional and the sum of discount factors for the quote currency. 
        Returns:
            float: Fixed rate for the swap
        """
        try:
            floating_leg_pv = 0
            fixed_leg_pv = 0
            annuity_factor_fixed_leg = 0

            #Cash flow exchange during the term of the swap.
            for i in range(1, self.term*self.payment_frequency + 1):
                time_to_cash_flow = i / self.payment_frequency
                discount_factor_base = self.get_discount_factor(self.base_currency, time_to_cash_flow)
                discount_factor_quote = self.get_discount_factor(self.quote_currency, time_to_cash_flow)

                floating_leg_pv += self.notional*floating_rate * discount_factor_base
                annuity_factor_fixed_leg += discount_factor_quote
            
            #Notional exchange at the end of the swap.
            discount_factor_base_expiration = self.get_discount_factor(self.base_currency, self.term)
            discount_factor_quote_expiration = self.get_discount_factor(self.quote_currency, self.term)
            
            floating_leg_pv += self.notional * discount_factor_base_expiration

            #Fixed rate calculation. Since the notional exchange at the start and end of the swap is the same, it cancels out in the ratio. 
            #Notional is discounted with a quote currency factor and subtracted from the floating leg PV to account for the difference in the disounting factors for the notional exchange at the expiration.
            fixed_rate = ((floating_leg_pv-self.notional*discount_factor_quote_expiration)/self.spot_rate) / (annuity_factor_fixed_leg * (self.notional/self.spot_rate))
            
            return fixed_rate
        except Exception as e:
            raise RuntimeError(f"Error calculating fixed rate: {e}")

def read_swap_curves(file_path: str) -> dict:
    """Read swap curves from a CSV file. 
    The CSV file should have columns 'Term', 'Currency1', 'Currency2', etc.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        dict: Dictionary of swap curves for different currencies (key - currency, value - Curve object)
    """    
    try:
        curves = pd.read_csv(file_path)
        curves = curves.melt(id_vars=['Term'], var_name='Currency', value_name='Rate')

        swap_curves = {}

        for currency in curves['Currency'].unique():
            currency_curve = (curves[curves['Currency'] == currency].set_index('Term')['Rate'].str.replace('%','').astype(float)/100).to_dict()
            swap_curves[currency] = Curve(currency_curve)

        return swap_curves
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File {file_path} is empty.")
    except Exception as e:
        raise RuntimeError(f"Error reading swap curves: {e}")

if __name__ == "__main__":
    """Main function for running the cross currency swap calculation. 
    """
    try:
    # Uncomment below for the custom input
        # notional = float(input("Enter the notional amount (in base currency): "))
        # term = float(input("Enter the term of the swap (in years in float format, e.g. 1.0): "))
        # frequency = int(input("Enter the frequency of cash flow exchange per year (e.g., 1 for yearly): "))
        # spot_rate = float(input("Enter the spot exchange rate (quote currency per unit of base currency): "))
        # base_currency = input("Enter the base currency (e.g., RUB): ").strip().upper()
        # quote_currency = input("Enter the quote currency (e.g., CNY): ").strip().upper()
        # file_path = input("Enter the path to the swap curve CSV file: ").strip()
        # floating_rate = float(input("Enter the floating rate for the base currency at the swap inception: "))

        notional = float(1000000000)
        term = int(1)
        frequency = int(4)
        spot_rate = float(12.8)
        base_currency = str('RUB')
        quote_currency = str('CNY')
        floating_rate = float(0.16)
        file_path = 'swaps/swap_curves.csv'

        swap_curves = read_swap_curves(file_path)

        swap = CrossCurrencySwap(notional, term, base_currency, quote_currency, spot_rate, swap_curves, frequency, floating_rate)
        fixed_rate = swap.calculate_fixed_rate()
        print(f"The fixed rate (swap quotation) for the {quote_currency} leg is: {fixed_rate:.4%}")
    except Exception as e:
        print(f"An error occurred: {e}")
