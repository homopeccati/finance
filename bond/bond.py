# Set project root dynamically
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from scipy.optimize import fsolve
from utils.curve import Curve

class Bond:
    """Bond class defining the main input parameters and providing for the accured interest and remaining cash flows calculation.
    """    
    def __init__(self, face_value: int, coupon_rate: int, years_to_maturity: int, issue_date: date, settlement_date: date, payment_frequency: date, currency: str, credit_spread: float):
        """Constructor for Bond class
        """
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.issue_date = datetime.strptime(issue_date, '%d.%m.%Y').date()
        self.settlement_date = datetime.strptime(settlement_date, '%d.%m.%Y').date()
        self.payment_frequency = payment_frequency
        self.currency = currency
        self.credit_spread = credit_spread

    def calculate_accrued_interest(self) -> float:
        """Calculate the accrued interest for the bond. 
        The accrued interest is calculated as the product of the face value, coupon rate, and the fraction of the period since the last coupon payment.

        Returns:
            float: Accrued interest for the bond (in currency units)
        """        
        days_in_period = 365 // self.payment_frequency
        last_coupon_date = self.issue_date

        while last_coupon_date + timedelta(days=days_in_period) <= self.settlement_date:
            last_coupon_date += timedelta(days=days_in_period)

        accrued_days = (self.settlement_date - last_coupon_date).days
        accrued_interest = self.face_value * (self.coupon_rate / self.payment_frequency) * (accrued_days / days_in_period)
        return accrued_interest

    def get_remaining_cash_flows(self) -> tuple:
        """Get the remaining cash flows and payment schedule for the bond. 
        The cash flows are calculated as the coupon payments and the face value payment at maturity. 
        The payment schedule is calculated based on the payment frequency. 

        Returns:
            tuple: Remaining cash flows (list) and payment schedule (as the dates list) for the bond
        """        
        coupon_payment = self.face_value * (self.coupon_rate / self.payment_frequency)
        cash_flows = []
        payment_dates = []

        for i in range(1, self.years_to_maturity * self.payment_frequency + 1):
            payment_date = self.issue_date + timedelta(days=(365 // self.payment_frequency) * i)
            if payment_date > self.settlement_date:
                cash_flow = coupon_payment
                if i == self.years_to_maturity * self.payment_frequency:
                    cash_flow += self.face_value
                cash_flows.append(cash_flow)
                payment_dates.append(payment_date)

        return cash_flows, payment_dates

class BondCalculator:
    """BondCalculator class for calculating the dirty price, yield to maturity (YTM), and modified duration for a bond
    """    
    def __init__(self, bond, curve):
        """Constructor for BondCalculator class. 
        Defines the bond and curve objects for the calculations."""
        self.bond = bond
        self.curve = curve

    def calculate_dirty_price(self):
        """Calculate the dirty price for the bond. 
        The dirty price is calculated as the sum of the present values of the remaining cash flows and the accrued interest.

        Returns:
            float: Dirty price for the bond (in currency units)
        """        
        cash_flows, payment_dates = self.bond.get_remaining_cash_flows()
        dirty_price = 0

        for cash_flow, payment_date in zip(cash_flows, payment_dates):
            t = (payment_date - self.bond.settlement_date).days / 365
            discount_rate = self.curve.get_discount_rate(t, self.bond.credit_spread)
            dirty_price += cash_flow / (1 + discount_rate/self.bond.payment_frequency) ** (t*self.bond.payment_frequency)

        accrued_interest = self.bond.calculate_accrued_interest()
        dirty_price += accrued_interest

        return dirty_price
    
    def calculate_ytm(self):
        """Calculate the yield to maturity (YTM) for the bond. 
        Uses the fsolve function from the scipy library to solve the equation for the YTM.

        Returns:
            float: Yield to maturity (YTM) for the bond (as a percentage)
        """        
        cash_flows, payment_dates = self.bond.get_remaining_cash_flows()
        dirty_price = self.calculate_dirty_price()

        def price_error(ytm: float):
            """Calculate the error between the calculated price and the actual price.

            Args:
                ytm (float): Yield to maturity (YTM) for the bond

            Returns:
                float: Error between the calculated price and the actual price
            """            
            return sum(cf / (1 + ytm / self.bond.payment_frequency)**(((t - self.bond.settlement_date).days / 365) * self.bond.payment_frequency) for cf, t in zip(cash_flows, payment_dates)) - dirty_price

        ytm = fsolve(price_error, 0.01)[0]
        return ytm

    def calculate_modified_duration(self):
        """Calculate the modified duration for the bond. 
        The modified duration is calculated as the Macaulay duration divided by (1 + YTM / payment frequency).

        Returns:
            float: Modified duration for the bond.
        """        
        cash_flows, payment_dates = self.bond.get_remaining_cash_flows()
        dirty_price = self.calculate_dirty_price()
        ytm = self.calculate_ytm()

        macaulay_duration = sum(((t - self.bond.settlement_date).days / 365) * cf / dirty_price for cf, t in zip(cash_flows, payment_dates))
        modified_duration = macaulay_duration / (1 + ytm / self.bond.payment_frequency)
        return modified_duration

def read_bond_data(file_path: str):
    """Read bond data from a CSV file. Can be used for more than one bond at a time.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list: List of Bond objects created from the CSV data.
    """    
    try:
        bond_data = pd.read_csv(file_path)
        bonds = [Bond(
            face_value = row['Face Value'],
            coupon_rate = row['Coupon Rate'],
            years_to_maturity = row['Years to Maturity'],
            issue_date = row['Issue Date'],
            settlement_date = row['Settlement Date'],
            payment_frequency = row['Payment Frequency'],
            currency = row['Currency'],
            credit_spread = float(row['Credit Spread'].replace('%',''))/100
        ) for _, row in bond_data.iterrows()]
        return bonds
    except Exception as e:
        print(f"Error reading bond data: {e}")
        return []

def read_yield_curve(file_path: str):
    """Read yield curve data from a CSV file and create a Curve object.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Curve: Curve object created from the CSV data.
    """    
    try:
        curve_data = pd.read_csv(file_path)
        rates = dict(zip(curve_data['Term'], curve_data['Rate'].str.replace('%','').astype(float)/100))
        return Curve(rates)
    except Exception as e:
        print(f"Error reading yield curve data: {e}")
        return None

if __name__ == "__main__":
    """Main function for the bond analysis. Reads bond data and yield curve data from CSV files and performs the analysis for each bond.
    """
    try:
        bond_csv_path = input("Enter the path to the bond data CSV file: ")
        yield_curve_csv_path = input("Enter the path to the yield curve CSV file: ")

        bonds = read_bond_data(bond_csv_path)
        yield_curve = read_yield_curve(yield_curve_csv_path)

        if not bonds or yield_curve is None:
            raise ValueError("Failed to read bond data or yield curve data.")

        for i, bond in enumerate(bonds, start=1):
            calculator = BondCalculator(bond, yield_curve)
            print(f"Bond {i} Analysis:")
            print(f"  Accrued Interest: {bond.calculate_accrued_interest():.2f}")
            dirty_price = calculator.calculate_dirty_price()
            print(f"  Dirty Price: {dirty_price:.2f}")
            ytm = calculator.calculate_ytm()
            print(f"  Yield to Maturity (YTM): {ytm:.2%}")
            modified_duration = calculator.calculate_modified_duration()
            print(f"  Modified Duration: {modified_duration:.4f}\n")
    except Exception as e:
        print(f"Error in bond analysis: {e}")
