class Curve:
    """Curve class for calculating the discount rates based on the given curve data.
    """    
    def __init__(self, curve: dict):
        """Constructor for Curve class. 
        Defines the curve data as a dictionary with tenors as keys and rates as values.
        """
        self.curve = curve
    
    def get_discount_rate(self, years: int, credit_spread: float = 0.0) -> float:
        """Get the discount rate for a given number of periods. 
        Interpolates the rate if the period is not in the curve data. 

        Args:
            years (int): The period for which the discount rate is calculated
            credit_spread (float, optional): Credit spread to be added to the discount rate. Defaults to 0.0.

        Returns:
            float: Discount rate for the given period
        """        
        tenors = sorted(self.curve.keys())
        if years <= tenors[0]:
            return self.curve[tenors[0]] + credit_spread
        if years >= tenors[-1]:
            return self.curve[tenors[-1]] + credit_spread

        for i in range(len(tenors) - 1):
            if tenors[i] <= years < tenors[i + 1]:
                rate1, rate2 = self.curve[tenors[i]], self.curve[tenors[i + 1]]
                interpolated_rate = rate1 + (rate2 - rate1) * (years - tenors[i]) / (tenors[i + 1] - tenors[i])
                return interpolated_rate + credit_spread
