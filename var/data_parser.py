import pandas as pd
import requests
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from io import StringIO

# Function to fetch stock price data in chunks for multiple tickers
def fetch_underlier_data(tickers: list, start_date: str, end_date: str, chunk_size=100) -> pd.DataFrame:
    """Fetch stock price data in chunks for multiple tickers.

    Args:
        tickers (list): List of stock tickers to fetch data for.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        chunk_size (int, optional): Number of days to fetch in each API call. Defaults to 100. 

    Returns:
        pd.DataFrame: DataFrame containing the fetched stock price data.
    """    
    all_data = pd.DataFrame()

    # Convert start and end date strings to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    for ticker in tickers:
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
        
        # Loop to fetch data in chunks
        current_start_date = start_date
        while current_start_date < end_date:
            # Prepare the date range for the current chunk
            next_date = current_start_date + timedelta(days=chunk_size)
            if next_date > end_date:
                next_date = end_date

            # Format dates as strings for the API request
            start_date_str = current_start_date.strftime('%Y-%m-%d')
            next_date_str = next_date.strftime('%Y-%m-%d')

            # Prepare the API request parameters
            params = {
                'marketprice_board': 1,
                'from': start_date_str,
                'till': next_date_str,
            }

            # Send the API request
            response = requests.get(url, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()  # Assuming the data is in JSON format
                data = pd.DataFrame(data['history']['data'], columns=data['history']['columns'])
                all_data = pd.concat([all_data, data])
                print(f"Fetched data for {ticker} from {start_date_str} to {next_date_str}")
            else:
                print(f"Failed to fetch data for {ticker} from {start_date_str} to {next_date_str}")
                break  # You can handle retries here if needed

            # Move to the next chunk
            current_start_date = next_date

    return all_data

def fetch_option_data(tickers: list, date: str) -> pd.DataFrame:
    """Fetch option data for multiple tickers.

    Args:
        tickers (list): List of option tickers to fetch data for.
        date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing the fetched option data.
    """    
    final_data = pd.DataFrame()
    for ticker in tickers:
        url = f"https://iss.moex.com/iss/engines/futures/markets/options/securities/{ticker}.json"
        
        params = {
            date:'2025-01-31',
            'marketprice_board': 1
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        data = pd.DataFrame(data['securities']['data'], columns=data['securities']['columns'])
        final_data = pd.concat([final_data if not final_data.empty else None,
                                data if not data.empty else None])
    
    return final_data

def fetch_risk_free_rate() -> dict:
    """Fetch the latest risk-free rates from the Bank of Russia website.

    Raises:
        Exception: If the request to the Bank of Russia website fails.
        Exception: If the yield curve table is not found on the website.

    Returns:
        dict: A dictionary containing the latest risk-free rates.
    """    
    
    url = "https://www.cbr.ru/eng/hd_base/zcyc_params/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from Bank of Russia")
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Locate the latest table with the yield data
    table = soup.find("table")
    if table is None:
        raise Exception("Could not find the yield curve table")
    
    # Extract table data into a DataFrame
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Rename columns for clarity (based on the structure of the table)
    df.columns = ["Date", 0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    
    # Get the most recent data
    latest_data = (df.iloc[-1,1:]/100).to_dict()

    return latest_data
    
def ticker_mapping() -> pd.DataFrame:
    """Fetch the mapping of stock tickers to option tickers from the MOEX website.

    Returns:
        pd.DataFrame: DataFrame containing the mapping of stock tickers to option tickers.
        
    Note: This function is not implemented in the final version of the project.
    """    
    url = 'https://www.moex.com/s205'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0'}
    response = requests.get(url, headers = headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'table1'})
    df = pd.read_html(str(table))[0]
    df.columns = ['Group','Ticker Stock','Ticker Option','Underlier']
    
    return df

