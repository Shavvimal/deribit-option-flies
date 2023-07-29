# Deribit Options Data: Analyzing Option Flies with Python

In summary, this script is a powerful tool for fetching options data from the Deribit exchange, calculating option spreads (flies), and presenting the results in a structured pandas DataFrame. It can be useful for traders and analysts who want to perform quantitative analysis on options data and explore various trading strategies:

```python
import warnings
from datetime import datetime, timedelta, timezone
import aiohttp
import asyncio
import json
import pandas as pd
from dateutil import parser
import os
import time
import sys
import numpy as np

warnings.filterwarnings('ignore')

class deribit_async:
    ''' Class for deribit options collection'''
    # main, run flies_iterator()

    def __init__(self):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    @staticmethod
    def transform_subset(subset: pd.DataFrame):
        '''Transforms a subset of the options data (having been grouped by Put/Call and Expiry) by creating columns for strike shifts'''
        # step = (subset.strike - subset.strike.shift(1)).min()
        step = 250
        min_strike = subset.strike.min()
        max_strike = subset.strike.max()
        new_index = pd.Index(
            np.arange(min_strike, max_strike+step, step, dtype=int))

        subset = subset.set_index("strike").reindex(
            new_index).reset_index().rename(columns={'index': 'strike'})

        subset['mark_strike_minus_2000'] = subset.mark_price.shift(8)
        subset['mark_strike_plus_2000'] = subset.mark_price.shift(-8)

        subset['mark_strike_minus_1000'] = subset.mark_price.shift(4)
        subset['mark_strike_plus_1000'] = subset.mark_price.shift(-4)

        subset['mark_strike_minus_500'] = subset.mark_price.shift(2)
        subset['mark_strike_plus_500'] = subset.mark_price.shift(-2)

        return subset

    @staticmethod
    def flies(mark_price, mark_strike_minus_width, mark_strike_plus_width):
        '''Function to create Flies and return price for certain fly width'''
        return mark_strike_minus_width - 2*(mark_price) + mark_strike_plus_width

    def flies_calc_subset(self, subset_input: pd.DataFrame, PC: str):
        '''Will return 2000, 1000 and 500 flies for a subset of the options data'''
        subset = subset_input
        subset[f'2000{PC}F'] = subset.apply(lambda row: self.flies(
            row['mark_price'], row['mark_strike_minus_2000'], row['mark_strike_plus_2000']), axis=1)
        subset[f'1000{PC}F'] = subset.apply(lambda row: self.flies(
            row['mark_price'], row['mark_strike_minus_1000'], row['mark_strike_plus_1000']), axis=1)
        subset[f'500{PC}F'] = subset.apply(lambda row: self.flies(
            row['mark_price'], row['mark_strike_minus_500'], row['mark_strike_plus_500']), axis=1)

        # drop the rows where the flies are NaN
        df = subset[['Time_UTC', 'underlying_price', 'strike',
                     'mark_price', f'2000{PC}F', f'1000{PC}F', f'500{PC}F']]
        df = df.dropna(subset=['underlying_price', 'mark_price'])

        return df

    def flies_calculation_orchestrator(self, df: pd.DataFrame, PC: str):
        '''Will return 2000, 1000 and 500 flies for the subset provided'''
        df = self.transform_subset(df)
        df = self.flies_calc_subset(df, PC)
        return df

    async def flies_iterator(self, currency: str = "BTC", kind: str = "option"):

        df = await self.transform_option_data(currency, kind)
        d = []

        for exp in df.expiry.unique().tolist():
            for put_call in ['P', 'C']:
                subset = df[(df.expiry == exp) & (df.PC == put_call)].sort_values(
                    by=['strike'])[['strike', 'mark_price', 'underlying_price', 'Time_UTC']]
                df_ret = self.flies_calculation_orchestrator(subset, put_call)
                df_ret['PC'] = put_call
                df_ret['expiry'] = exp
                d.extend(df_ret.to_dict(orient='records'))
        df_transformed = pd.DataFrame(d)

        res = df_transformed[['Time_UTC', 'underlying_price', 'strike', 'PC', 'expiry',
                              'mark_price', '2000PF', '1000PF', '500PF', '2000CF', '1000CF', '500CF']]

        puts = res[res['PC'] == 'P']
        puts.drop(columns=['2000CF', '1000CF', '500CF'], inplace=True)
        puts['estimated_probability_2000PF'] = (
            puts['2000PF'] * puts['underlying_price']) / 2000
        puts['estimated_probability_1000PF'] = (
            puts['1000PF'] * puts['underlying_price']) / 1000
        puts['estimated_probability_500PF'] = (
            puts['500PF'] * puts['underlying_price']) / 500

        calls = res[res['PC'] == 'C']
        calls.drop(columns=['2000PF', '1000PF', '500PF'], inplace=True)
        calls['estimated_probability_2000CF'] = (
            calls['2000CF'] * calls['underlying_price']) / 2000
        calls['estimated_probability_1000CF'] = (
            calls['1000CF'] * calls['underlying_price']) / 1000
        calls['estimated_probability_500CF'] = (
            calls['500CF'] * calls['underlying_price']) / 500

        df = pd.merge(puts, calls, how="outer", on=[
                      "strike", "expiry"], suffixes=('_puts', '_calls'))

        df['expiry'] = pd.to_datetime(df.expiry, format="%d%b%y")
        df = df.sort_values(['expiry', 'strike'], ascending=[True, True])

        # Save df as csv
        df.to_csv(f'./{currency}_flies.csv', index=False)

    async def transform_option_data(self, currency: str = "BTC", kind: str = "option"):
        '''Calls get_option_data() and transforms as a pandas dataframe. Returns raw data'''
        data = await self.get_option_data(currency, kind)
        df = pd.DataFrame.from_records(data)
        df = df[['creation_timestamp', 'instrument_name',
                 'mark_price', 'underlying_price']]
        df['Time_UTC'] = pd.to_datetime(df['creation_timestamp'], unit='ms')
        df[['asset', 'expiry', 'strike', 'PC']
           ] = df.instrument_name.str.split('-', expand=True)
        df['strike'] = df['strike'].astype(int)
        return df

    async def get_option_data(self, currency: str = "BTC", kind: str = "option"):
        '''Will retrieve option data from deribit'''
        url = f'https://deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind={kind}'
        async with aiohttp.ClientSession() as session:
            url_params = {'currency': currency, 'kind': kind}
            data = await self.http_call(url, url_params, session)
        await session.close()
        return data

    async def http_call(self, url, url_params, session):
        ''' Async API request. Returns Data from provided url and url Params'''
        try:
            async with session.get(url=url, params=url_params) as response:
                resp = await response.read()
                response = json.loads(resp)
                data = response['result']
                await session.close()
            return data
        except Exception as e:
            print(
                f"\nUnable to get data from url: \n{p.CYAN}{url}{p.ENDC} \ndue to: \n{p.FAIL}{e}\n{response}{p.ENDC}")


deribit = deribit_async()
asyncio.run(deribit.flies_iterator())

```

The `flies_iterator` method is an asynchronous function that orchestrates the process of collecting options data and calculating various option spreads (flies). It first calls the `transform_option_data` method to fetch raw options data from Deribit and transform it into a pandas DataFrame. Then, for each unique expiry date and put/call combination, it calculates the 2000, 1000, and 500 flies using the `flies_calculation_orchestrator` method.
Finally, it merges the put and call fly data into a single DataFrame and performs some additional calculations on the data before returning it.

The script creates an instance of the `deribit_async` class, named deribit, and it then runs the flies_iterator method asynchronously using `asyncio.run(deribit.flies_iterator())`.

What it returns should look like:
|Time_UTC_puts|underlying_price_puts|strike|PC_puts|expiry |mark_price_puts|2000PF |1000PF |500PF |estimated_probability_2000PF|estimated_probability_1000PF|estimated_probability_500PF|Time_UTC_calls|underlying_price_calls|PC_calls|mark_price_calls|2000CF |1000CF |500CF |estimated_probability_2000CF|estimated_probability_1000CF|estimated_probability_500CF|
|-------------|---------------------|------|-------|----------|---------------|----------|----------|-----------------------|----------------------------|----------------------------|---------------------------|--------------|----------------------|--------|----------------|----------|----------|-----------------------|----------------------------|----------------------------|---------------------------|
|2023-07-29 22:19:10.724|30064.46 |5000 |P |2023-12-29|0.00065208 | | | | | | |2023-07-29 22:19:10.723|30064.46 |C |0.83434536 | | | | | | |
|2023-07-29 22:19:10.713|30064.46 |10000 |P |2023-12-29|0.00139242 | | | | | | |2023-07-29 22:19:10.724|30064.46 |C |0.66877898 | | | | | | |
|2023-07-29 22:19:10.723|30064.46 |11000 |P |2023-12-29|0.00177016 | |1.8549999999999556e-05| | |0.0005576957329999866 | |2023-07-29 22:19:10.717|30064.46 |C |0.63589537 | |1.8550000000061573e-05| | |0.0005576957330018511 | |
|2023-07-29 22:19:10.719|30064.46 |12000 |P |2023-12-29|0.00216645 |0.0003844300000000003|5.299999999998882e-07| |0.005778840178900004 |1.5934163799996638e-05 | |2023-07-29 22:19:10.724|30064.46 |C |0.60303031 |0.00038445000000009166|5.400000000488347e-07| |0.0057791408235013785 |1.623480840146819e-05 | |
|2023-07-29 22:19:10.712|30064.46 |13000 |P |2023-12-29|0.00256327 |0.0011050499999999998|0.0003648200000000001| |0.016611365761499998 |0.010968116297200002 | |2023-07-29 22:19:10.713|30064.46 |C |0.57016579 |0.0011050499999999408|0.0003648199999999324| |0.01661136576149911 |0.010968116297197968 | |
|2023-07-29 22:19:10.724|30064.46 |14000 |P |2023-12-29|0.00332491 |0.0009406099999999997|0.0003748800000000002| |0.014139465860299997 |0.011270564764800005 | |2023-07-29 22:19:10.723|30064.46 |C |0.53766609 |0.0009406000000000136|0.0003748700000000271| |0.014139315538000205 |0.011270264120200815 | |
|2023-07-29 22:19:10.724|30064.46 |15000 |P |2023-12-29|0.00446143 |0.0005547199999999999|-0.000173969999999999| |0.008338678625599998 |-0.00523031410619997 | |2023-07-29 22:19:10.713|30064.46 |C |0.50554126 |0.0005547200000000085|-0.00017395999999997303| |0.008338678625600128 |-0.005230013461599189 | |
|2023-07-29 22:19:10.712|30064.46 |16000 |P |2023-12-29|0.00542398 |0.001148370000000001|0.0005277799999999994| |0.017262561965100016 |0.01586742069879998 | |2023-07-29 22:19:10.716|30064.46 |C |0.47324247 |0.0011483599999999594|0.0005277699999999275| |0.01726241164279939 |0.01586712005419782 | |
