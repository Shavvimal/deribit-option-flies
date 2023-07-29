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
