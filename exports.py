import numpy as np
import pandas as pd

def build_csv_dataframe(history, years_arr, age_arr, percentile=50):
    data = {
        "Calendar Year": years_arr,
        "Age": age_arr,
        "Rate of Return": np.percentile(history['port_return'], percentile, axis=0),
        "Inflation Rate": np.percentile(history['inflation'], percentile, axis=0),
        "Real Rate of Return": np.percentile(history['real_return'], percentile, axis=0),
        "Cumulative Inflation Multiplier": np.percentile(history['cum_inf'], percentile, axis=0), 
        "Home Value": np.percentile(history['home_value'], percentile, axis=0),
        "Taxable ETF Balance": np.percentile(history['taxable_bal'], percentile, axis=0),
        "Roth IRA Balance": np.percentile(history['roth_bal'], percentile, axis=0),
        "Trad IRA Balance": np.percentile(history['ira_bal'], percentile, axis=0),
        "HSA Balance": np.percentile(history['hsa_bal'], percentile, axis=0),
        "Money Market Balance": np.percentile(history['cash_bal'], percentile, axis=0),
        "Remaining TSP Account": np.percentile(history['tsp_bal'], percentile, axis=0), 
        "TSP Withdrawal": np.percentile(history['tsp_withdrawal'], percentile, axis=0), 
        "Trad IRA Withdrawal": np.percentile(history['ira_withdrawal'], percentile, axis=0), 
        "Annual Roth IRA Withdrawal": np.percentile(history['roth_withdrawal'], percentile, axis=0),    
        "Annual Taxable Withdrawal": np.percentile(history['taxable_withdrawal'], percentile, axis=0),  
        "Annual Cash/MM Withdrawal": np.percentile(history['cash_withdrawal'], percentile, axis=0),      
        "Salary Income": np.percentile(history['salary_income'], percentile, axis=0),
        "Total Pension (FERS + Mil)": np.percentile(history['pension_income'], percentile, axis=0),
        "VA Disability Pay": np.percentile(history['va_income'], percentile, axis=0) if 'va_income' in history else np.zeros(len(years_arr)),
        "Social Security": np.percentile(history['ss_income'], percentile, axis=0),
        "RMD Amount": np.percentile(history['rmds'], percentile, axis=0),
        "Extra RMD Amount": np.percentile(history['extra_rmd'], percentile, axis=0),
        "Roth Conversion Amount": np.percentile(history['roth_conversion'], percentile, axis=0),
        "Roth Taxes Paid from Cash": np.percentile(history['roth_taxes_from_cash'], percentile, axis=0), 
        "IRS Taxable Income": np.percentile(history['taxable_income'], percentile, axis=0), 
        "MAGI (IRMAA Base)": np.percentile(history['magi'], percentile, axis=0), 
        "Federal Taxes": np.percentile(history['taxes_fed'], percentile, axis=0),
        "State Taxes": np.percentile(history['taxes_state'], percentile, axis=0),
        "Medicare Cost": np.percentile(history['medicare_cost'], percentile, axis=0),
        "Health Insurance Cost": np.percentile(history['health_cost'], percentile, axis=0),
        "Additional Expenses (Smile Curve)": np.percentile(history['additional_expenses'], percentile, axis=0),
        "Total Expenses": np.percentile(history['taxes_fed'] + history['taxes_state'] + history['medicare_cost'] + history['health_cost'] + history['mortgage_cost'] + history['additional_expenses'], percentile, axis=0),
        "Net Spendable Annual": np.percentile(history['net_spendable'], percentile, axis=0),
        "Total Income": np.percentile(history['net_spendable'] + history['taxes_fed'] + history['taxes_state'] + history['medicare_cost'] + history['health_cost'] + history['mortgage_cost'] + history['additional_expenses'], percentile, axis=0),
        "Ending Total Balance (Nominal $)": np.percentile(history['total_bal'], percentile, axis=0),
        "Ending Total Balance (Today's $)": np.percentile(history['total_bal_real'], percentile, axis=0), 
        "Withdrawal Constraint Active": ["Yes" if flag > 0 else "No" for flag in np.percentile(history['constraint_active'], percentile, axis=0)]
    }
    
    df = pd.DataFrame(data)
    
    # By checking if the maximum absolute value is less than 0.0001, we mathematically bypass floating point inaccuracy bugs.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_drop = [col for col in numeric_cols if df[col].abs().max() < 1e-4]
    
    # Ensure we never accidentally drop the Calendar Year or Age columns
    cols_to_drop = [c for c in cols_to_drop if c not in ["Calendar Year", "Age"]]
    df = df.drop(columns=cols_to_drop)
    
    return df