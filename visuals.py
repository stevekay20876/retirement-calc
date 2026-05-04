import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_wealth_trajectory(history, target_floor, years_arr, baseline_data=None):
    fig = go.Figure()
    
    if baseline_data is not None:
        base_hist = baseline_data['history']
        base_years = np.arange(baseline_data['start_year'], baseline_data['start_year'] + base_hist['total_bal_real'].shape[1])
        fig.add_trace(go.Scatter(x=base_years, y=np.median(base_hist['total_bal_real'], axis=0), mode='lines', name='Baseline Median', line=dict(color='gray', width=2, dash='dot')))
        
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['total_bal_real'], axis=0), mode='lines', name='Median (50th)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=years_arr, y=np.percentile(history['total_bal_real'], 90, axis=0), mode='lines', name='Optimistic (90th)', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=years_arr, y=np.percentile(history['total_bal_real'], 10, axis=0), mode='lines', name='Pessimistic (10th)', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=years_arr, y=[target_floor] * len(years_arr), mode='lines', name='Target Legacy Floor', line=dict(color='black', dash='dot')))
    
    fig.update_layout(
        title="Stochastic Portfolio Projections (In Today's Dollars)", 
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_fan_chart(history, years_arr):
    p10 = np.percentile(history['total_bal_real'], 10, axis=0)
    p25 = np.percentile(history['total_bal_real'], 25, axis=0)
    p50 = np.median(history['total_bal_real'], axis=0)
    p75 = np.percentile(history['total_bal_real'], 75, axis=0)
    p90 = np.percentile(history['total_bal_real'], 90, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=p90, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=years_arr, y=p10, mode='lines', fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', line=dict(width=0), name='10th-90th Pct Range'))
    fig.add_trace(go.Scatter(x=years_arr, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=years_arr, y=p25, mode='lines', fill='tonexty', fillcolor='rgba(70, 130, 180, 0.4)', line=dict(width=0), name='25th-75th Pct Range'))
    fig.add_trace(go.Scatter(x=years_arr, y=p50, mode='lines', name='Median Path (50th)', line=dict(color='darkblue', width=3)))
    
    fig.update_layout(
        title="Sequence of Return Risk Fan Chart (In Today's Dollars)", 
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_liquidity_timeline(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['home_value'], axis=0), mode='lines', stackgroup='one', name='Home Value', fillcolor='lightgray', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['tsp_bal'], axis=0), mode='lines', stackgroup='one', name='TSP'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['ira_bal'], axis=0), mode='lines', stackgroup='one', name='Trad IRA'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxable_bal'], axis=0), mode='lines', stackgroup='one', name='Taxable'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['hsa_bal'], axis=0), mode='lines', stackgroup='one', name='HSA'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['roth_bal'], axis=0), mode='lines', stackgroup='one', name='Roth IRA'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['cash_bal'], axis=0), mode='lines', stackgroup='one', name='Cash Buffer'))
    
    fig.update_layout(
        title="Total Net Worth Forecast (Nominal)", 
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_cash_flow_sources(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['salary_income'], axis=0), name="Salary", marker_color='purple'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['ss_income'], axis=0), name="Social Security", marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['pension_income'], axis=0), name="Total Pension", marker_color='#ff7f0e'))
    
    if 'va_income' in history:
        fig.add_trace(go.Bar(x=years_arr, y=np.median(history['va_income'], axis=0), name="VA Disability Pay", marker_color='#e377c2'))
        
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['tsp_withdrawal'], axis=0), name="TSP Withdrawal", marker_color='#2ca02c'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['ira_withdrawal'], axis=0), name="IRA Withdrawal", marker_color='#98df8a'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['roth_withdrawal'], axis=0), name="Roth Withdrawal", marker_color='green'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['hsa_withdrawal'], axis=0), name="HSA Withdrawal", marker_color='#17becf'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['taxable_withdrawal'], axis=0), name="Taxable Withdrawal", marker_color='orange'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['cash_withdrawal'], axis=0), name="Cash Withdrawal", marker_color='gray'))
    
    total_need = np.median(history['net_spendable'] + history['taxes_fed'] + history['taxes_state'] + history['health_cost'] + history['medicare_cost'] + history['mortgage_cost'] + history['additional_expenses'], axis=0)
    fig.add_trace(go.Scatter(x=years_arr, y=total_need, mode='lines', name='Total Spending Need', line=dict(color='black', width=2)))
    
    fig.update_layout(
        barmode='stack', 
        title="Income Sources vs Total Spending Need", 
        template="plotly_white"
    )
    return fig

def plot_income_gap(history, years_arr, baseline_data=None):
    va_inc = np.median(history['va_income'], axis=0) if 'va_income' in history else 0
    guaranteed_income = np.median(history['salary_income'] + history['ss_income'] + history['pension_income'], axis=0) + va_inc
    total_expenses = np.median(history['taxes_fed'] + history['health_cost'] + history['medicare_cost'] + history['mortgage_cost'] + history['additional_expenses'] + history['net_spendable'], axis=0)
    
    fig = go.Figure()
    
    if baseline_data is not None:
        base_hist = baseline_data['history']
        base_years = np.arange(baseline_data['start_year'], baseline_data['start_year'] + base_hist['taxes_fed'].shape[1])
        base_total_expenses = np.median(base_hist['taxes_fed'] + base_hist['health_cost'] + base_hist['medicare_cost'] + base_hist['mortgage_cost'] + base_hist['additional_expenses'] + base_hist['net_spendable'], axis=0)
        fig.add_trace(go.Scatter(x=base_years, y=base_total_expenses, mode='lines', name='Baseline Expenses', line=dict(color='pink', width=2, dash='dash')))
        
    fig.add_trace(go.Scatter(x=years_arr, y=total_expenses, mode='lines', name='Total Expenses / Lifestyle', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=years_arr, y=guaranteed_income, mode='lines', fill='tozeroy', name='Guaranteed Base', line=dict(color='blue')))
    
    fig.update_layout(
        title="Income Gap Mapping", 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_expenses_breakdown(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['taxes_fed'], axis=0), name="Federal/State Taxes", marker_color='crimson'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['medicare_cost'], axis=0), name="Medicare + IRMAA", marker_color='orange'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['health_cost'], axis=0), name="Health / OOP", marker_color='purple'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['mortgage_cost'], axis=0), name="Mortgage Payment", marker_color='brown'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['additional_expenses'], axis=0), name="Additional Expenses (Smile)", marker_color='pink'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['net_spendable'], axis=0), name="Discretionary Lifestyle", marker_color='teal'))
    
    fig.update_layout(
        barmode='stack', 
        title="Itemized Core Expenses", 
        template="plotly_white"
    )
    return fig

def plot_income_volatility(history, years_arr):
    med_spend = np.median(history['net_spendable'], axis=0)
    high_spend = np.percentile(history['net_spendable'], 90, axis=0)
    low_spend = np.percentile(history['net_spendable'], 10, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=high_spend, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=years_arr, y=low_spend, mode='lines', fill='tonexty', fillcolor='rgba(0,128,128,0.2)', line=dict(width=0), name='Spending Range (10th-90th)'))
    fig.add_trace(go.Scatter(x=years_arr, y=med_spend, mode='lines', name='Median Spending', line=dict(color='teal', width=3)))
    
    fig.update_layout(
        title="Variable Spending: Guardrail Adaptive Cash Flows", 
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_withdrawal_hierarchy(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['tsp_withdrawal'], axis=0), name="TSP Withdrawal", marker_color='#2ca02c'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['ira_withdrawal'], axis=0), name="Trad IRA Withdrawal", marker_color='#98df8a'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['roth_withdrawal'], axis=0), name="Roth Withdrawal", marker_color='green'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['hsa_withdrawal'], axis=0), name="HSA Withdrawal", marker_color='#17becf'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['taxable_withdrawal'], axis=0), name="Taxable Withdrawal", marker_color='orange'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['cash_withdrawal'], axis=0), name="Cash Withdrawal", marker_color='gray'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['extra_rmd'], axis=0), name="Reinvested RMD", marker_color='darkgray'))
    
    fig.update_layout(
        barmode='stack', 
        title="Dynamic Account Liquidation Hierarchy", 
        template="plotly_white"
    )
    return fig

def plot_taxes_and_rmds(history, years_arr, baseline_data=None):
    fig = go.Figure()
    
    if baseline_data is not None:
        base_hist = baseline_data['history']
        base_years = np.arange(baseline_data['start_year'], baseline_data['start_year'] + base_hist['taxes_fed'].shape[1])
        fig.add_trace(go.Scatter(x=base_years, y=np.median(base_hist['taxes_fed'], axis=0), mode='lines', name='Base Fed Taxes', line=dict(color='pink', dash='dash')))
        fig.add_trace(go.Scatter(x=base_years, y=np.median(base_hist['rmds'], axis=0), mode='lines', name='Base RMDs', line=dict(color='lightgoldenrodyellow', dash='dash')))
        
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxes_fed'], axis=0), mode='lines', fill='tozeroy', name='Federal Taxes', line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['rmds'], axis=0), mode='lines', name='RMD Volume', line=dict(color='orange', width=3)))
    
    fig.update_layout(
        title="Tax Trajectory vs Mandatory RMD Cliffs", 
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_legacy_breakdown(history):
    tsp_term = np.median(history['tsp_bal'][:, -1])
    ira_term = np.median(history['ira_bal'][:, -1])
    roth_term = np.median(history['roth_bal'][:, -1])
    hsa_term = np.median(history['hsa_bal'][:, -1])
    taxable_term = np.median(history['taxable_bal'][:, -1])
    cash_term = np.median(history['cash_bal'][:, -1])
    home_term = np.median(history['home_value'][:, -1])
    
    labels = ['TSP (Tax-Deferred)', 'Trad IRA (Tax-Deferred)', 'Roth IRA (Tax-Free)', 'Taxable Investments', 'Cash/MM', 'Real Estate', 'HSA']
    values = [max(0, tsp_term), max(0, ira_term), max(0, roth_term), max(0, taxable_term), max(0, cash_term), max(0, home_term), max(0, hsa_term)]
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#ff7f0e', '#d62728', 'gray', '#17becf']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
    fig.update_layout(
        title="Terminal Estate Composition (At Life Expectancy)",
        template="plotly_white"
    )
    return fig

def plot_roth_strategy_comparison(roth_results):
    strategies = list(roth_results.keys())
    wealths = [roth_results[s]['wealth'] for s in strategies]
    winner_idx = np.argmax(wealths)
    colors = ['gray'] * len(strategies)
    colors[winner_idx] = '#00837B'
    
    fig = px.bar(x=wealths, y=strategies, orientation='h', title="Strategic Scenario Comparison (Terminal Legacy in Today's $)")
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        xaxis_title="", 
        yaxis_title="",
        template="plotly_white"
    )
    return fig

def plot_roth_tax_impact(roth_results, winner, years_arr):
    fig = go.Figure()
    base_tax = roth_results['Baseline (None)']['tax_path']
    opt_tax = roth_results[winner]['tax_path']
    
    fig.add_trace(go.Scatter(x=years_arr, y=base_tax, mode='lines', fill='tozeroy', name='Baseline Lifetime Taxes', line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=years_arr, y=opt_tax, mode='lines', fill='tozeroy', name=f'Optimal ({winner}) Taxes', line=dict(color='#00837B')))
    
    fig.update_layout(
        title="Lifetime Tax Liability Comparison", 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_ss_breakeven(ss_fra, age_arr, years_arr, fra_age=67):
    early_stream = [(ss_fra * 0.7) * (0.79 if yr >= 2035 else 1.0) if age >= 62 else 0 for age, yr in zip(age_arr, years_arr)]
    fra_stream = [(ss_fra * 1.0) * (0.79 if yr >= 2035 else 1.0) if age >= fra_age else 0 for age, yr in zip(age_arr, years_arr)]
    delayed_stream = [(ss_fra * 1.24) * (0.79 if yr >= 2035 else 1.0) if age >= 70 else 0 for age, yr in zip(age_arr, years_arr)]
    
    cum_early = np.cumsum(early_stream)
    cum_fra = np.cumsum(fra_stream)
    cum_delayed = np.cumsum(delayed_stream)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=age_arr, y=cum_early, mode='lines', name='Claim at 62'))
    fig.add_trace(go.Scatter(x=age_arr, y=cum_fra, mode='lines', name=f'Claim at FRA ({fra_age})'))
    fig.add_trace(go.Scatter(x=age_arr, y=cum_delayed, mode='lines', name='Claim at 70'))
    
    fig.update_layout(
        title="Cumulative Guaranteed Income Breakeven Analysis", 
        xaxis_title="Age", 
        yaxis_title="",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_medicare_comparison(history, years_arr, inputs):
    fig = go.Figure()
    medicare_irmaa = np.median(history['medicare_cost'], axis=0)
    health_prem = np.median(history['health_cost'], axis=0)
    
    fig.add_trace(go.Scatter(x=years_arr, y=health_prem, mode='lines', stackgroup='one', name=f"{inputs['health_plan']} Premium + OOP", line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=years_arr, y=medicare_irmaa, mode='lines', stackgroup='one', name="Medicare Part B + IRMAA", line=dict(color='orange')))
    
    fig.update_layout(
        title="Lifetime Healthcare Cost Comparison", 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_tornado(base_legacy, sens_results):
    df = pd.DataFrame(sens_results)
    
    df['Swing'] = df['Positive Impact'].abs() + df['Negative Impact'].abs()
    df = df.sort_values('Swing', ascending=True)
    
    fig = go.Figure()
    
    def format_val(val):
        if val >= 0: return f"+${val:,.0f}"
        return f"-${abs(val):,.0f}"
    
    fig.add_trace(go.Bar(
        y=df['Factor'],
        x=df['Negative Impact'],
        orientation='h',
        name='Downside Risk',
        marker_color='indianred',
        text=[format_val(x) for x in df['Negative Impact']],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        y=df['Factor'],
        x=df['Positive Impact'],
        orientation='h',
        name='Upside Potential',
        marker_color='mediumseagreen',
        text=[format_val(x) for x in df['Positive Impact']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Sensitivity Analysis (Impact on Median Legacy vs Baseline: ${base_legacy:,.0f})",
        barmode='relative',
        xaxis_title="Change in Terminal Liquid Legacy ($)",
        yaxis_title="",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig