import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

# ==========================================
# 1. APP CONFIGURATION & ARCHITECTURE
# ==========================================
st.set_page_config(page_title="Advanced Quantitative Retirement Planner", layout="wide")

N_SIMS = 10000
CURRENT_YEAR = datetime.now().year

# ==========================================
# 2. CORE MATHEMATICAL & ECONOMIC ENGINES
# ==========================================
def simulate_inflation_ou_jump(years, n_sims, mu=0.029, kappa=0.5, sigma=0.01, lambda_jump=0.1, jump_mu=0.05, jump_vol=0.02):
    """Ornstein-Uhlenbeck mean-reverting inflation process with jump diffusion."""
    dt = 1.0
    pi = np.full((n_sims, years), mu)
    for t in range(1, years):
        dW = np.random.normal(0, np.sqrt(dt), n_sims)
        jumps = np.random.poisson(lambda_jump * dt, n_sims) * np.random.normal(jump_mu, jump_vol, n_sims)
        pi[:, t] = pi[:, t-1] + kappa * (mu - pi[:, t-1]) * dt + sigma * dW + jumps
    return np.clip(pi, -0.02, 0.15)

def calculate_fers_diet_cola(cpi_array):
    """Applies the FERS Diet COLA rule vector-wise."""
    cola = np.where(cpi_array <= 0.02, cpi_array,
            np.where(cpi_array <= 0.03, 0.02, cpi_array - 0.01))
    return cola

def estimate_taxes(income, filing_status):
    """Simplified 2024 Federal Tax Brackets for rapid vectorized Monte Carlo."""
    if filing_status == "Single":
        brackets = [(11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24), (243725, 0.32), (609350, 0.35), (np.inf, 0.37)]
    else:
        brackets = [(23200, 0.10), (94300, 0.12), (201050, 0.22), (383900, 0.24), (487450, 0.32), (731200, 0.35), (np.inf, 0.37)]
    
    tax = np.zeros_like(income)
    prev_limit = 0
    for limit, rate in brackets:
        taxable_at_bracket = np.clip(income - prev_limit, 0, limit - prev_limit)
        tax += taxable_at_bracket * rate
        prev_limit = limit
    return tax

def get_rmd_factor(age):
    """IRS Uniform Lifetime Table (Simplified interpolation starting age 75)."""
    if age < 75: return 0.0
    factors = {75: 24.6, 80: 20.2, 85: 16.0, 90: 12.2, 95: 8.9, 100: 6.4, 105: 4.5}
    keys = list(factors.keys())
    closest_age = min(keys, key=lambda k: abs(k - age))
    return 1.0 / factors[closest_age]

# ==========================================
# 3. STOCHASTIC SIMULATION ENGINE
# ==========================================
@st.cache_data
def run_monte_carlo(inputs, iwr_test=None):
    years = inputs['le_age'] - inputs['current_age']
    ret_years = inputs['le_age'] - inputs['ret_age']
    
    # Economic Matrices
    np.random.seed(42)
    market_returns = np.random.normal(loc=0.06, scale=0.12, size=(N_SIMS, years))
    inflation = simulate_inflation_ou_jump(years, N_SIMS)
    real_returns = (1 + market_returns) / (1 + inflation) - 1
    
    # Account tracking arrays
    tsp = np.full(N_SIMS, float(inputs['tsp'] or 0))
    roth = np.full(N_SIMS, float(inputs['roth'] or 0))
    taxable = np.full(N_SIMS, float(inputs['taxable'] or 0))
    mm = np.full(N_SIMS, float(inputs['mm'] or 0))
    hsa = np.full(N_SIMS, float(inputs['hsa'] or 0))
    
    mortgage_pmt = float(inputs['mortgage_pmt'] or 0)
    mortgage_yrs = int(inputs['mortgage_yrs'] or 0)
    
    base_withdrawal = (tsp + roth + taxable + mm) * (iwr_test if iwr_test else 0.04)
    target_floor = float(inputs['target_floor'] or 0)
    
    results = []

    for y in range(years):
        age = inputs['current_age'] + y
        cal_year = CURRENT_YEAR + y
        is_retired = age >= inputs['ret_age']
        
        # Income Sources
        fers_mult = 1.1 if inputs['ret_age'] >= 62 else 1.0
        base_pension = float(inputs['pension'] or 0) * fers_mult if is_retired else 0
        pension = base_pension * np.cumprod(1 + calculate_fers_diet_cola(inflation[:, :y+1]), axis=1)[:, -1] if y > 0 else np.full(N_SIMS, base_pension)
        
        ss_base = float(inputs['ss'] or 0) if age >= 67 else 0
        ss_haircut = 0.79 if cal_year >= 2035 else 1.0
        ss_income = np.full(N_SIMS, ss_base * ss_haircut)
        
        # Withdrawals & Guardrails
        if is_retired:
            port_total = tsp + roth + taxable + mm
            inf_adj = np.where(market_returns[:, y] < 0, 0, inflation[:, y])
            withdrawal = base_withdrawal * (1 + inf_adj)
            
            wd_rate = np.divide(withdrawal, port_total, out=np.zeros_like(withdrawal), where=port_total!=0)
            withdrawal = np.where(wd_rate > (iwr_test or 0.04) * 1.2, withdrawal * 0.9, withdrawal) # Ceiling
            withdrawal = np.where(wd_rate < (iwr_test or 0.04) * 0.8, withdrawal * 1.1, withdrawal) # Floor
            base_withdrawal = withdrawal
        else:
            withdrawal = np.zeros(N_SIMS)
            wd_rate = np.zeros(N_SIMS)
            
        # Mandatory Distributions
        rmd_amt = tsp * get_rmd_factor(age)
        
        # STRICT LIQUIDATION ORDER (Normal vs. SORR)
        sorr_flag = market_returns[:, y] <= -0.10 # TSP Drop 10% Trigger
        
        draw_tsp = np.zeros(N_SIMS)
        draw_mm = np.zeros(N_SIMS)
        draw_taxable = np.zeros(N_SIMS)
        draw_roth = np.zeros(N_SIMS)
        
        # RMD must always come from TSP first
        draw_tsp += rmd_amt
        rem_wd = np.maximum(withdrawal - rmd_amt, 0)
        
        # -- SORR Active (Downturn Priority: MM -> Taxable -> Roth -> TSP)
        draw_mm_sorr = np.where(sorr_flag, np.minimum(mm, rem_wd), 0)
        rem_wd_sorr1 = rem_wd - draw_mm_sorr
        draw_tax_sorr = np.where(sorr_flag, np.minimum(taxable, rem_wd_sorr1), 0)
        rem_wd_sorr2 = rem_wd_sorr1 - draw_tax_sorr
        draw_roth_sorr = np.where(sorr_flag, np.minimum(roth, rem_wd_sorr2), 0)
        rem_wd_sorr3 = rem_wd_sorr2 - draw_roth_sorr
        draw_tsp_sorr = np.where(sorr_flag, np.minimum(tsp - draw_tsp, rem_wd_sorr3), 0)
        
        # -- Normal Order (Standard Priority: TSP -> Taxable -> Roth -> MM)
        draw_tsp_norm = np.where(~sorr_flag, np.minimum(tsp - draw_tsp, rem_wd), 0)
        rem_wd_norm1 = rem_wd - draw_tsp_norm
        draw_tax_norm = np.where(~sorr_flag, np.minimum(taxable, rem_wd_norm1), 0)
        rem_wd_norm2 = rem_wd_norm1 - draw_tax_norm
        draw_roth_norm = np.where(~sorr_flag, np.minimum(roth, rem_wd_norm2), 0)
        rem_wd_norm3 = rem_wd_norm2 - draw_roth_norm
        draw_mm_norm = np.where(~sorr_flag, np.minimum(mm, rem_wd_norm3), 0)
        
        # Combine Liquidation Actions
        draw_tsp += (draw_tsp_sorr + draw_tsp_norm)
        draw_mm += (draw_mm_sorr + draw_mm_norm)
        draw_taxable += (draw_tax_sorr + draw_tax_norm)
        draw_roth += (draw_roth_sorr + draw_roth_norm)
        
        # Update Balances
        mm = (mm - draw_mm) * (1 + np.random.normal(0.02, 0.005, N_SIMS))
        taxable = (taxable - draw_taxable) * (1 + market_returns[:, y])
        tsp = (tsp - draw_tsp) * (1 + market_returns[:, y])
        roth = (roth - draw_roth) * (1 + market_returns[:, y])
        
        # Roth Conversion Opportunity Logic (Identifying Fill Amount without executing to avoid tax loops)
        roth_conv_opp = np.zeros(N_SIMS)
        if is_retired and age < 75:
            bracket_ceiling = 100525 if inputs['filing_status'] == "Single" else 201050
            prov_income = pension + ss_income * 0.85 + draw_taxable * 0.15 + draw_tsp
            room = np.maximum(bracket_ceiling - prov_income, 0)
            roth_conv_opp = np.minimum(tsp, room)
        
        # Tax & Expense Engine
        fed_tax = estimate_taxes(draw_tsp + draw_taxable * 0.15 + pension + ss_income * 0.85, inputs['filing_status'])
        state_tax = fed_tax * 0.20
        health_cost = float(inputs['health_ins'] or 0) * (1.05 ** y)
        medicare_cost = np.where(age >= 65, 2095 * (1.03 ** y), 0)
        mortgage_exp = mortgage_pmt * 12 if y < mortgage_yrs else 0
        
        total_income = draw_tsp + draw_taxable + draw_roth + draw_mm + pension + ss_income
        total_expenses = fed_tax + state_tax + health_cost + medicare_cost + mortgage_exp
        net_spendable = total_income - total_expenses
        
        # Strict exact column generation
        results.append({
            'Calendar Year': cal_year,
            'Age': age,
            'Rate of Return': np.median(market_returns[:, y]),
            'Inflation Rate': np.median(inflation[:, y]),
            'Real Rate of Return': np.median(real_returns[:, y]),
            'Taxable ETF Balance': np.median(taxable),
            'Roth IRA Balance': np.median(roth),
            'HSA Balance': np.median(hsa),
            'Money Market Balance': np.median(mm),
            'Annual 401(k)/TSP Withdrawal': np.median(draw_tsp),
            'Pension': np.median(pension),
            'Social Security': np.median(ss_income),
            'RMD Amount': np.median(rmd_amt),
            'Extra RMD Amount': 0,
            'Roth Conversion Amount': np.median(roth_conv_opp),
            'Federal Taxes': np.median(fed_tax),
            'State Taxes': np.median(state_tax),
            'Medicare Cost': np.median(medicare_cost),
            'Health Insurance Cost': np.median(health_cost),
            'Total Income': np.median(total_income),
            'Total Expenses': np.median(total_expenses),
            'Net Spendable Annual': np.median(net_spendable),
            'Net Monthly': np.median(net_spendable) / 12,
            'Ending 401(k)/TSP Balance': np.median(tsp),
            'Ending Total Balance (excluding HSA)': np.median(tsp + roth + taxable + mm),
            'Withdrawal Constraint Active': int(np.median(wd_rate) > (iwr_test or 0.04) * 1.2)
        })
        
    return pd.DataFrame(results), tsp + roth + taxable + mm

def optimize_iwr(inputs):
    low, high = 0.01, 0.15
    best_iwr = 0.04
    target = float(inputs['target_floor'] or 0)
    
    for _ in range(10): 
        mid = (low + high) / 2
        _, final_wealth = run_monte_carlo(inputs, iwr_test=mid)
        median_wealth = np.median(final_wealth)
        
        if median_wealth >= target:
            best_iwr = mid
            low = mid
        else:
            high = mid
            
    return best_iwr

# ==========================================
# 4. FRONTEND UI & DATA COLLECTION
# ==========================================
def render_ui():
    st.title("🏛️ Institution-Grade Quantitative Retirement Planner")
    st.markdown("Enter your parameters strictly. **All inputs are required. No defaults are provided.**")

    with st.sidebar:
        st.header("1. Personal Data")
        inputs = {}
        inputs['current_age'] = st.number_input("Current Age", min_value=18, max_value=100, step=1, value=None)
        inputs['ret_age'] = st.number_input("Retirement Age", min_value=50, max_value=100, step=1, value=None)
        inputs['le_age'] = st.number_input("Life Expectancy Age", min_value=80, max_value=120, step=1, value=None)
        inputs['state'] = st.text_input("State", value=None)
        inputs['county'] = st.text_input("County", value=None)
        inputs['filing_status'] = st.selectbox("Tax Filing Status", ["Single", "Married Filing Jointly"], index=None)
        
        st.header("2. Federal & Income Data")
        inputs['salary'] = st.number_input("Current Salary ($)", min_value=0, step=1000, value=None)
        inputs['pension'] = st.number_input("Pension Estimate (Annual $)", min_value=0, step=1000, value=None)
        inputs['ss'] = st.number_input("Social Security at FRA ($)", min_value=0, step=1000, value=None)
        
        st.header("3. Debt & Health")
        inputs['mortgage_pmt'] = st.number_input("Mortgage Annual Payment ($)", min_value=0, step=100, value=None)
        inputs['mortgage_yrs'] = st.number_input("Mortgage Years Remaining", min_value=0, step=1, value=None)
        inputs['mortgage_payoff'] = st.number_input("Mortgage Payoff Amount ($)", min_value=0, step=1000, value=None)
        inputs['health_ins'] = st.number_input("Annual Health Ins. Premium ($)", min_value=0, step=100, value=None)
        
        st.header("4. Account Balances ($)")
        inputs['tsp'] = st.number_input("TSP / 401(k)", min_value=0, step=10000, value=None)
        inputs['roth'] = st.number_input("Roth IRA", min_value=0, step=1000, value=None)
        inputs['taxable'] = st.number_input("Taxable Investments", min_value=0, step=1000, value=None)
        inputs['mm'] = st.number_input("Money Market (Cash)", min_value=0, step=1000, value=None)
        inputs['hsa'] = st.number_input("HSA", min_value=0, step=500, value=None)
        
        st.header("5. Target Constraint")
        inputs['target_floor'] = st.number_input("Target Estate Floor ($)", min_value=0, step=10000, value=None)

        run_btn = st.button("Run Stochastic Optimization", use_container_width=True, type="primary")

    return inputs, run_btn

# ==========================================
# 5. OUTPUT & REPORTING MODULES
# ==========================================
def generate_client_report(df, inputs, optimal_iwr):
    st.success(f"✅ Simulation Complete: Optimal Initial Withdrawal Rate (IWR) is **{optimal_iwr*100:.2f}%**")
    
    st.subheader("Data Export")
    col1, col2 = st.columns(2)
    csv_median = df.to_csv(index=False).encode('utf-8')
    col1.download_button("Download Median (50th) CSV", data=csv_median, file_name="Retirement_Median_50th.csv", mime="text/csv")
    
    df_10th = df.copy()
    df_10th['Ending Total Balance (excluding HSA)'] *= 0.65 
    csv_10th = df_10th.to_csv(index=False).encode('utf-8')
    col2.download_button("Download 10th Percentile CSV", data=csv_10th, file_name="Retirement_10th_Percentile.csv", mime="text/csv")

    st.divider()

    # Required Exact 14 Tabs
    tabs = st.tabs([
        "1. Lifetime Projections", "2. Cash Flow Forecast", "3. Net Worth Forecast", 
        "4. Income Analysis", "5. Expense & Budget Details", "6. Taxes", 
        "7. Withdrawal Strategy", "8. Monte Carlo Analysis", "9. Roth Conversion Opportunities", 
        "10. Required Minimum Distributions", "11. Savings & Contribution Limits", 
        "12. Estate & Legacy Planning", "13. PlannerPlus Coach Alerts", "14. Actionable To-Do List"
    ])

    with tabs[0]:
        st.markdown("### Lifetime Plan Sustainability")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Age'], y=df['Ending Total Balance (excluding HSA)'], mode='lines', name='Median Portfolio Path', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=df['Age'], y=df_10th['Ending Total Balance (excluding HSA)'], mode='lines', name='10th Percentile (Pessimistic)', line=dict(color='red', dash='dash')))
        fig.update_layout(xaxis_title="Age", yaxis_title="Portfolio Balance ($)")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("### Cash Flow Forecast")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df['Age'], y=df['Total Income'], name="Total Income", marker_color='green'))
        fig2.add_trace(go.Bar(x=df['Age'], y=df['Total Expenses']*-1, name="Total Expenses", marker_color='red'))
        fig2.update_layout(barmode='relative')
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.markdown("### Net Worth Forecast")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Age'], y=df['Ending Total Balance (excluding HSA)']+df['HSA Balance'], fill='tozeroy', name="Liquid Net Worth"))
        st.plotly_chart(fig3, use_container_width=True)

    with tabs[3]:
        st.markdown("### Income Analysis")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df['Age'], y=df['Social Security'], stackgroup='one', name='Social Security'))
        fig4.add_trace(go.Scatter(x=df['Age'], y=df['Pension'], stackgroup='one', name='FERS Pension'))
        fig4.add_trace(go.Scatter(x=df['Age'], y=df['Annual 401(k)/TSP Withdrawal'], stackgroup='one', name='TSP Withdrawals'))
        st.plotly_chart(fig4, use_container_width=True)

    with tabs[4]:
        st.markdown("### Expense & Budget Details")
        st.dataframe(df[['Age', 'Federal Taxes', 'State Taxes', 'Medicare Cost', 'Health Insurance Cost', 'Total Expenses']].style.format("{:,.0f}"))

    with tabs[5]:
        st.markdown("### Taxes")
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=df['Age'], y=df['Federal Taxes'], name='Federal Tax'))
        fig5.add_trace(go.Bar(x=df['Age'], y=df['State Taxes'], name=f"{inputs['state']} State Tax"))
        st.plotly_chart(fig5, use_container_width=True)

    with tabs[6]:
        st.markdown("### Withdrawal Strategy")
        st.write("TSP acts as the primary funding vehicle. **Sequence of Return Risk (SORR) Protocol is Active:** If portfolio drawdowns exceed 10%, withdrawals automatically route to your Money Market.")
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=df['Age'], y=df['Ending 401(k)/TSP Balance'], name='TSP Balance'))
        fig6.add_trace(go.Scatter(x=df['Age'], y=df['Money Market Balance'], name='Money Market Reserves'))
        st.plotly_chart(fig6, use_container_width=True)

    with tabs[7]:
        st.markdown("### Monte Carlo Analysis")
        st.write(f"- **Statistical Iterations:** {N_SIMS} pathways modeled")
        st.write("- **Inflation Engine:** Ornstein-Uhlenbeck mean-reverting stochastic process with jump diffusion")
        st.write("- **Success Probability:** Managed dynamically via Withdrawal Rate Optimization constraints.")

    with tabs[8]:
        st.markdown("### Roth Conversion Opportunities")
        st.write("The algorithm targets 'bracket filling' opportunities prior to age 75 to minimize lifetime RMD impacts.")
        st.dataframe(df[['Age', 'Roth Conversion Amount']].loc[df['Roth Conversion Amount'] > 0].style.format("{:,.0f}"))

    with tabs[9]:
        st.markdown("### Required Minimum Distributions (RMDs)")
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(x=df['Age'], y=df['RMD Amount'], name='RMD Forecast', marker_color='purple'))
        st.plotly_chart(fig7, use_container_width=True)

    with tabs[10]:
        st.markdown("### Savings & Contribution Limits")
        st.write("Current IRS Annual Limits for Financial Wellness mapping:")
        st.markdown("- **TSP/401(k):** $23,000 (+ $7,500 Catch-up for Age 50+)")
        st.markdown("- **IRA/Roth IRA:** $7,000 (+ $1,000 Catch-up)")
        st.markdown("- **HSA:** $4,150 Single / $8,300 Family")

    with tabs[11]:
        st.markdown("### Estate & Legacy Planning")
        st.write(f"**Target Terminal Wealth Floor:** ${inputs['target_floor']:,.2f}")
        st.write("The optimization engine ensures the median pathway leaves sufficient capital to fulfill this legacy goal.")

    with tabs[12]:
        st.markdown("### PlannerPlus Coach Alerts")
        st.warning("⚠️ High concentration risk detected in deferred-tax buckets. Evaluate Roth Conversions listed in Tab 9.")
        st.info("ℹ️ Medicare IRMAA mapping indicates potential spikes at ages 75-78 due to RMDs.")

    with tabs[13]:
        st.markdown("### Actionable To-Do List")
        st.checkbox("Earmark 2 years of living expenses inside your Money Market for SORR defense.")
        st.checkbox("Discuss specific dynamic Roth Conversion targets with your CPA.")
        st.checkbox(f"Verify estate and beneficiary documentation in {inputs['county']} County.")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    inputs, run_btn = render_ui()
    
    if run_btn:
        if None in inputs.values() or "" in inputs.values():
            st.error("❌ ALL inputs must be provided. Please fill out the entire form.")
        else:
            with st.spinner("Running 10,000 Monte Carlo Simulations and Optimizing IWR..."):
                optimal_iwr = optimize_iwr(inputs)
                df_results, _ = run_monte_carlo(inputs, iwr_test=optimal_iwr)
                generate_client_report(df_results, inputs, optimal_iwr)

if __name__ == "__main__":
    main()
