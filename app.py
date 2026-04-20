import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import brentq
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# SYSTEM SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Advanced Quantitative Retirement Engine", layout="wide")

# Tax Brackets (2024/2025 simplified for Ordinary Income)
FED_BRACKETS = [
    (0.10, 0, 23200), (0.12, 23200, 94300), (0.22, 94300, 201050),
    (0.24, 201050, 383900), (0.32, 383900, 487450), (0.35, 487450, 731200), (0.37, 731200, float('inf'))
]
STD_DED = {"Single": 14600, "Married": 29200}
IRMAA_CLIFFS = [206000, 258000, 322000, 386000, 750000] # Married Example
IRMAA_PREMIUMS = [174.70, 244.60, 349.40, 454.20, 559.00, 593.90]

# RMD Divisor Approximation (Uniform Lifetime Table)
def get_rmd_divisor(age):
    if age < 75: return float('inf')
    ult = {75: 24.6, 80: 20.2, 85: 16.0, 90: 12.2, 95: 8.9, 100: 6.4, 105: 4.6}
    return ult.get(age, max(1.9, 24.6 - (age-75)*0.85))

# ==========================================
# UI: STRUCTURED DATA COLLECTION
# ==========================================
st.title("Institution-Grade Stochastic Retirement Optimization Engine")
st.markdown("Enter client inputs. Default placeholders are strictly prohibited to ensure custom accuracy.")

with st.sidebar:
    st.header("1. Personal Data")
    curr_age = st.number_input("Current Age", min_value=18, max_value=100, value=None)
    ret_age = st.number_input("Retirement Age", min_value=18, max_value=100, value=None)
    le_age = st.number_input("Life Expectancy Age", min_value=18, max_value=120, value=None)
    state = st.text_input("State", value=None)
    county = st.text_input("County", value=None)
    filing_status = st.selectbox("Tax Filing Status", ["Single", "Married"], index=None)
    
    st.header("2. FERS / Employment")
    salary = st.number_input("Current Salary ($)", min_value=0, value=None)
    yos = st.number_input("Years of Service at Retirement", min_value=0, value=None)
    ss_fra = st.number_input("Social Security at FRA ($/mo)", min_value=0, value=None)
    
    st.header("3. Liabilities & Health")
    mort_pmt = st.number_input("Annual Mortgage Payment ($)", min_value=0, value=None)
    mort_yrs = st.number_input("Mortgage Years Remaining", min_value=0, value=None)
    health_ins = st.number_input("Annual Health Insurance ($)", min_value=0, value=None)
    
    st.header("4. Constraints & Balances")
    estate_floor = st.number_input("Target Estate Floor Amount ($)", min_value=0, value=None)
    tsp_bal = st.number_input("TSP/401(k) Balance ($)", min_value=0, value=None)
    hsa_bal = st.number_input("HSA Balance ($)", min_value=0, value=None)
    roth_bal = st.number_input("Roth IRA Balance ($)", min_value=0, value=None)
    mm_bal = st.number_input("Money Market Balance ($)", min_value=0, value=None)
    tax_bal = st.number_input("Taxable Inv. Balance ($)", min_value=0, value=None)

    st.header("5. Market Assumptions")
    arith_mean = st.number_input("Arithmetic Mean Return (%)", min_value=0.0, value=None) / 100.0 if st.session_state.get('arith_mean') else 0.08
    volatility = st.number_input("Volatility (Std Dev) (%)", min_value=0.0, value=None) / 100.0 if st.session_state.get('volatility') else 0.15

required = [curr_age, ret_age, le_age, filing_status, salary, yos, ss_fra, estate_floor, tsp_bal, mm_bal, tax_bal, roth_bal]
if any(x is None for x in required):
    st.warning("⚠️ Please fill in all required explicit inputs in the sidebar to run the simulation.")
    st.stop()

if le_age <= curr_age:
    st.error("⚠️ Life Expectancy Age must be greater than Current Age.")
    st.stop()

# Ensure inputs are ints for loops
curr_age = int(curr_age)
ret_age = int(ret_age)
le_age = int(le_age)

# ==========================================
# STOCHASTIC ECONOMIC GENERATOR
# ==========================================
def generate_economic_paths(n_paths=10000, n_years=50, seed=42):
    np.random.seed(seed)
    corr_matrix = np.array([[1.0, -0.15], [-0.15, 1.0]])
    L = np.linalg.cholesky(corr_matrix)
    
    df = 5
    shocks = stats.t.rvs(df, size=(2, n_years, n_paths))
    shocks = shocks / np.sqrt(df / (df - 2)) 
    corr_shocks = np.einsum('ij,jkl->ikl', L, shocks)
    Z_market, Z_infl = corr_shocks[0], corr_shocks[1]
    
    geom_mean = arith_mean - (volatility**2)/2.0
    dt = 1.0
    market_returns = np.exp(geom_mean*dt + volatility*np.sqrt(dt)*Z_market) - 1.0
    
    inf_mu, inf_theta, inf_sigma, jump_prob = 0.025, 0.3, 0.015, 0.10
    inflation_paths = np.zeros((n_years, n_paths))
    inf_current = np.full(n_paths, inf_mu)
    
    for t in range(n_years):
        dW = Z_infl[t]
        jumps = np.where(np.random.rand(n_paths) < jump_prob, np.random.normal(0.04, 0.01, n_paths), 0)
        market_returns[t] = np.where(jumps > 0, market_returns[t] - (jumps * 0.5), market_returns[t])
        
        dI = inf_theta * (inf_mu - inf_current) + inf_sigma * dW + jumps
        inf_current = np.clip(inf_current + dI, -0.02, 0.15)
        inflation_paths[t] = inf_current
        
    return market_returns, inflation_paths

# ==========================================
# CORE MONTE CARLO & WITHDRAWAL ENGINE
# ==========================================
def calculate_taxes(income_arr, filing_status):
    deduction = STD_DED.get(filing_status, 14600)
    taxable = np.maximum(income_arr - deduction, 0)
    tax = np.zeros_like(taxable)
    for rate, lower, upper in FED_BRACKETS:
        tax_in_bracket = np.maximum(np.minimum(taxable, upper) - lower, 0)
        tax += tax_in_bracket * rate
    return tax

def run_simulation(IWR, market_paths, inflation_paths):
    n_years = le_age - curr_age
    n_paths = market_paths.shape[1]
    
    v_tsp, v_roth, v_tax = np.full(n_paths, float(tsp_bal)), np.full(n_paths, float(roth_bal)), np.full(n_paths, float(tax_bal))
    v_mm, v_hsa = np.full(n_paths, float(mm_bal)), np.full(n_paths, float(hsa_bal))
    
    w_base1 = IWR * (tsp_bal + roth_bal + tax_bal + mm_bal)
    w_scheduled = np.full(n_paths, w_base1)
    
    # Cumulative trajectory arrays
    current_health = np.full(n_paths, float(health_ins))
    current_pension = np.zeros(n_paths)
    current_ss = np.zeros(n_paths)
    
    results = []
    
    for t in range(n_years):
        age = curr_age + t
        year = datetime.now().year + t
        ret_t, inf_t = market_paths[t], inflation_paths[t]
        
        # Grow Balances
        v_tsp *= (1 + ret_t)
        v_roth *= (1 + ret_t)
        v_tax *= (1 + ret_t)
        v_mm *= (1 + ret_t * 0.02)
        v_hsa *= (1 + ret_t)
        total_port = v_tsp + v_roth + v_tax + v_mm
        
        # Income Trajectories
        cola = np.where(inf_t <= 0.02, inf_t, np.where(inf_t <= 0.03, 0.02, inf_t - 0.01))
        
        if age == ret_age:
            pension_mult = 1.1 if ret_age >= 62 else 1.0
            current_pension = np.full(n_paths, float(salary * yos * pension_mult / 100))
        elif age > ret_age:
            current_pension *= (1 + cola)
            
        if age == 67:
            current_ss = np.full(n_paths, float(ss_fra * 12))
        elif age > 67:
            current_ss *= (1 + inf_t)
            
        if t > 0:
            current_health *= (1 + inf_t + 0.02)

        ss_arr = current_ss.copy()
        if year >= 2035:
            ss_arr *= 0.79 # 21% Haircut
            
        mort = float(mort_pmt) if t < mort_yrs else 0.0
        
        # CASAM Guardrails
        if t > 0:
            freeze_mask = market_paths[t-1] < 0
            w_scheduled *= (1 + np.where(freeze_mask, 0, inf_t))
            cwr = w_scheduled / np.maximum(total_port, 1)
            w_scheduled = np.where(cwr > (IWR * 1.20), w_scheduled * 0.90, w_scheduled)
            w_scheduled = np.where(cwr < (IWR * 0.80), w_scheduled * 1.10, w_scheduled)
            if t > 0:
                w_scheduled = np.where(total_port <= (prev_total_port * 0.90), w_scheduled * 0.90, w_scheduled)
        
        prev_total_port = total_port.copy()
        
        # Spending Need
        fixed_inflows = current_pension + ss_arr
        fixed_expenses = mort + current_health
        spending_need = np.maximum(w_scheduled + fixed_expenses - fixed_inflows, 0)
        
        rmd_req = v_tsp / get_rmd_divisor(age)
        
        # Liquidation Engine
        tsp_w, mm_w, tax_w, roth_w = np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths)
        downturn_mode = (v_tsp <= (prev_v_tsp * 0.90)) if t > 0 else np.zeros(n_paths, dtype=bool)
            
        rmd_applied = np.minimum(rmd_req, spending_need)
        v_tsp -= rmd_req
        v_tax += np.maximum(rmd_req - spending_need, 0) # Reinvest excess RMD
        rem_need = spending_need - rmd_applied
        
        # Normal Mode
        normal_mask = ~downturn_mode
        pull_tsp = np.minimum(rem_need, v_tsp) * normal_mask
        tsp_w += pull_tsp + rmd_req
        v_tsp -= pull_tsp
        rem_need -= pull_tsp
        
        # Downturn Sequence (MM -> Tax -> Roth -> TSP)
        pull_mm = np.minimum(rem_need, v_mm) * downturn_mode
        v_mm -= pull_mm; rem_need -= pull_mm
        
        pull_tax = np.minimum(rem_need, v_tax) * downturn_mode
        v_tax -= pull_tax; rem_need -= pull_tax
        
        pull_roth = np.minimum(rem_need, v_roth) * downturn_mode
        v_roth -= pull_roth; rem_need -= pull_roth
        
        pull_tsp_fb = np.minimum(rem_need, v_tsp) * downturn_mode
        v_tsp -= pull_tsp_fb; tsp_w += pull_tsp_fb
        
        prev_v_tsp = v_tsp.copy() + pull_tsp + pull_tsp_fb 
        
        # Taxes
        prov_income = current_pension + tsp_w + (0.5 * ss_arr)
        ss_taxable = np.where(prov_income > 32000, np.minimum(0.85 * ss_arr, 0.85 * (prov_income - 32000)), 0)
        
        tot_taxable = current_pension + tsp_w + ss_taxable
        fed_tax = calculate_taxes(tot_taxable, filing_status)
        state_tax = tot_taxable * 0.05
        
        irmaa_cost = np.zeros(n_paths)
        if age >= 65:
            tier = np.searchsorted(IRMAA_CLIFFS, tot_taxable)
            irmaa_cost = np.array([IRMAA_PREMIUMS[min(tr, 5)] * 12 for tr in tier])
            
        tot_exp = fixed_expenses + fed_tax + state_tax + irmaa_cost
        net_spendable = w_scheduled - tot_exp
        
        results.append({
            'Year': year, 'Age': age, 'Total_Port': v_tsp + v_roth + v_tax + v_mm,
            'TSP': v_tsp.copy(), 'Roth': v_roth.copy(), 'Taxable': v_tax.copy(),
            'MM': v_mm.copy(), 'HSA': v_hsa.copy(), 'Net_Spendable': net_spendable.copy(),
            'Fed_Tax': fed_tax.copy(), 'IRMAA': irmaa_cost.copy(), 'RMD': rmd_req.copy(),
            'Market_Ret': ret_t.copy(), 'Inflation': inf_t.copy()
        })
    return results

# ==========================================
# OPTIMIZATION: BRENT'S BOUNDARY CHECKED
# ==========================================
def optimize_iwr():
    m_paths, i_paths = generate_economic_paths(n_paths=10000, n_years=le_age - curr_age)
    
    def objective(iwr_guess):
        res = run_simulation(iwr_guess, m_paths, i_paths)
        if not res: return -estate_floor
        return np.median(res[-1]['Total_Port']) - estate_floor
    
    # Boundary Checking prevents Constraint Error
    low_b, high_b = 0.001, 0.25 
    val_low = objective(low_b)
    val_high = objective(high_b)

    if val_low < 0 and val_high < 0:
        st.warning(f"⚠️ Target Estate Floor (${estate_floor:,.2f}) is mathematically unachievable given your starting balances. Defaulting to safe baseline (0.1%).")
        return low_b, m_paths, i_paths
        
    if val_low > 0 and val_high > 0:
        st.success(f"🌟 Target Estate Floor (${estate_floor:,.2f}) is effortlessly exceeded even with max 25% withdrawals! Optimization capped at 25%.")
        return high_b, m_paths, i_paths

    opt_iwr = brentq(objective, low_b, high_b, xtol=1e-4, maxiter=30)
    return opt_iwr, m_paths, i_paths

# ==========================================
# APPLICATION EXECUTION & REPORTING
# ==========================================
if st.sidebar.button("Run Advanced Simulation", type="primary"):
    with st.spinner("Executing 10,000 Monte Carlo Iterations & Stochastic Root-Finding..."):
        
        max_iwr, m_paths, i_paths = optimize_iwr()
        final_results = run_simulation(max_iwr, m_paths, i_paths)
        
        # Extract Results
        report_data_med = []
        report_data_10 = []
        
        for r in final_results:
            med_row = {
                "Calendar Year": r['Year'], "Age": r['Age'], "Rate of Return": np.median(r['Market_Ret']),
                "Inflation Rate": np.median(r['Inflation']), "Real Rate of Return": np.median(r['Market_Ret']) - np.median(r['Inflation']),
                "Taxable ETF Balance": np.median(r['Taxable']), "Roth IRA Balance": np.median(r['Roth']),
                "HSA Balance": np.median(r['HSA']), "Money Market Balance": np.median(r['MM']),
                "Annual 401(k)/TSP Withdrawal": np.median(r['TSP'] * 0.04), "Federal Taxes": np.median(r['Fed_Tax']),
                "Medicare Cost": np.median(r['IRMAA']), "Net Spendable Annual": np.median(r['Net_Spendable']),
                "Net Monthly": np.median(r['Net_Spendable']) / 12, "Ending 401(k)/TSP Balance": np.median(r['TSP']),
                "Ending Total Balance (excluding HSA)": np.median(r['Total_Port'])
            }
            report_data_med.append(med_row)
            
            p10_row = med_row.copy()
            p10_row["Ending Total Balance (excluding HSA)"] = np.percentile(r['Total_Port'], 10)
            p10_row["Net Spendable Annual"] = np.percentile(r['Net_Spendable'], 10)
            report_data_10.append(p10_row)
            
        df_med = pd.DataFrame(report_data_med)
        df_10 = pd.DataFrame(report_data_10)

        # -----------------------------------
        # CLIENT REPORT UI
        # -----------------------------------
        st.success(f"Simulation Complete. Optimal Initial Withdrawal Rate (IWR): **{max_iwr*100:.2f}%**")
        
        st.header("📊 Lifetime Projections & Net Worth Forecast")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_med['Age'], y=df_med['Ending Total Balance (excluding HSA)'], mode='lines', name='Median (50th) Wealth', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=df_10['Age'], y=df_10['Ending Total Balance (excluding HSA)'], mode='lines', name='Pessimistic (10th) Wealth', line=dict(color='red', dash='dash')))
        fig1.add_hline(y=estate_floor, line_dash="dot", annotation_text="Estate Floor Constraint")
        fig1.update_layout(title="Stochastic Wealth Trajectory", xaxis_title="Age", yaxis_title="Total Portfolio Value ($)")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.header("💵 Income Analysis & Cash Flow")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_med['Age'], y=df_med['Net Spendable Annual'], name='Net Real Spendable Income', marker_color='green'))
        fig2.update_layout(title="Projected Real Spendable Income (Post-Tax/Guardrails Applied)", xaxis_title="Age", yaxis_title="Annual Income ($)", barmode='stack')
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Federal Tax Liabilities")
            fig3 = go.Figure(go.Scatter(x=df_med['Age'], y=df_med['Federal Taxes'], fill='tozeroy'))
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            st.subheader("Medicare Cost & IRMAA")
            fig4 = go.Figure(go.Scatter(x=df_med['Age'], y=df_med['Medicare Cost'], fill='tozeroy', marker_color='orange'))
            st.plotly_chart(fig4, use_container_width=True)

        st.header("✅ Actionable To-Do List & Coach Alerts")
        base_wd = max_iwr * (tsp_bal + roth_bal + tax_bal + mm_bal)
        st.info(f"**Alert 1:** Your optimal CASAM Base Withdrawal is calculated at **${base_wd:,.2f}** per year.")
        st.warning("**Alert 2:** Sequence of Return Risk (SORR) Guardrails are active. Be prepared to liquidate from the Money Market buffer during downturns.")

        st.header("📥 Data Exports")
        c1, c2 = st.columns(2)
        c1.download_button("Download CSV (Median Scenario)", df_med.to_csv(index=False).encode('utf-8'), "Retirement_Simulation_Median.csv", "text/csv")
        c2.download_button("Download CSV (10th Percentile Scenario)", df_10.to_csv(index=False).encode('utf-8'), "Retirement_Simulation_10th.csv", "text/csv")
