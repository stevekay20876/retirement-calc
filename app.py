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

FED_BRACKETS = [
    (0.10, 0, 23200), (0.12, 23200, 94300), (0.22, 94300, 201050),
    (0.24, 201050, 383900), (0.32, 383900, 487450), (0.35, 487450, 731200), (0.37, 731200, float('inf'))
]
STD_DED = {"Single": 14600, "Married": 29200}
IRMAA_CLIFFS = [206000, 258000, 322000, 386000, 750000]
IRMAA_PREMIUMS = [174.70, 244.60, 349.40, 454.20, 559.00, 593.90]

def get_rmd_divisor(age):
    if age < 75: return float('inf')
    ult = {75: 24.6, 80: 20.2, 85: 16.0, 90: 12.2, 95: 8.9, 100: 6.4, 105: 4.6}
    return ult.get(age, max(1.9, 24.6 - (age-75)*0.85))

# ==========================================
# UI: STRUCTURED DATA COLLECTION
# ==========================================
st.title("Institution-Grade Stochastic Retirement Optimization")

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

if int(le_age) <= int(curr_age):
    st.error("⚠️ Life Expectancy Age must be greater than Current Age.")
    st.stop()

curr_age, ret_age, le_age = int(curr_age), int(ret_age), int(le_age)

# ==========================================
# STOCHASTIC ECONOMIC GENERATOR
# ==========================================
def generate_economic_paths(n_paths=10000, n_years=50, seed=42):
    np.random.seed(seed)
    corr_matrix = np.array([[1.0, -0.15], [-0.15, 1.0]])
    L = np.linalg.cholesky(corr_matrix)
    df = 5
    shocks = stats.t.rvs(df, size=(2, n_years, n_paths)) / np.sqrt(df / (df - 2)) 
    corr_shocks = np.einsum('ij,jkl->ikl', L, shocks)
    Z_market, Z_infl = corr_shocks[0], corr_shocks[1]
    
    geom_mean = arith_mean - (volatility**2)/2.0
    market_returns = np.exp(geom_mean*1.0 + volatility*1.0*Z_market) - 1.0
    
    inf_mu, inf_theta, inf_sigma, jump_prob = 0.025, 0.3, 0.015, 0.10
    inflation_paths = np.zeros((n_years, n_paths))
    inf_current = np.full(n_paths, inf_mu)
    
    for t in range(n_years):
        jumps = np.where(np.random.rand(n_paths) < jump_prob, np.random.normal(0.04, 0.01, n_paths), 0)
        market_returns[t] = np.where(jumps > 0, market_returns[t] - (jumps * 0.5), market_returns[t])
        dI = inf_theta * (inf_mu - inf_current) + inf_sigma * Z_infl[t] + jumps
        inf_current = np.clip(inf_current + dI, -0.02, 0.15)
        inflation_paths[t] = inf_current
        
    return market_returns, inflation_paths

def calculate_taxes(income_arr, filing_status):
    deduction = STD_DED.get(filing_status, 14600)
    taxable = np.maximum(income_arr - deduction, 0)
    tax = np.zeros_like(taxable)
    for rate, lower, upper in FED_BRACKETS:
        tax += np.maximum(np.minimum(taxable, upper) - lower, 0) * rate
    return tax

# ==========================================
# CORE MONTE CARLO & WITHDRAWAL ENGINE
# ==========================================
def run_simulation(IWR, market_paths, inflation_paths):
    n_years, n_paths = market_paths.shape
    v_tsp, v_roth, v_tax, v_mm, v_hsa = (np.full(n_paths, float(bal)) for bal in (tsp_bal, roth_bal, tax_bal, mm_bal, hsa_bal))
    w_scheduled = np.full(n_paths, IWR * (tsp_bal + roth_bal + tax_bal + mm_bal))
    
    current_health = np.full(n_paths, float(health_ins))
    current_pension, current_ss = np.zeros(n_paths), np.zeros(n_paths)
    
    results = []
    prev_v_tsp = v_tsp.copy()
    prev_total_port = v_tsp + v_roth + v_tax + v_mm

    for t in range(n_years):
        age, year = curr_age + t, datetime.now().year + t
        ret_t, inf_t = market_paths[t], inflation_paths[t]
        
        # Grow Balances
        v_tsp *= (1 + ret_t); v_roth *= (1 + ret_t); v_tax *= (1 + ret_t); v_hsa *= (1 + ret_t)
        v_mm *= (1 + ret_t * 0.02)
        total_port = v_tsp + v_roth + v_tax + v_mm
        
        # Inflows & Outflows
        cola = np.where(inf_t <= 0.02, inf_t, np.where(inf_t <= 0.03, 0.02, inf_t - 0.01))
        if age == ret_age: current_pension = np.full(n_paths, float(salary * yos * (1.1 if ret_age >= 62 else 1.0) / 100))
        elif age > ret_age: current_pension *= (1 + cola)
            
        if age == 67: current_ss = np.full(n_paths, float(ss_fra * 12))
        elif age > 67: current_ss *= (1 + inf_t)
            
        if t > 0: current_health *= (1 + inf_t + 0.02)

        ss_arr = current_ss * (0.79 if year >= 2035 else 1.0)
        mort = float(mort_pmt) if t < mort_yrs else 0.0
        
        # CASAM Guardrails
        if t > 0:
            w_scheduled *= (1 + np.where(market_paths[t-1] < 0, 0, inf_t))
            cwr = w_scheduled / np.maximum(total_port, 1)
            w_scheduled = np.where(cwr > (IWR * 1.20), w_scheduled * 0.90, w_scheduled)
            w_scheduled = np.where(cwr < (IWR * 0.80), w_scheduled * 1.10, w_scheduled)
            w_scheduled = np.where(total_port <= (prev_total_port * 0.90), w_scheduled * 0.90, w_scheduled)
        
        prev_total_port = total_port.copy()
        
        fixed_inflows = current_pension + ss_arr
        fixed_expenses = mort + current_health
        spending_need = np.maximum(w_scheduled + fixed_expenses - fixed_inflows, 0)
        
        rmd_req = v_tsp / get_rmd_divisor(age)
        
        # Liquidation Engine
        tsp_w, mm_w, tax_w, roth_w = np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths), np.zeros(n_paths)
        downturn_mode = (v_tsp <= (prev_v_tsp * 0.90)) if t > 0 else np.zeros(n_paths, dtype=bool)
            
        rmd_applied = np.minimum(rmd_req, spending_need)
        v_tsp -= rmd_req
        v_tax += np.maximum(rmd_req - spending_need, 0) 
        rem_need = spending_need - rmd_applied
        
        # Withdrawals
        normal_mask = ~downturn_mode
        pull_tsp = np.minimum(rem_need, v_tsp) * normal_mask
        tsp_w += pull_tsp + rmd_req; v_tsp -= pull_tsp; rem_need -= pull_tsp
        
        pull_mm = np.minimum(rem_need, v_mm) * downturn_mode
        v_mm -= pull_mm; mm_w += pull_mm; rem_need -= pull_mm
        
        pull_tax = np.minimum(rem_need, v_tax) * downturn_mode
        v_tax -= pull_tax; tax_w += pull_tax; rem_need -= pull_tax
        
        pull_roth = np.minimum(rem_need, v_roth) * downturn_mode
        v_roth -= pull_roth; roth_w += pull_roth; rem_need -= pull_roth
        
        pull_tsp_fb = np.minimum(rem_need, v_tsp) * downturn_mode
        v_tsp -= pull_tsp_fb; tsp_w += pull_tsp_fb
        
        prev_v_tsp = v_tsp.copy() + pull_tsp + pull_tsp_fb 
        
        # Taxes & Expenses
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
        
        results.append({
            'Year': year, 'Age': age, 'Total_Port': total_port.copy(),
            'TSP': v_tsp.copy(), 'Roth': v_roth.copy(), 'Taxable': v_tax.copy(),
            'MM': v_mm.copy(), 'HSA': v_hsa.copy(), 'Net_Spendable': (w_scheduled - tot_exp).copy(),
            'TSP_W': tsp_w.copy(), 'MM_W': mm_w.copy(), 'Tax_W': tax_w.copy(), 'Roth_W': roth_w.copy(),
            'Pension': current_pension.copy(), 'SS': ss_arr.copy(),
            'Fed_Tax': fed_tax.copy(), 'State_Tax': state_tax.copy(), 'IRMAA': irmaa_cost.copy(),
            'Fixed_Exp': fixed_expenses.copy(), 'Total_Exp': tot_exp.copy(), 'RMD': rmd_req.copy(),
            'Market_Ret': ret_t.copy(), 'Inflation': inf_t.copy()
        })
    return results

def optimize_iwr():
    m_paths, i_paths = generate_economic_paths(n_paths=10000, n_years=le_age - curr_age)
    
    def objective(iwr_guess):
        res = run_simulation(iwr_guess, m_paths, i_paths)
        return np.median(res[-1]['Total_Port']) - estate_floor
    
    low_b, high_b = 0.001, 0.25 
    if objective(low_b) < 0: return low_b, m_paths, i_paths
    if objective(high_b) > 0: return high_b, m_paths, i_paths
    return brentq(objective, low_b, high_b, xtol=1e-4, maxiter=30), m_paths, i_paths

# ==========================================
# 12. CLIENT REPORT STRUCTURE (BOLDIN FORMAT)
# ==========================================
if st.sidebar.button("Run Advanced Simulation", type="primary"):
    with st.spinner("Executing 10,000 Monte Carlo Iterations & Stochastic Optimization..."):
        
        max_iwr, m_paths, i_paths = optimize_iwr()
        final_results = run_simulation(max_iwr, m_paths, i_paths)
        
        # Aggregate Median Data for the Report
        report_data = []
        for r in final_results:
            report_data.append({k: np.median(v) if isinstance(v, np.ndarray) else v for k, v in r.items()})
        df = pd.DataFrame(report_data)
        
        # Calculate Percentiles & Probabilities
        final_wealths = final_results[-1]['Total_Port']
        prob_success = np.mean(final_wealths >= estate_floor) * 100
        wealth_10th = [np.percentile(r['Total_Port'], 10) for r in final_results]
        wealth_90th = [np.percentile(r['Total_Port'], 90) for r in final_results]

        # ---------------------------------------------------------
        # REPORT DASHBOARD UI
        # ---------------------------------------------------------
        st.success(f"Simulation Complete. Optimal Initial Withdrawal Rate: **{max_iwr*100:.2f}%**")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Lifetime Projections", 
            "💵 Cash Flow & Income", 
            "📈 Net Worth Forecast", 
            "🏛️ Taxes & Withdrawals", 
            "💡 Coach Alerts & To-Do"
        ])

        # TAB 1: LIFETIME PROJECTIONS & MONTE CARLO
        with tab1:
            st.header("Lifetime Projections & Monte Carlo Analysis")
            st.markdown("A statistical overview of your plan's sustainability through varying market conditions.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Monte Carlo Probability of Success", f"{prob_success:.1f}%", help="Probability of ending above Estate Floor")
            c2.metric("Median Terminal Wealth", f"${df['Total_Port'].iloc[-1]:,.0f}")
            c3.metric("Pessimistic (10th Pct) Wealth", f"${wealth_10th[-1]:,.0f}")

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=df['Age'], y=wealth_90th, mode='lines', name='Optimistic (90th)', line=dict(color='lightgreen', dash='dot')))
            fig_mc.add_trace(go.Scatter(x=df['Age'], y=df['Total_Port'], mode='lines', name='Median (50th)', line=dict(color='blue', width=3)))
            fig_mc.add_trace(go.Scatter(x=df['Age'], y=wealth_10th, mode='lines', name='Pessimistic (10th)', line=dict(color='red', dash='dash')))
            fig_mc.add_hline(y=estate_floor, line_dash="solid", line_color="black", annotation_text="Target Estate Floor")
            fig_mc.update_layout(title="Stochastic Wealth Trajectory", xaxis_title="Age", yaxis_title="Total Assets ($)", hovermode="x unified")
            st.plotly_chart(fig_mc, use_container_width=True)

        # TAB 2: CASH FLOW & INCOME ANALYSIS
        with tab2:
            st.header("Cash Flow Forecast & Income Analysis")
            st.markdown("Breakdown of all revenue streams versus projected living expenses.")
            
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(x=df['Age'], y=df['SS'], name='Social Security', marker_color='#1f77b4'))
            fig_cf.add_trace(go.Bar(x=df['Age'], y=df['Pension'], name='Pension', marker_color='#ff7f0e'))
            total_withdrawals = df['TSP_W'] + df['MM_W'] + df['Tax_W'] + df['Roth_W']
            fig_cf.add_trace(go.Bar(x=df['Age'], y=total_withdrawals, name='Portfolio Withdrawals', marker_color='#2ca02c'))
            fig_cf.add_trace(go.Scatter(x=df['Age'], y=df['Total_Exp'] + df['Net_Spendable'], mode='lines', name='Total Spending Need (Incl. Taxes)', line=dict(color='black', width=2)))
            
            fig_cf.update_layout(barmode='stack', title="Income Sources vs. Total Expenses", xaxis_title="Age", yaxis_title="Annual Cash Flow ($)")
            st.plotly_chart(fig_cf, use_container_width=True)
            
            st.subheader("Expense & Budget Details")
            fig_exp = go.Figure()
            fig_exp.add_trace(go.Bar(x=df['Age'], y=df['Fixed_Exp'], name='Fixed Costs (Mortgage/Health)', marker_color='gray'))
            fig_exp.add_trace(go.Bar(x=df['Age'], y=df['Fed_Tax'] + df['State_Tax'], name='Taxes', marker_color='red'))
            fig_exp.add_trace(go.Bar(x=df['Age'], y=df['IRMAA'], name='Medicare IRMAA', marker_color='orange'))
            fig_exp.update_layout(barmode='stack', title="Itemized Core Expenses", xaxis_title="Age", yaxis_title="Expenses ($)")
            st.plotly_chart(fig_exp, use_container_width=True)

        # TAB 3: NET WORTH FORECAST
        with tab3:
            st.header("Net Worth Forecast")
            st.markdown("Projections of total portfolio allocation over time.")
            
            fig_nw = go.Figure()
            fig_nw.add_trace(go.Scatter(x=df['Age'], y=df['TSP'], mode='lines', stackgroup='one', name='Tax-Deferred (TSP/401k)'))
            fig_nw.add_trace(go.Scatter(x=df['Age'], y=df['Roth'], mode='lines', stackgroup='one', name='Tax-Free (Roth)'))
            fig_nw.add_trace(go.Scatter(x=df['Age'], y=df['Taxable'], mode='lines', stackgroup='one', name='Taxable Investments'))
            fig_nw.add_trace(go.Scatter(x=df['Age'], y=df['MM'], mode='lines', stackgroup='one', name='Money Market Cash'))
            fig_nw.add_trace(go.Scatter(x=df['Age'], y=df['HSA'], mode='lines', stackgroup='one', name='HSA Balance'))
            
            fig_nw.update_layout(title="Asset Liquidity Timeline", xaxis_title="Age", yaxis_title="Balance ($)", hovermode="x unified")
            st.plotly_chart(fig_nw, use_container_width=True)

        # TAB 4: TAXES, RMDS, & WITHDRAWAL STRATEGY
        with tab4:
            st.header("Withdrawal Strategy & Taxes")
            
            st.subheader("Account Liquidation Order")
            fig_wd = go.Figure()
            fig_wd.add_trace(go.Bar(x=df['Age'], y=df['TSP_W'], name='TSP / 401(k) Draws', marker_color='#9467bd'))
            fig_wd.add_trace(go.Bar(x=df['Age'], y=df['Tax_W'], name='Taxable Inv Draws', marker_color='#8c564b'))
            fig_wd.add_trace(go.Bar(x=df['Age'], y=df['MM_W'], name='Money Market Draws', marker_color='#e377c2'))
            fig_wd.add_trace(go.Bar(x=df['Age'], y=df['Roth_W'], name='Roth IRA Draws', marker_color='#17becf'))
            fig_wd.update_layout(barmode='stack', title="Dynamic Account Withdrawals (SORR Governed)", xaxis_title="Age", yaxis_title="Amount ($)")
            st.plotly_chart(fig_wd, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Federal & State Taxes")
                fig_tx = go.Figure(go.Scatter(x=df['Age'], y=df['Fed_Tax']+df['State_Tax'], fill='tozeroy', marker_color='red'))
                st.plotly_chart(fig_tx, use_container_width=True)
            with c2:
                st.subheader("Required Min. Distributions (RMDs)")
                fig_rmd = go.Figure(go.Scatter(x=df['Age'], y=df['RMD'], fill='tozeroy', marker_color='purple'))
                st.plotly_chart(fig_rmd, use_container_width=True)

        # TAB 5: COACH ALERTS & ACTIONABLE TO-DO
        with tab5:
            st.header("PlannerPlus Coach Alerts")
            st.markdown("Automated monitoring of specific risks, oversights, and opportunities.")
            
            # Dynamic Alerts Logic
            peak_tax_age = df.loc[df['Fed_Tax'].idxmax(), 'Age']
            max_irmaa = df['IRMAA'].max()
            mm_depleted_age = df.loc[df['MM'] <= 0, 'Age'].min() if (df['MM'] <= 0).any() else None

            if peak_tax_age >= 75:
                st.warning(f"⚠️ **RMD Tax Spike Alert:** Your taxes peak at age {peak_tax_age} due to mandatory RMDs. **Opportunity:** Consider systematic Roth Conversions before age 75 to smooth out this tax burden.")
            else:
                st.success("✅ **Tax Efficiency:** Your lifetime taxes appear relatively smooth without major RMD-driven cliffs.")
                
            if max_irmaa > 0:
                st.warning(f"⚠️ **Medicare IRMAA Alert:** Your income triggers Medicare surcharges (IRMAA) in retirement reaching up to ${max_irmaa:,.0f}/yr. Review capital gains harvesting and Roth usage to manage MAGI.")
                
            if mm_depleted_age:
                st.info(f"🛡️ **SORR Buffer Depletion:** Under median downturn conditions, your Money Market safe-buffer is depleted by age {mm_depleted_age}. Ensure you maintain enough cash equivalence entering the 'Go-Go' years.")

            st.subheader("Actionable To-Do List")
            st.checkbox(f"Set Year 1 Safe Withdrawal Limit to exactly ${(max_iwr * (tsp_bal + roth_bal + tax_bal + mm_bal)):,.2f}.")
            st.checkbox("Consolidate current 401(k) / TSP accounts for streamlined RMD execution.")
            st.checkbox(f"Review Estate Planning docs to ensure ${estate_floor:,.2f} floor transfers optimally to heirs.")
            st.checkbox("Meet with a CPA to model discrete Roth Conversions in the 'Tax Valley' between Retirement and age 75.")

        # STRICT CSV EXPORTS
        st.divider()
        st.header("📥 Data Exports")
        csv_med = df.to_csv(index=False).encode('utf-8')
        
        # Build 10th percentile dataframe specifically for export mapping
        df_10 = df.copy()
        for idx, r in enumerate(final_results):
            df_10.at[idx, 'Total_Port'] = np.percentile(r['Total_Port'], 10)
            df_10.at[idx, 'Net_Spendable'] = np.percentile(r['Net_Spendable'], 10)
        csv_10 = df_10.to_csv(index=False).encode('utf-8')
        
        c1, c2 = st.columns(2)
        c1.download_button("Download Full CSV (Median Scenario)", csv_med, "Retirement_Simulation_Median.csv", "text/csv")
        c2.download_button("Download Full CSV (10th Percentile)", csv_10, "Retirement_Simulation_10th.csv", "text/csv")
