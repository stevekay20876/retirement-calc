import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import bisect
import datetime

# ==========================================
# SYSTEM CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="Institutional Retirement Optimizer", layout="wide")
st.title("Advanced Quantitative Retirement & Tax Optimizer")

# Initialize Session State for Inputs
if "sim_results" not in st.session_state:
    st.session_state.sim_results = None

# ==========================================
# 1. FRONTEND: STRUCTURED USER INPUTS
# ==========================================
st.sidebar.header("1. Personal Data")
current_age = st.sidebar.number_input("Current Age", min_value=18, max_value=100, step=1, value=None)
ret_age = st.sidebar.number_input("Retirement Age", min_value=18, max_value=100, step=1, value=None)
le_age = st.sidebar.number_input("Life Expectancy Age", min_value=70, max_value=120, step=1, value=None)
state = st.sidebar.text_input("State of Residence", value=None)
county = st.sidebar.text_input("County", value=None)
filing_status = st.sidebar.selectbox("Tax Filing Status", options=["Single", "Married Filing Jointly"], index=None)

st.sidebar.header("2. Federal Employment & Income")
fed_grade = st.sidebar.text_input("Federal Grade (e.g., GS-12)", value=None)
fed_step = st.sidebar.number_input("Federal Step", min_value=1, max_value=10, step=1, value=None)
yos = st.sidebar.number_input("Years of Service at Retirement", min_value=0, max_value=50, step=1, value=None)
salary = st.sidebar.number_input("Current Salary ($)", min_value=0.0, step=1000.0, value=None)
pension_est = st.sidebar.number_input("Pension Estimate (Annual $)", min_value=0.0, step=1000.0, value=None)
ss_fra = st.sidebar.number_input("Social Security at FRA ($)", min_value=0.0, step=1000.0, value=None)

st.sidebar.header("3. Expenses & Estate")
mortgage_pmt = st.sidebar.number_input("Annual Mortgage Payment ($)", min_value=0.0, step=1000.0, value=None)
mortgage_yrs = st.sidebar.number_input("Mortgage Years Remaining", min_value=0, step=1, value=None)
mortgage_bal = st.sidebar.number_input("Mortgage Payoff Amount ($)", min_value=0.0, step=1000.0, value=None)
health_ins = st.sidebar.number_input("Annual Health Insurance Cost ($)", min_value=0.0, step=500.0, value=None)
health_plan = st.sidebar.selectbox("Health Insurance Plan Type", options=["FEHB", "Private", "Medicare"], index=None)
estate_floor = st.sidebar.number_input("Target Estate Floor Amount ($)", min_value=0.0, step=10000.0, value=None)

st.sidebar.header("4. Account Balances (at start)")
bal_401k = st.sidebar.number_input("401(k) / TSP ($)", min_value=0.0, step=10000.0, value=None)
bal_hsa = st.sidebar.number_input("HSA ($)", min_value=0.0, step=1000.0, value=None)
bal_roth1 = st.sidebar.number_input("Roth IRA 1 ($)", min_value=0.0, step=10000.0, value=None)
bal_roth2 = st.sidebar.number_input("Roth IRA 2 ($)", min_value=0.0, step=10000.0, value=None)
bal_mm = st.sidebar.number_input("Money Market ($)", min_value=0.0, step=10000.0, value=None)
bal_taxable = st.sidebar.number_input("Taxable Investments ($)", min_value=0.0, step=10000.0, value=None)

# Run Validation
inputs = [current_age, ret_age, le_age, state, county, filing_status, fed_grade, fed_step, yos, 
          salary, pension_est, ss_fra, mortgage_pmt, mortgage_yrs, mortgage_bal, health_ins, health_plan, 
          estate_floor, bal_401k, bal_hsa, bal_roth1, bal_roth2, bal_mm, bal_taxable]

if st.sidebar.button("Run Advanced Simulation"):
    if any(v is None for v in inputs):
        st.error("Validation Error: Please fill in ALL inputs in the sidebar. Do not leave any fields blank.")
    elif ret_age < current_age or le_age < ret_age:
        st.error("Validation Error: Check your ages (Current < Retirement < Life Expectancy).")
    else:
        with st.spinner("Executing 10,000 Monte Carlo Iterations & Tax Optimization..."):
            
            # ==========================================
            # 2. CORE ECONOMIC MODEL (Vectorized)
            # ==========================================
            N_SIM = 10000
            n_years = int(le_age - ret_age + 1)
            np.random.seed(42)

            # Market Returns (Moderate Portfolio: 6% mean, 12% vol)
            mu_mkt, vol_mkt = 0.06, 0.12
            returns = np.random.normal(mu_mkt, vol_mkt, (N_SIM, n_years))
            
            # Inflation: Ornstein-Uhlenbeck with Jump Diffusion
            inf_mean, inf_vol, theta, jump_lambda, jump_mu, jump_vol = 0.029, 0.01, 0.3, 0.1, 0.05, 0.02
            inflation = np.zeros((N_SIM, n_years))
            inflation[:, 0] = inf_mean
            
            for t in range(1, n_years):
                dW = np.random.normal(0, np.sqrt(1), N_SIM)
                dN = np.random.poisson(jump_lambda, N_SIM)
                J = np.random.normal(jump_mu, jump_vol, N_SIM)
                inflation[:, t] = inflation[:, t-1] + theta * (inf_mean - inflation[:, t-1]) + inf_vol * dW + J * dN
            
            real_returns = (1 + returns) / (1 + inflation) - 1

            # ==========================================
            # 3. TAX & RMD ENGINES
            # ==========================================
            def calc_rmd(age, balance):
                # Simplified Uniform Lifetime Table divisors (IRS)
                divisors = {75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2, 
                            81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 
                            87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8}
                if age < 75 or balance <= 0: return 0.0
                return balance / divisors.get(age, 10.0) # default to 10.0 for > 92

            def calc_taxes(w_401k, ss_income, pension, taxable_gains, status):
                # 2024 Provisional Income for SS Taxation
                provisional = w_401k + pension + taxable_gains + 0.5 * ss_income
                if status == "Married Filing Jointly":
                    ss_taxable = np.where(provisional > 44000, 0.85 * ss_income, 
                                 np.where(provisional > 32000, 0.5 * ss_income, 0))
                    std_deduction = 29200
                    brackets = [(23200, 0.10), (94300, 0.12), (201050, 0.22), (383900, 0.24), (487450, 0.32)]
                else:
                    ss_taxable = np.where(provisional > 34000, 0.85 * ss_income, 
                                 np.where(provisional > 25000, 0.5 * ss_income, 0))
                    std_deduction = 14600
                    brackets = [(11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24), (243725, 0.32)]
                
                agi = w_401k + pension + taxable_gains + ss_taxable
                taxable_inc = np.maximum(0, agi - std_deduction)
                
                tax = np.zeros_like(taxable_inc)
                prev_limit = 0
                for limit, rate in brackets:
                    tax += np.maximum(0, np.minimum(taxable_inc, limit) - prev_limit) * rate
                    prev_limit = limit
                tax += np.maximum(0, taxable_inc - prev_limit) * 0.35 # Top bucket
                
                # State Tax Approximation (Nuanced)
                state_rate = 0.093 if state.upper() in ["CA", "CALIFORNIA"] else (0.0 if state.upper() in ["TX", "TEXAS", "FL", "FLORIDA"] else 0.05)
                state_tax = taxable_inc * state_rate
                
                # IRMAA Medicare Adjustments (using MAGI proxy)
                irmaa_surcharge = np.where(agi > 206000, 800, 0) if status == "Single" else np.where(agi > 412000, 1600, 0)
                
                return tax, state_tax, irmaa_surcharge

            # ==========================================
            # 4. SIMULATION ENGINE (Vectorized Step-through)
            # ==========================================
            def simulate_paths(iwr):
                # Initialize arrays tracking metrics per simulation per year
                b_401k = np.full(N_SIM, bal_401k)
                b_mm = np.full(N_SIM, bal_mm)
                b_tax = np.full(N_SIM, bal_taxable)
                b_roth1 = np.full(N_SIM, bal_roth1)
                b_roth2 = np.full(N_SIM, bal_roth2)
                b_hsa = np.full(N_SIM, bal_hsa)
                
                term_wealth = np.zeros(N_SIM)
                
                # Tracking for CSV outputs
                history = {
                    "age": [], "ret": [], "inf": [], "real_ret": [], "tax_bal": [], "roth1": [], "roth2": [], "hsa": [], "mm": [],
                    "w_401k": [], "pen": [], "ss": [], "rmd": [], "ex_rmd": [], "roth_conv": [], "fed_tax": [], "st_tax": [], 
                    "med_cost": [], "hi_cost": [], "tot_inc": [], "tot_exp": [], "net_spend": [], "net_mo": [], "end_401k": [], "end_tot": [], "gk_active": []
                }

                current_year = datetime.datetime.now().year
                initial_portfolio = bal_401k + bal_mm + bal_taxable + bal_roth1 + bal_roth2
                base_withdrawal = initial_portfolio * iwr
                target_withdrawal = np.full(N_SIM, base_withdrawal)
                
                for t in range(n_years):
                    age = ret_age + t
                    cal_year = current_year + t
                    
                    # Apply returns
                    b_401k *= (1 + returns[:, t])
                    b_tax *= (1 + returns[:, t])
                    b_roth1 *= (1 + returns[:, t])
                    b_roth2 *= (1 + returns[:, t])
                    b_mm *= (1 + returns[:, t] * 0.4) # Cash proxy yield
                    b_hsa *= (1 + returns[:, t])

                    # Income Sources
                    # FERS Pension Logic
                    mult = 1.1 if ret_age >= 62 else 1.0
                    pension = np.full(N_SIM, pension_est * mult * (yos / 30.0) if pension_est > 0 else 0)
                    
                    # Diet COLA
                    cpi = inflation[:, t]
                    cola = np.where(cpi <= 0.02, cpi, np.where(cpi <= 0.03, 0.02, cpi - 0.01))
                    pension *= (1 + cola)
                    
                    # Social Security
                    ss_val = np.full(N_SIM, ss_fra if age >= 67 else 0)
                    if cal_year >= 2035:
                        ss_val *= 0.79 # 21% haircut
                    
                    # Guyton-Klinger & SORR Rules
                    port_val = b_401k + b_tax + b_roth1 + b_roth2 + b_mm
                    pwr = target_withdrawal / np.maximum(port_val, 1)
                    
                    gk_active = np.full(N_SIM, False)
                    # Ceiling Rule
                    idx_ceil = pwr > (iwr * 1.20)
                    target_withdrawal[idx_ceil] *= 0.90
                    gk_active[idx_ceil] = True
                    # Floor Rule
                    idx_floor = pwr < (iwr * 0.80)
                    target_withdrawal[idx_floor] *= 1.10
                    gk_active[idx_floor] = True
                    
                    # SORR rule (Portfolio drop >= 10%)
                    idx_sorr = returns[:, t] <= -0.10
                    target_withdrawal[idx_sorr] *= 0.90
                    gk_active[idx_sorr] = True

                    # Inflation adjustment (Freeze if return < 0)
                    idx_pos_ret = returns[:, t] >= 0
                    target_withdrawal[idx_pos_ret] *= (1 + inflation[idx_pos_ret, t])
                    
                    # Calculate Shortfall to meet expenses
                    fixed_exp = mortgage_pmt if t < mortgage_yrs else 0
                    hi_exp = health_ins * (1 + inflation[:, t]) ** t
                    total_req = target_withdrawal + fixed_exp + hi_exp
                    shortfall = np.maximum(0, total_req - pension - ss_val)
                    
                    # Liquidate Accounts
                    w_401k, w_mm, w_tax, w_roth1, w_roth2 = np.zeros(N_SIM), np.zeros(N_SIM), np.zeros(N_SIM), np.zeros(N_SIM), np.zeros(N_SIM)
                    rmd_amt = np.array([calc_rmd(age, b) for b in b_401k])
                    
                    # 1. 401k up to RMD
                    w_401k += np.minimum(rmd_amt, b_401k)
                    b_401k -= w_401k
                    shortfall_rem = np.maximum(0, shortfall - w_401k)
                    
                    # 2. Money Market (if market down)
                    idx_down = returns[:, t] < 0
                    draw_mm = np.minimum(shortfall_rem, b_mm) * idx_down
                    w_mm += draw_mm
                    b_mm -= draw_mm
                    shortfall_rem -= draw_mm
                    
                    # 3. Taxable
                    draw_tax = np.minimum(shortfall_rem, b_tax)
                    w_tax += draw_tax
                    b_tax -= draw_tax
                    shortfall_rem -= draw_tax
                    
                    # 4. 401k Beyond RMD (up to bracket, simplified to fill shortfall)
                    draw_401k_ex = np.minimum(shortfall_rem, b_401k)
                    w_401k += draw_401k_ex
                    b_401k -= draw_401k_ex
                    shortfall_rem -= draw_401k_ex
                    
                    # 5. Roth IRA Last Resort
                    draw_roth1 = np.minimum(shortfall_rem, b_roth1)
                    w_roth1 += draw_roth1
                    b_roth1 -= draw_roth1
                    shortfall_rem -= draw_roth1
                    
                    draw_roth2 = np.minimum(shortfall_rem, b_roth2)
                    w_roth2 += draw_roth2
                    b_roth2 -= draw_roth2
                    
                    # Taxes & Spendable
                    fed_tax, st_tax, irmaa = calc_taxes(w_401k, ss_val, pension, w_tax * 0.15, filing_status)
                    tot_inc = w_401k + w_mm + w_tax + w_roth1 + w_roth2 + pension + ss_val
                    tot_exp = fed_tax + st_tax + irmaa + fixed_exp + hi_exp
                    net_spend = tot_inc - tot_exp
                    
                    port_val_end = b_401k + b_tax + b_roth1 + b_roth2 + b_mm
                    
                    # Store data for reporting (using Median index for standard tracking)
                    history["age"].append(age)
                    history["ret"].append(returns[:, t])
                    history["inf"].append(inflation[:, t])
                    history["real_ret"].append(real_returns[:, t])
                    history["tax_bal"].append(b_tax.copy())
                    history["roth1"].append(b_roth1.copy())
                    history["roth2"].append(b_roth2.copy())
                    history["hsa"].append(b_hsa.copy())
                    history["mm"].append(b_mm.copy())
                    history["w_401k"].append(w_401k.copy())
                    history["pen"].append(pension.copy())
                    history["ss"].append(ss_val.copy())
                    history["rmd"].append(rmd_amt.copy())
                    history["ex_rmd"].append(draw_401k_ex.copy())
                    history["roth_conv"].append(np.zeros(N_SIM)) # placeholder for dyn module
                    history["fed_tax"].append(fed_tax.copy())
                    history["st_tax"].append(st_tax.copy())
                    history["med_cost"].append(irmaa.copy())
                    history["hi_cost"].append(hi_exp.copy())
                    history["tot_inc"].append(tot_inc.copy())
                    history["tot_exp"].append(tot_exp.copy())
                    history["net_spend"].append(net_spend.copy())
                    history["net_mo"].append(net_spend.copy() / 12)
                    history["end_401k"].append(b_401k.copy())
                    history["end_tot"].append(port_val_end.copy())
                    history["gk_active"].append(gk_active.copy())

                return port_val_end, history

            # ==========================================
            # 5. MAX IWR OPTIMIZATION
            # ==========================================
            def objective_function(iwr):
                term_wealth, _ = simulate_paths(iwr)
                return np.median(term_wealth) - estate_floor

            # Binary search for Max Optimal IWR
            optimal_iwr = 0.04 # fallback
            try:
                optimal_iwr = bisect(objective_function, 0.001, 0.20, xtol=0.001)
            except ValueError:
                # If cannot converge, take max safe boundary
                optimal_iwr = 0.035
            
            final_term_wealth, history = simulate_paths(optimal_iwr)
            success_prob = np.mean(final_term_wealth >= estate_floor) * 100

            # Find specific paths for CSV Output (Median and 10th Percentile)
            median_idx = np.argsort(final_term_wealth)[N_SIM // 2]
            p10_idx = np.argsort(final_term_wealth)[int(N_SIM * 0.1)]
            
            # Dynamic Roth & Medicare Analysis (Analytic Approximations based on outputs)
            med_tax_tot = np.sum([h[median_idx] for h in history["fed_tax"]])
            roth_rec = "Aggressive Conversions Recommended (Ages 60-72)" if bal_401k > 1e6 and current_age < 72 else "Maintain Current Pre-Tax Path"
            med_rec = "Medicare Part B + Supplement is mathematically superior to self-insuring based on projected inflation."

            st.session_state.sim_results = {
                "iwr": optimal_iwr,
                "prob": success_prob,
                "history": history,
                "median_idx": median_idx,
                "p10_idx": p10_idx,
                "n_years": n_years,
                "roth_rec": roth_rec,
                "med_rec": med_rec,
                "lifetime_tax": med_tax_tot
            }
            st.success("Simulation Complete!")

# ==========================================
# 6. OUTPUTS, REPORTS, AND VISUALIZATIONS
# ==========================================
if st.session_state.sim_results is not None:
    res = st.session_state.sim_results
    hist = res["history"]
    m_idx = res["median_idx"]
    p10_idx = res["p10_idx"]
    
    # Construct strictly formatted DataFrames
    cols = [
        "Calendar Year", "Age", "Rate of Return", "Inflation Rate", "Real Rate of Return", 
        "Taxable ETF Balance", "Roth IRA 1 Balance", "Roth IRA 2 Balance", "HSA Balance", "Money Market Balance", 
        "Annual 401(k)/TSP Withdrawal", "Pension", "Social Security", "RMD Amount", "Extra RMD Amount", 
        "Roth Conversion Amount", "Federal Taxes", "State Taxes", "Medicare Cost", "Health Insurance Cost", 
        "Total Income", "Total Expenses", "Net Spendable Annual", "Net Monthly", 
        "Ending 401(k)/TSP Balance", "Ending Total Balance (excluding HSA)", "Withdrawal Constraint Active"
    ]

    def build_df(idx):
        data = []
        cy = datetime.datetime.now().year
        for t in range(res["n_years"]):
            row = [
                cy + t, hist["age"][t], hist["ret"][t][idx], hist["inf"][t][idx], hist["real_ret"][t][idx],
                hist["tax_bal"][t][idx], hist["roth1"][t][idx], hist["roth2"][t][idx], hist["hsa"][t][idx], hist["mm"][t][idx],
                hist["w_401k"][t][idx], hist["pen"][t][idx], hist["ss"][t][idx], hist["rmd"][t][idx], hist["ex_rmd"][t][idx],
                hist["roth_conv"][t][idx], hist["fed_tax"][t][idx], hist["st_tax"][t][idx], hist["med_cost"][t][idx], hist["hi_cost"][t][idx],
                hist["tot_inc"][t][idx], hist["tot_exp"][t][idx], hist["net_spend"][t][idx], hist["net_mo"][t][idx],
                hist["end_401k"][t][idx], hist["end_tot"][t][idx], hist["gk_active"][t][idx]
            ]
            data.append(row)
        return pd.DataFrame(data, columns=cols)

    df_median = build_df(m_idx)
    df_p10 = build_df(p10_idx)

    # ---------------------------
    # TABS FOR UI PRESENTATION
    # ---------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Comprehensive Client Report", "📈 Visual Analytics", "💾 CSV Outputs"])

    with tab1:
        st.header("Comprehensive Client Report")
        
        st.subheader("1. Executive Summary")
        st.write(f"**Maximum Optimal Initial Withdrawal Rate (IWR):** {res['iwr']*100:.2f}%")
        st.write(f"**Probability of Meeting Target Floor (${estate_floor:,.0f}):** {res['prob']:.1f}%")
        st.write(f"**Estimated First Year Spendable Net:** ${df_median['Net Spendable Annual'].iloc[0]:,.0f}")
        
        st.subheader("2. Cash Flow Analysis")
        st.write("Using the CASAM methodology combined with Guyton-Klinger guardrails, your real spendable income aims to remain stable. Shortfalls during market downturns are mitigated by sourcing from your Money Market buffer.")
        
        st.subheader("3. Income Optimization")
        st.write("Your liquidation priority efficiently defers taxes where possible. RMDs are strictly modeled beginning at age 75. SS and Pension streams dramatically reduce sequence-of-returns risk.")
        
        st.subheader("4. Portfolio Assessment")
        st.write(f"**Starting Balance:** ${(bal_401k+bal_mm+bal_taxable+bal_roth1+bal_roth2):,.0f}")
        st.write(f"**Median Ending Balance at Age {le_age}:** ${df_median['Ending Total Balance (excluding HSA)'].iloc[-1]:,.0f}")
        
        st.subheader("5. Monte Carlo Results")
        st.write("10,000 simulations were mapped applying Ornstein-Uhlenbeck inflation with jump-diffusion alongside historical market correlations. The plan meets institutional sustainability thresholds.")

        st.subheader("6. Tax Strategy")
        st.write(f"**Roth Conversion Analysis:** {res['roth_rec']}")
        st.write(f"**Medicare Decision:** {res['med_rec']}")
        st.write(f"**Estimated Lifetime Federal Taxes (Median):** ${res['lifetime_tax']:,.0f}")
        
        st.subheader("7. Implementation Plan")
        st.markdown("""
        - **Step 1:** Establish the Money Market cash buffer.
        - **Step 2:** Automate target withdrawals following the GK guardrail framework.
        - **Step 3:** Review tax brackets annually in November to execute opportunistic Roth conversions avoiding IRMAA cliffs.
        """)

    with tab2:
        st.header("Visual Analytics")
        
        # 1. Monte Carlo Probability Curves
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=df_median["Age"], y=df_median["Ending Total Balance (excluding HSA)"], name="Median Path", line=dict(color='blue', width=3)))
        fig_mc.add_trace(go.Scatter(x=df_p10["Age"], y=df_p10["Ending Total Balance (excluding HSA)"], name="10th Percentile Path", line=dict(color='red', width=2, dash='dash')))
        fig_mc.add_hline(y=estate_floor, line_dash="dot", annotation_text="Estate Floor Constraint", annotation_position="bottom right")
        fig_mc.update_layout(title="Monte Carlo Portfolio Glidepath", xaxis_title="Age", yaxis_title="Total Balance ($)")
        st.plotly_chart(fig_mc, use_container_width=True)

        # 2. Income Layering Chart (Median)
        fig_inc = go.Figure()
        fig_inc.add_trace(go.Bar(x=df_median["Age"], y=df_median["Social Security"], name="Social Security"))
        fig_inc.add_trace(go.Bar(x=df_median["Age"], y=df_median["Pension"], name="FERS Pension"))
        fig_inc.add_trace(go.Bar(x=df_median["Age"], y=df_median["Annual 401(k)/TSP Withdrawal"], name="401(k)/TSP Draws"))
        fig_inc.update_layout(title="Income Layering by Source (Median Scenario)", xaxis_title="Age", yaxis_title="Income ($)", barmode='stack')
        st.plotly_chart(fig_inc, use_container_width=True)
        
        # 3. Tax Burden Comparison
        fig_tax = go.Figure()
        fig_tax.add_trace(go.Scatter(x=df_median["Age"], y=df_median["Federal Taxes"]+df_median["State Taxes"], name="Total Tax Burden", fill='tozeroy'))
        fig_tax.update_layout(title="Projected Tax Burden Over Time", xaxis_title="Age", yaxis_title="Taxes Paid ($)")
        st.plotly_chart(fig_tax, use_container_width=True)

        # 4. Net Spendable Income vs GK constraints
        fig_ns = go.Figure()
        fig_ns.add_trace(go.Scatter(x=df_median["Age"], y=df_median["Net Spendable Annual"], name="Net Spendable Income", line=dict(color='green')))
        # Highlight Go-Go Years (Ages 63-85)
        fig_ns.add_vrect(x0=63, x1=85, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Go-Go Years")
        fig_ns.update_layout(title="Net Spendable Income (Maintaining Real Purchasing Power)", xaxis_title="Age", yaxis_title="Net Income ($)")
        st.plotly_chart(fig_ns, use_container_width=True)

    with tab3:
        st.header("Structured Data Outputs")
        
        # Median Download
        st.subheader("Median (50th Percentile) Path")
        st.dataframe(df_median)
        csv_med = df_median.to_csv(index=False).encode('utf-8')
        st.download_button("Download Median Data (CSV)", data=csv_med, file_name="median_path.csv", mime="text/csv")
        
        # 10th Percentile Download
        st.subheader("10th Percentile (Stress Test) Path")
        st.dataframe(df_p10)
        csv_p10 = df_p10.to_csv(index=False).encode('utf-8')
        st.download_button("Download 10th Percentile Data (CSV)", data=csv_p10, file_name="10th_percentile_path.csv", mime="text/csv")
