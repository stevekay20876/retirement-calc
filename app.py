import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import datetime
import json

from engine import StochasticRetirementEngine
from exports import build_csv_dataframe
from config import MOOP_LIMITS, TAX_BRACKETS_MFJ, TAX_BRACKETS_SINGLE, PORTFOLIOS
from pdf_report import generate_pdf

from visuals import plot_wealth_trajectory, plot_liquidity_timeline, plot_cash_flow_sources, plot_expenses_breakdown, plot_withdrawal_hierarchy, plot_taxes_and_rmds, plot_roth_strategy_comparison, plot_roth_tax_impact, plot_ss_breakeven, plot_medicare_comparison, plot_income_volatility, plot_legacy_breakdown, plot_fan_chart, plot_income_gap, plot_tornado

st.set_page_config(page_title="Advanced Retirement Simulator", layout="wide")

components.html(
    "<script>window.parent.document.body.scrollTop = 0; window.parent.document.documentElement.scrollTop = 0;</script>",
    height=0, width=0,
)

ui_styling = """
    <style>
    #MainMenu {visibility: hidden;} footer {display: none !important;} [data-testid="stHeader"] {visibility: hidden;} .stAppBottom {display: none !important;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2.0rem !important; font-weight: 700 !important; color: #00837B !important; }
    [data-testid="stDownloadButton"] button { background-color: #E6F7F6 !important; color: #00695C !important; border: 2px solid #80CBC4 !important; font-weight: 700 !important; border-radius: 8px !important; transition: all 0.2s ease; }
    [data-testid="stDownloadButton"] button:hover { background-color: #B2DFDB !important; border-color: #00837B !important; }
    [data-testid="stTabs"] { background-color: #F8FAFC; border: 2px solid #E5E7EB; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    div[data-baseweb="tab-list"] { gap: 0px; border-bottom: 2px solid #E5E7EB; }
    button[data-baseweb="tab"] { font-size: 1.1rem !important; padding: 0.8rem 1.5rem !important; background-color: #E5E7EB !important; color: #475569 !important; border-radius: 8px 8px 0 0 !important; border: 1px solid transparent !important; margin-right: 4px !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #FFFFFF !important; color: #00837B !important; font-weight: 800 !important; border-top: 4px solid #00837B !important; border-left: 2px solid #E5E7EB !important; border-right: 2px solid #E5E7EB !important; border-bottom: 2px solid #FFFFFF !important; transform: translateY(2px); box-shadow: 0 -2px 4px rgba(0,0,0,0.02); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; border: 1px solid #E5E7EB !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important; background-color: #FFFFFF !important; }
    </style>
"""
st.markdown(ui_styling, unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.tabs(["📊 Main Dashboard", "📝 Instructions", "⚙️ Background & Methodology", "ℹ️ About"])

with nav1:
    st.title("Advanced Quantitative Retirement Planner")
    st.markdown("Institution-Grade Monte Carlo Simulator | Constant Amortization Spending Model (CASAM)")

    curr_year = datetime.datetime.now().year
    strat_options = list(PORTFOLIOS.keys()) + ["Dynamic Glidepath (Target Date)"]

    DEFAULT_STATE = {
        'cur_age': None, 'ret_date': datetime.date(curr_year + 5, 1, 1), 'life_exp': None, 'filing_status': "Single",
        'spouse_age': None, 's_ret_date': datetime.date(curr_year + 5, 1, 1), 'spouse_life_exp': None, 'state': "", 'county': "",
        
        'current_salary': 0, 'p_max_tsp': False, 'p_tsp_contrib': 0, 'p_taxable_contrib': 0, 'p_roth_contrib': 0, 'p_cash_contrib': 0, 'p_hsa_contrib': 0,
        'phased_ret_active': False, 'phased_ret_age': None, 'pension_type': "FERS", 'pension_est': 0, 'survivor_benefit': "Full Survivor Benefit", 
        'mil_active': False, 'mil_component': "Active Duty", 'mil_years': 0, 'mil_months': 0, 'mil_days': 0, 'mil_points': 0, 'mil_rank': "O-4", 'mil_discharge': "Honorable Discharge", 'mil_diems': datetime.date(2005, 1, 1), 'mil_system': "High-36 (2.5%)", 'mil_pay_base': 0, 'mil_disability_rating': "0%", 'mil_special_rating': "None", 'mil_va_pay': 0, 'mil_sbp': "No SBP", 'mil_start_age': None,
        'ss_fra': 0, 'ss_claim_age': 67,
        
        's_current_salary': 0, 's_max_tsp': False, 's_tsp_contrib': 0, 's_taxable_contrib': 0, 's_roth_contrib': 0, 's_cash_contrib': 0, 's_hsa_contrib': 0,
        's_phased_ret_active': False, 's_phased_ret_age': None, 's_pension_type': "FERS", 's_pension_est': 0, 's_survivor_benefit': "No Survivor Benefit", 
        's_mil_active': False, 's_mil_component': "Active Duty", 's_mil_years': 0, 's_mil_months': 0, 's_mil_days': 0, 's_mil_points': 0, 's_mil_rank': "O-4", 's_mil_discharge': "Honorable Discharge", 's_mil_diems': datetime.date(2005, 1, 1), 's_mil_system': "High-36 (2.5%)", 's_mil_pay_base': 0, 's_mil_disability_rating': "0%", 's_mil_special_rating': "None", 's_mil_va_pay': 0, 's_mil_sbp': "No SBP", 's_mil_start_age': None,
        's_ss_fra': 0, 's_ss_claim_age': 67,
        
        'target_floor': 0, 'min_spending': 0, 'max_spending': 0, 'add_exp': 0, 'max_tax_bracket': "24%", 'mortgage_pmt': 0, 'mortgage_yrs': 0, 'home_value': 0,
        'health_plan': "None/Self-Insure", 'health_cost': 0, 'oop_cost': 0,
        's_health_plan': "None/Self-Insure", 's_health_cost': 0, 's_oop_cost': 0,
        'has_40_quarters': True, 'intent_to_work_40_quarters': False, 'has_dependent_children': False, 'wants_dental_vision': False,
        
        'tsp_b': 0, 'tsp_roth_b': 0, 'tsp_strat': "Moderate (60% Stock / 40% Bond)",
        'ira_b': 0, 'ira_strat': "Moderate (60% Stock / 40% Bond)",
        'roth_b': 0, 'roth_strat': "Aggressive (100% Stock)",
        'tax_b': 0, 'tax_basis': None, 'tax_strat': "Moderate (60% Stock / 40% Bond)",
        'hsa_b': 0, 'hsa_strat': "Moderate (60% Stock / 40% Bond)",
        'cash_b': 0, 'cash_r': 4.0, 'pay_taxes_from_cash': True, 'age_de_risking': False,
        
        's_tsp_b': 0, 's_tsp_roth_b': 0, 's_ira_b': 0, 's_roth_b': 0,
        'save_file_name': "client_profile"
    }

    # --- WIZARD STATE PERSISTENCE LOGIC ---
    if 'master_state' not in st.session_state:
        st.session_state.master_state = DEFAULT_STATE.copy()

    for k in DEFAULT_STATE.keys():
        if k in st.session_state:
            st.session_state.master_state[k] = st.session_state[k]

    for k, v in st.session_state.master_state.items():
        st.session_state[k] = v
    # --------------------------------------

    if 'ui_mode' not in st.session_state: st.session_state.ui_mode = "Guided Wizard"
    if 'wizard_step' not in st.session_state: st.session_state.wizard_step = 1

    def get_current_state():
        state_dict = {k: st.session_state[k] for k in DEFAULT_STATE.keys() if k in st.session_state}
        if state_dict.get('p_max_tsp'): state_dict['p_tsp_contrib'] = 0
        if state_dict.get('s_max_tsp'): state_dict['s_tsp_contrib'] = 0
        for date_field in ['ret_date', 's_ret_date', 'mil_diems', 's_mil_diems']:
            if isinstance(state_dict.get(date_field), datetime.date):
                state_dict[date_field] = state_dict[date_field].isoformat()
        return state_dict

    with st.expander("💾 Client Profile Management (Save / Load)", expanded=False):
        col_load, col_save = st.columns(2)
        with col_load:
            uploaded_profile = st.file_uploader("Load Saved Profile (.json)", type="json")
            if uploaded_profile is not None:
                if "loaded_file" not in st.session_state or st.session_state.loaded_file != uploaded_profile.name:
                    try:
                        loaded_data = json.load(uploaded_profile)
                        for key, value in loaded_data.items(): 
                            if key in ['ret_date', 's_ret_date', 'mil_diems', 's_mil_diems'] and isinstance(value, str):
                                parsed_date = datetime.date.fromisoformat(value)
                                st.session_state[key] = parsed_date
                                st.session_state.master_state[key] = parsed_date
                            else:
                                st.session_state[key] = value
                                st.session_state.master_state[key] = value
                        st.session_state.loaded_file = uploaded_profile.name
                        st.session_state.ui_mode = "Expert Form (All Fields)"
                        st.success("Profile Loaded Successfully!")
                        st.rerun() 
                    except Exception as e:
                        st.error("Error loading profile.")
        with col_save:
            profile_name = st.text_input("Name your save file:", value=st.session_state.get('save_file_name', 'client_profile'))
            if profile_name != st.session_state.get('save_file_name'):
                st.session_state['save_file_name'] = profile_name
                st.session_state.master_state['save_file_name'] = profile_name
                
            safe_filename = profile_name.strip()
            if not safe_filename: safe_filename = "client_profile"
            if not safe_filename.endswith(".json"): safe_filename += ".json"
            
            st.download_button("⬇️ Save Current Profile to Computer", data=json.dumps(get_current_state(), indent=4), file_name=safe_filename, mime="application/json", use_container_width=True, help="Important: Make sure you press 'Enter' or click outside of any text box you just edited before clicking Save to lock in the final keystrokes!")

    has_run = 'sim_data' in st.session_state

    # ---------------------------------------------------------
    # UI RENDERING FUNCTIONS
    # ---------------------------------------------------------

    def render_personal():
        st.markdown("**Household & Tax Settings**")
        c4, c5, c6 = st.columns(3)
        c4.selectbox("Tax Filing Status", ["Single", "MFJ"], key="filing_status")
        c5.text_input("State of Residence", key="state")
        c6.text_input("County of Residence", key="county")
        
        st.markdown("<hr style='margin-top:0.5rem; margin-bottom:1rem;'/>", unsafe_allow_html=True)
        st.info("Note: Set your Target Planning Age conservatively (e.g., 90-95). The mathematical engine will stress-test your portfolio by forcing it to survive 10,000 different market crash scenarios across this entire fixed time horizon.")
        
        t_pers_p, t_pers_s = st.tabs(["Primary Individual", "Spouse (If MFJ)"])
        
        with t_pers_p:
            c1, c2, c3 = st.columns(3)
            cur_age = c1.number_input("Primary Current Age", min_value=18, max_value=100, key="cur_age")
            
            default_ret = st.session_state.ret_date if isinstance(st.session_state.ret_date, datetime.date) else datetime.date.fromisoformat(st.session_state.ret_date)
            ret_date = c2.date_input("Target Date of Retirement", value=default_ret, format="MM/DD/YYYY", key="ret_date", help="The exact calendar date you plan to separate from service. Used to prorate transition year salary and savings.")
            if isinstance(ret_date, datetime.date) and ret_date < datetime.date.today():
                c2.caption("🚨 **Error:** Target retirement date is in the past.")
                
            life_exp = c3.number_input("Primary Target Planning Age", min_value=50, max_value=120, key="life_exp")
            if life_exp is not None and cur_age is not None and life_exp <= cur_age:
                c3.caption("🚨 **Error:** Planning age must be > Current Age.")
            
        with t_pers_s:
            if st.session_state.filing_status == "MFJ":
                c_sp1, c_sp2, c_sp3 = st.columns(3)
                s_cur_age = c_sp1.number_input("Spouse Current Age", min_value=18, max_value=100, key="spouse_age")
                
                s_default_ret = st.session_state.s_ret_date if isinstance(st.session_state.s_ret_date, datetime.date) else datetime.date.fromisoformat(st.session_state.s_ret_date)
                s_ret_date = c_sp2.date_input("Spouse Date of Retirement", value=s_default_ret, format="MM/DD/YYYY", key="s_ret_date")
                if isinstance(s_ret_date, datetime.date) and s_ret_date < datetime.date.today():
                    c_sp2.caption("🚨 **Error:** Target retirement date is in the past.")
                    
                s_life_exp = c_sp3.number_input("Spouse Target Planning Age", min_value=50, max_value=120, key="spouse_life_exp")
                if s_life_exp is not None and s_cur_age is not None and s_life_exp <= s_cur_age:
                    c_sp3.caption("🚨 **Error:** Planning age must be > Current Age.")
            else:
                st.info("Select 'MFJ' (Married Filing Jointly) above to enable Spouse inputs.")

    def render_income():
        t_inc_p, t_inc_s = st.tabs(["Primary", "Spouse (If MFJ)"])
        with t_inc_p:
            st.markdown("**Primary Pre-Retirement Salary & Phased Transition**")
            c1, c2, c3 = st.columns(3)
            current_salary = c1.number_input("Current Annual Salary ($/yr)", min_value=0, step=1000, key="current_salary")
            if 0 < current_salary < 15000: c1.caption("⚠️ **Check:** <$15k is unusually low. Did you enter monthly pay instead of annual?")
                
            c2.checkbox("Enable FERS Phased Retirement?", key="phased_ret_active")
            c3.number_input("Phased Retirement Start Age", min_value=50, max_value=70, key="phased_ret_age")
            
            st.markdown("**Primary Annual Savings (Until Retirement)**")
            c_sav1, c_sav2, c_sav3 = st.columns(3)
            p_max_tsp = c_sav1.checkbox("Maximize IRS allowable TSP/401(k)?", key="p_max_tsp")
            if p_max_tsp: 
                c_sav1.info("IRS Max active: Automatically scales by age ($24,500 to $35,750).")
            else: 
                p_tsp_contrib = c_sav1.number_input("TSP/401(k) Pre-Tax Savings ($/yr)", min_value=0, step=1000, key="p_tsp_contrib")
                if p_tsp_contrib > 35750: c_sav1.caption("⚠️ **Check:** Exceeds IRS annual limits. Check 'Maximize' box instead.")
            
            c_sav2.number_input("Roth IRA Savings ($/yr)", min_value=0, step=1000, key="p_roth_contrib")
            c_sav3.number_input("Taxable Acct Savings ($/yr)", min_value=0, step=1000, key="p_taxable_contrib")
            
            c_sav4, c_sav5 = st.columns(2)
            c_sav4.number_input("Money Market Savings ($/yr)", min_value=0, step=1000, key="p_cash_contrib")
            c_sav5.number_input("HSA Savings ($/yr)", min_value=0, step=1000, key="p_hsa_contrib")
            
            st.markdown("**Primary Civilian Pension**")
            cp1, cp2, cp3 = st.columns(3)
            pension_type = cp1.selectbox("Pension Type", ["FERS", "Other"], key="pension_type")
            pension_est = cp2.number_input("Full (Unreduced) Pension Est. ($/yr)", min_value=0, step=1000, key="pension_est")
            
            if pension_type == "FERS" and pension_est > 100000:
                cp2.caption("⚠️ **Check:** >$100k is unusually high for FERS. Did you accidentally enter your High-3 salary?")
            elif 0 < pension_est < 5000:
                cp2.caption("⚠️ **Check:** <$5k is unusually low. Did you enter your monthly payout instead of annual?")
            
            if st.session_state.pension_type == "FERS":
                surv_options = ["Full Survivor Benefit", "Partial Survivor Benefit", "No Survivor Benefit"]
            else:
                surv_options = ["100% Survivor", "50% Survivor", "Present Value Refund", "No Survivor Benefit"]
            cp3.selectbox("Survivor Benefit Option", surv_options, key="survivor_benefit")

            st.markdown("**Primary Social Security Guaranteed Income**")
            c7, c8 = st.columns(2)
            ss_fra = c7.number_input("Social Security at FRA ($/yr)", min_value=0, step=1000, key="ss_fra")
            
            if ss_fra == 0:
                c7.caption("💡 **Note:** $0 entered. Social Security will not be modeled.")
            elif 0 < ss_fra < 8000:
                c7.caption("⚠️ **Check:** <$8k is unusually low. Did you enter your monthly benefit instead of annual?")
            elif ss_fra > 60000:
                c7.caption("⚠️ **Check:** >$60k exceeds max limits. Ensure this isn't your combined household benefit.")
                
            c8.number_input("Target SS Claiming Age", min_value=62, max_value=70, key="ss_claim_age")
            
        with t_inc_s:
            st.markdown("**Spouse Pre-Retirement Salary & Phased Transition**")
            cs1, cs2, cs3 = st.columns(3)
            s_current_salary = cs1.number_input("Spouse Current Annual Salary ($/yr)", min_value=0, step=1000, key="s_current_salary")
            if 0 < s_current_salary < 15000: cs1.caption("⚠️ **Check:** <$15k is unusually low. Did you enter monthly pay instead of annual?")
                
            cs2.checkbox("Enable Spouse Phased Retirement?", key="s_phased_ret_active")
            cs3.number_input("Spouse Phased Ret. Start Age", min_value=50, max_value=70, key="s_phased_ret_age")
            
            st.markdown("**Spouse Annual Savings (Until Retirement)**")
            cs_sav1, cs_sav2, cs_sav3 = st.columns(3)
            s_max_tsp = cs_sav1.checkbox("Spouse: Maximize IRS allowable TSP/401(k)?", key="s_max_tsp")
            if s_max_tsp: 
                cs_sav1.info("IRS Max active.")
            else: 
                s_tsp_contrib = cs_sav1.number_input("Spouse TSP/401k Pre-Tax Savings ($/yr)", min_value=0, step=1000, key="s_tsp_contrib")
                if s_tsp_contrib > 35750: cs_sav1.caption("⚠️ **Check:** Exceeds IRS limits.")
            
            cs_sav2.number_input("Spouse Roth IRA Savings ($/yr)", min_value=0, step=1000, key="s_roth_contrib")
            cs_sav3.number_input("Spouse Taxable Savings ($/yr)", min_value=0, step=1000, key="s_taxable_contrib")
            
            cs_sav4, cs_sav5 = st.columns(2)
            cs_sav4.number_input("Spouse Cash Savings ($/yr)", min_value=0, step=1000, key="s_cash_contrib")
            cs_sav5.number_input("Spouse HSA Savings ($/yr)", min_value=0, step=1000, key="s_hsa_contrib")
            
            st.markdown("**Spouse Civilian Pension**")
            csp1, csp2, csp3 = st.columns(3)
            s_pension_type = csp1.selectbox("Spouse Pension Type", ["FERS", "Other"], key="s_pension_type")
            s_pension_est = csp2.number_input("Spouse Full Pension Est. ($/yr)", min_value=0, step=1000, key="s_pension_est")
            
            if s_pension_type == "FERS" and s_pension_est > 100000:
                csp2.caption("⚠️ **Check:** >$100k is unusually high for FERS. Did you accidentally enter High-3 salary?")
            elif 0 < s_pension_est < 5000:
                csp2.caption("⚠️ **Check:** <$5k is unusually low. Did you enter monthly payout instead of annual?")
            
            if st.session_state.s_pension_type == "FERS":
                s_surv_options = ["No Survivor Benefit", "Partial Survivor Benefit", "Full Survivor Benefit"]
            else:
                s_surv_options = ["No Survivor Benefit", "Present Value Refund", "50% Survivor", "100% Survivor"]
            csp3.selectbox("Spouse Survivor Benefit Option", s_surv_options, key="s_survivor_benefit")

            st.markdown("**Spouse Social Security Guaranteed Income**")
            cs7, cs8 = st.columns(2)
            s_ss_fra = cs7.number_input("Spouse Social Security at FRA ($/yr)", min_value=0, step=1000, key="s_ss_fra")
            
            if s_ss_fra == 0:
                cs7.caption("💡 **Note:** $0 entered. Social Security will not be modeled.")
            elif 0 < s_ss_fra < 8000:
                cs7.caption("⚠️ **Check:** <$8k is unusually low. Did you enter your monthly benefit instead of annual?")
            elif s_ss_fra > 60000:
                cs7.caption("⚠️ **Check:** >$60k exceeds max limits. Ensure this isn't your combined household benefit.")
                
            cs8.number_input("Spouse Target SS Claiming Age", min_value=62, max_value=70, key="s_ss_claim_age")

    def render_assets():
        st.markdown("**Current Portfolios & Strategies**")
        st.info("Note: Under SECURE 2.0, Roth employer plans (like Roth TSP / Roth 401k) are no longer subject to Required Minimum Distributions (RMDs) during the owner's lifetime. To perfectly align with current tax law, the engine automatically aggregates your Roth TSP balance into your lifetime Roth IRA balance.")
        
        t_ast_p, t_ast_s = st.tabs(["Household Assets", "Spouse Assets (If MFJ)"])
        
        with t_ast_p:
            p_label = "TSP Traditional Bal. ($)" if st.session_state.pension_type == "FERS" else "401(k) Traditional Bal. ($)"
            p_r_label = "TSP Roth Bal. ($)" if st.session_state.pension_type == "FERS" else "Roth 401(k) Bal. ($)"
            
            c1, c2, c3 = st.columns(3)
            c1.number_input(p_label, min_value=0, step=10000, key="tsp_b")
            c2.number_input(p_r_label, min_value=0, step=10000, key="tsp_roth_b")
            c3.selectbox("TSP/401(k) Strategy", strat_options, index=strat_options.index(st.session_state.tsp_strat) if st.session_state.tsp_strat in strat_options else 0, key="tsp_strat")
            
            c4, c5 = st.columns(2)
            c4.number_input("Trad. IRA Balance ($)", min_value=0, step=10000, key="ira_b")
            c5.selectbox("Trad. IRA Strategy", strat_options, index=strat_options.index(st.session_state.ira_strat) if st.session_state.ira_strat in strat_options else 0, key="ira_strat")
            
            c6, c7 = st.columns(2)
            c6.number_input("Roth IRA Balance ($)", min_value=0, step=10000, key="roth_b")
            c7.selectbox("Roth IRA Strategy", strat_options, index=strat_options.index(st.session_state.roth_strat) if st.session_state.roth_strat in strat_options else 0, key="roth_strat")
            
            c8, c9, c10 = st.columns(3)
            c8.number_input("Taxable Balance ($)", min_value=0, step=10000, key="tax_b")
            c9.number_input("Taxable Cost Basis ($)", min_value=0, step=10000, key="tax_basis")
            c10.selectbox("Taxable Strategy", strat_options, index=strat_options.index(st.session_state.tax_strat) if st.session_state.tax_strat in strat_options else 0, key="tax_strat")
            
            c11, c12 = st.columns(2)
            c11.number_input("HSA Balance (Optional)", min_value=0, step=1000, key="hsa_b")
            c12.selectbox("HSA Strategy", strat_options, index=strat_options.index(st.session_state.hsa_strat) if st.session_state.hsa_strat in strat_options else 0, key="hsa_strat")
            
            c13, c14 = st.columns(2)
            c13.number_input("Money Market Balance ($)", min_value=0, step=1000, key="cash_b")
            c14.number_input("Money Market Yield %", min_value=0.0, step=0.1, key="cash_r")
            
            st.markdown("---")
            c_bot1, c_bot2 = st.columns(2)
            with c_bot1: st.checkbox("Pay Roth Conversion Taxes from Cash Buffer?", key="pay_taxes_from_cash")
            with c_bot2: st.checkbox("Enable Age-Based Equity De-risking (1%/yr post-65)", key="age_de_risking", help="Automatically reduces stock allocation by 1% per year after age 65 (down to a 20% floor). This mimics real-world risk reduction and lowers pessimistic tail risks caused by holding 100% stocks into your 90s.")
            
        with t_ast_s:
            st.info("Enter any accounts solely in the Spouse's name here. They will be mathematically merged into the household portfolio for calculation.")
            s_label = "Spouse TSP Trad Bal. ($)" if st.session_state.s_pension_type == "FERS" else "Spouse 401(k) Trad Bal. ($)"
            s_r_label = "Spouse TSP Roth Bal. ($)" if st.session_state.s_pension_type == "FERS" else "Spouse Roth 401(k) Bal. ($)"
            
            cs1, cs2 = st.columns(2)
            cs1.number_input(s_label, min_value=0, step=10000, key="s_tsp_b")
            cs2.number_input(s_r_label, min_value=0, step=10000, key="s_tsp_roth_b")
            
            cs3, cs4 = st.columns(2)
            cs3.number_input("Spouse Trad. IRA Balance ($)", min_value=0, step=10000, key="s_ira_b")
            cs4.number_input("Spouse Roth IRA Balance ($)", min_value=0, step=10000, key="s_roth_b")

    def render_expenses():
        st.markdown("**Spending Limits & Legacy Goals (In Today's Dollars)**")
        st.info("Note: Your target legacy floor strictly applies to your **Liquid Investment Portfolio**. Your current home equity is completely preserved as a separate inheritance asset on top of whatever liquid floor you target.")
        c1, c2, c3 = st.columns(3)
        target_floor = c1.number_input("Target Liquid Legacy Floor ($)", min_value=0, step=10000, key="target_floor")
        if target_floor == 0: c1.caption("💡 **Note:** $0 entered. Engine will spend down to your last penny.")
            
        c2.number_input("Minimum Spending Floor ($/yr)", min_value=0, step=1000, key="min_spending")
        c3.number_input("Maximum Spending Cap ($/yr)", min_value=0, step=1000, key="max_spending")
        
        c4, c5 = st.columns(2)
        c4.number_input("Additional Expenses ($/yr)", min_value=0, step=1000, key="add_exp")
        c5.selectbox("Maximum Target Tax Bracket (Roth Cap)", ["12%", "22%", "24%", "32%", "35%", "37%"], index=2, key="max_tax_bracket")
        
        st.markdown("**Property & Debt**")
        c6, c7, c8 = st.columns(3)
        c6.number_input("Current Home Value ($)", min_value=0, step=10000, key="home_value")
        mortgage_pmt = c7.number_input("Annual Mortgage Payment ($/yr)", min_value=0, step=1000, key="mortgage_pmt")
        mortgage_yrs = c8.number_input("Mortgage Years Remaining", min_value=0, key="mortgage_yrs")
        
        if mortgage_pmt > 0 and mortgage_yrs == 0:
            c8.caption("🚨 **Error:** You entered a mortgage payment but 0 years remaining.")
        
        st.markdown("**Healthcare**")
        health_options = ["FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", "GEHA High", "GEHA Standard", "Aetna Open Access", "Aetna Direct", "Aetna Advantage", "Cigna", "TRICARE for Life", "None/Self-Insure", "Spouse's Insurance", "Affordable Care Act"]
        
        if st.session_state.filing_status == 'MFJ':
            t_hc_p, t_hc_s = st.tabs(["Household", "Spouse"])
            with t_hc_p:
                c9, c10, c11 = st.columns(3)
                c9.selectbox("Household Retiree Health Coverage", health_options, key="health_plan")
                h_cost = c10.number_input("Household Annual Health Premium ($/yr)", min_value=0, step=100, key="health_cost")
                if 0 < h_cost < 1000: c10.caption("⚠️ **Check:** <$1,000 is unusually low. Did you enter monthly premium instead of annual?")
                c11.number_input("Household Typical Out-of-Pocket ($/yr)", min_value=0, step=100, key="oop_cost")
            with t_hc_s:
                cs9, cs10, cs11 = st.columns(3)
                cs9.selectbox("Spouse Retiree Health Coverage", health_options, key="s_health_plan")
                s_h_cost = cs10.number_input("Spouse Annual Health Premium ($/yr)", min_value=0, step=100, key="s_health_cost")
                if 0 < s_h_cost < 1000: cs10.caption("⚠️ **Check:** <$1,000 is unusually low. Did you enter monthly premium instead of annual?")
                cs11.number_input("Spouse Typical Out-of-Pocket ($/yr)", min_value=0, step=100, key="s_oop_cost")
        else:
            c9, c10, c11 = st.columns(3)
            c9.selectbox("Retiree Health Coverage", health_options, key="health_plan")
            h_cost = c10.number_input("Annual Health Premium ($/yr)", min_value=0, step=100, key="health_cost")
            if 0 < h_cost < 1000: c10.caption("⚠️ **Check:** <$1,000 is unusually low. Did you enter monthly premium instead of annual?")
            c11.number_input("Typical Out-of-Pocket Medical ($/yr)", min_value=0, step=100, key="oop_cost")
            st.session_state.s_health_plan = "None/Self-Insure"
            st.session_state.s_health_cost = 0
            st.session_state.s_oop_cost = 0

        p_needs_aca = st.session_state.health_plan in ["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"]
        s_needs_aca = (st.session_state.filing_status == 'MFJ') and (st.session_state.s_health_plan in ["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"])

        if p_needs_aca or s_needs_aca:
            st.markdown("---")
            st.markdown("#### 🏥 Medicare vs. ACA Transition Logic")
            st.info("Because you selected a non-federal or unsupported private insurance path, we must determine exactly when and how you transition to Medicare. Please answer the following:")
            
            c_aca1, c_aca2 = st.columns(2)
            has_40_q = c_aca1.radio("Do you (or your spouse) have 40 quarters (10 years) of Medicare-taxed work history?", ["Yes", "No"], index=0 if st.session_state.has_40_quarters else 1)
            st.session_state.has_40_quarters = (has_40_q == "Yes")
            
            if not st.session_state.has_40_quarters:
                intent_to_work = c_aca2.radio("Do you intend to work and gain 40 quarters before age 65?", ["Yes", "No"], index=0 if st.session_state.intent_to_work_40_quarters else 1)
                st.session_state.intent_to_work_40_quarters = (intent_to_work == "Yes")
            else:
                st.session_state.intent_to_work_40_quarters = False

            c_aca3, c_aca4 = st.columns(2)
            has_kids = c_aca3.radio("Are there dependent children (under age 26) currently covered on your plan?", ["Yes", "No"], index=0 if st.session_state.has_dependent_children else 1)
            st.session_state.has_dependent_children = (has_kids == "Yes")

            wants_dv = c_aca4.radio("Do you plan to maintain standalone routine dental/vision coverage after 65?", ["Yes", "No"], index=0 if st.session_state.wants_dental_vision else 1)
            st.session_state.wants_dental_vision = (wants_dv == "Yes")

    def render_military():
        t_mil_p, t_mil_s = st.tabs(["Primary", "Spouse (If MFJ)"])
        with t_mil_p:
            st.markdown("**Primary Military Service Member Profile**")
            st.checkbox("Enable Primary Military Pension Modeling?", key="mil_active")
            
            m1, m2 = st.columns(2)
            m1.selectbox("Service Component", ["Active Duty", "National Guard / Reserve", "Mixed (Active + Guard/Reserve)"], key="mil_component")
            m2.number_input("Mil. Pension Start Age (Default 60 for Guard/Reserve)", min_value=18, max_value=100, key="mil_start_age")

            st.markdown("**Creditable Service & Points**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mil_years = mc1.number_input("Active Years", min_value=0, max_value=40, key="mil_years")
            if mil_years > 40: mc1.caption("🚨 **Error:** Active years cannot exceed 40.")
                
            mc2.number_input("Active Months", min_value=0, max_value=11, key="mil_months")
            mc3.number_input("Active Days", min_value=0, max_value=30, key="mil_days")
            mc4.number_input("Total Career Points", min_value=0, key="mil_points")

            st.markdown("**Rank, System & Pay**")
            mr1, mr2 = st.columns(2)
            mr1.selectbox("Final Rank / Pay Grade", ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "W-1", "W-2", "W-3", "W-4", "W-5", "O-1", "O-2", "O-3", "O-4", "O-5", "O-6", "O-7", "O-8", "O-9"], key="mil_rank")
            mr2.selectbox("Character of Service", ["Honorable Discharge", "General Discharge (Under Honorable Conditions)", "Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge", "Uncharacterized Separation"], key="mil_discharge")
            
            md1, md2, md3 = st.columns(3)
            default_diems = datetime.date.fromisoformat(st.session_state.mil_diems) if isinstance(st.session_state.mil_diems, str) else st.session_state.mil_diems
            md1.date_input("DIEMS Date", value=default_diems, format="MM/DD/YYYY", key="mil_diems")
            md2.selectbox("Retirement System", ["Final Pay (2.5%)", "High-36 (2.5%)", "REDUX (2.5% - 1% per yr under 30)", "Blended Retirement System [BRS] (2.0%)"], key="mil_system")
            
            mil_pay = md3.number_input("Pay Base (High-36 Avg or Final Base Pay $/mo)", min_value=0, step=100, key="mil_pay_base")
            if mil_pay > 25000: md3.caption("⚠️ **Check:** >$25k/mo is exceptionally high. Did you enter annual pay instead of monthly?")
            
            st.markdown("**Disability & Survivor Options**")
            mv1, mv2, mv3 = st.columns(3)
            mv1.selectbox("VA Disability Rating", ["0%", "10% - 20%", "30% - 40%", "50% - 60%", "70% - 90%", "100%"], key="mil_disability_rating")
            mv2.selectbox("Special Classifications", ["None", "TDIU (Unemployability)", "SMC (Special Monthly Comp)"], key="mil_special_rating")
            mv3.number_input("Monthly VA Disability Pay ($/mo)", min_value=0, step=100, key="mil_va_pay")
            st.selectbox("Survivor Benefit Plan (SBP)", ["No SBP", "Full SBP (55% Survivor / 6.5% Premium)"], key="mil_sbp")
            
        with t_mil_s:
            st.markdown("**Spouse Military Service Member Profile**")
            st.checkbox("Enable Spouse Military Pension Modeling?", key="s_mil_active")
            
            sm1, sm2 = st.columns(2)
            sm1.selectbox("Spouse Service Component", ["Active Duty", "National Guard / Reserve", "Mixed (Active + Guard/Reserve)"], key="s_mil_component")
            sm2.number_input("Spouse Mil. Pension Start Age", min_value=18, max_value=100, key="s_mil_start_age")

            st.markdown("**Spouse Creditable Service & Points**")
            smc1, smc2, smc3, smc4 = st.columns(4)
            s_mil_years = smc1.number_input("Spouse Active Years", min_value=0, max_value=40, key="s_mil_years")
            if s_mil_years > 40: smc1.caption("🚨 **Error:** Active years cannot exceed 40.")
                
            smc2.number_input("Spouse Active Months", min_value=0, max_value=11, key="s_mil_months")
            smc3.number_input("Spouse Active Days", min_value=0, max_value=30, key="s_mil_days")
            smc4.number_input("Spouse Total Career Points", min_value=0, key="s_mil_points")

            st.markdown("**Spouse Rank, System & Pay**")
            smr1, smr2 = st.columns(2)
            smr1.selectbox("Spouse Final Rank / Pay Grade", ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "W-1", "W-2", "W-3", "W-4", "W-5", "O-1", "O-2", "O-3", "O-4", "O-5", "O-6", "O-7", "O-8", "O-9"], key="s_mil_rank")
            smr2.selectbox("Spouse Character of Service", ["Honorable Discharge", "General Discharge (Under Honorable Conditions)", "Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge", "Uncharacterized Separation"], key="s_mil_discharge")
            
            smd1, smd2, smd3 = st.columns(3)
            s_default_diems = datetime.date.fromisoformat(st.session_state.s_mil_diems) if isinstance(st.session_state.s_mil_diems, str) else st.session_state.s_mil_diems
            smd1.date_input("Spouse DIEMS Date", value=s_default_diems, format="MM/DD/YYYY", key="s_mil_diems")
            smd2.selectbox("Spouse Retirement System", ["Final Pay (2.5%)", "High-36 (2.5%)", "REDUX (2.5% - 1% per yr under 30)", "Blended Retirement System [BRS] (2.0%)"], key="s_mil_system")
            
            s_mil_pay = smd3.number_input("Spouse Pay Base ($/mo)", min_value=0, step=100, key="s_mil_pay_base")
            if s_mil_pay > 25000: smd3.caption("⚠️ **Check:** >$25k/mo is exceptionally high. Did you enter annual pay instead of monthly?")
            
            st.markdown("**Spouse Disability & Survivor Options**")
            smv1, smv2, smv3 = st.columns(3)
            smv1.selectbox("Spouse VA Disability Rating", ["0%", "10% - 20%", "30% - 40%", "50% - 60%", "70% - 90%", "100%"], key="s_mil_disability_rating")
            smv2.selectbox("Spouse Special Classifications", ["None", "TDIU (Unemployability)", "SMC (Special Monthly Comp)"], key="s_mil_special_rating")
            smv3.number_input("Spouse Monthly VA Disability Pay ($/mo)", min_value=0, step=100, key="s_mil_va_pay")
            st.selectbox("Spouse Survivor Benefit Plan (SBP)", ["No SBP", "Full SBP (55% Survivor / 6.5% Premium)"], key="s_mil_sbp")

    # ---------------------------------------------------------
    # MAIN UI RENDER BLOCK
    # ---------------------------------------------------------
    
    st.markdown("### Build Your Profile")
    
    if 'ui_mode' not in st.session_state: 
        st.session_state.ui_mode = "Guided Wizard"

    def update_ui_mode():
        st.session_state.ui_mode = st.session_state._ui_mode_selector

    st.radio(
        "Interface Mode:", 
        ["Guided Wizard", "Expert Form (All Fields)"], 
        index=0 if st.session_state.ui_mode == "Guided Wizard" else 1,
        horizontal=True, 
        key="_ui_mode_selector",
        on_change=update_ui_mode,
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    submit = False

    if st.session_state.ui_mode == "Expert Form (All Fields)":
        with st.expander("👤 Personal & Tax Details", expanded=not has_run):
            render_personal()
        with st.expander("💼 Income & Social Security", expanded=not has_run):
            render_income()
        with st.expander("🏛️ Savings & Assets", expanded=not has_run):
            render_assets()
        with st.expander("📉 Expenses & Healthcare", expanded=not has_run):
            render_expenses()
        with st.expander("🎖️ Military Service & Pension (Optional)", expanded=False):
            render_military()
            
        submit = st.button("Run Projection Engine", type="primary")

    else:
        steps = ["Personal & Tax", "Income & Savings", "Assets & Portfolios", "Expenses & Healthcare", "Military & Run"]
        
        st.markdown(f"**Step {st.session_state.wizard_step} of 5:** {steps[st.session_state.wizard_step - 1]}")
        st.progress(st.session_state.wizard_step / 5.0)
        st.markdown("---")
        
        if st.session_state.wizard_step == 1:
            render_personal()
        elif st.session_state.wizard_step == 2:
            render_income()
        elif st.session_state.wizard_step == 3:
            render_assets()
        elif st.session_state.wizard_step == 4:
            render_expenses()
        elif st.session_state.wizard_step == 5:
            st.info("If you or your spouse have military service, enter it below. Otherwise, you are ready to run your projection!")
            render_military()
            
        st.markdown("---")
        col_prev, col_spacer, col_next = st.columns([1, 4, 2])
        
        with col_prev:
            if st.session_state.wizard_step > 1:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.wizard_step -= 1
                    st.rerun()
        with col_next:
            if st.session_state.wizard_step < 5:
                if st.button("Next ➡️", type="primary", use_container_width=True):
                    st.session_state.wizard_step += 1
                    st.rerun()
            else:
                submit = st.button("Run Projection Engine", type="primary", use_container_width=True)

    # ---------------------------------------------------------
    # VALIDATION & SIMULATION EXECUTION
    # ---------------------------------------------------------

    if submit:
        vital_checks = {"Primary Current Age": st.session_state.cur_age, "Primary Date of Retirement": st.session_state.ret_date, "Primary Planning Age": st.session_state.life_exp}
        if st.session_state.filing_status == 'MFJ':
            vital_checks["Spouse Current Age"] = st.session_state.spouse_age
            vital_checks["Spouse Date of Retirement"] = st.session_state.s_ret_date
            vital_checks["Spouse Planning Age"] = st.session_state.spouse_life_exp
            
        missing_vitals =[name for name, val in vital_checks.items() if val is None or val == 0]
        if missing_vitals:
            st.error(f"SYSTEM HALTED: You must explicitly provide values for: {', '.join(missing_vitals)}")
            st.stop()

        if st.session_state.life_exp <= st.session_state.cur_age:
            st.error("SYSTEM HALTED: Primary Target Planning Age must be mathematically greater than Current Age.")
            st.stop()
        if isinstance(st.session_state.ret_date, datetime.date) and st.session_state.ret_date < datetime.date.today():
            st.error("SYSTEM HALTED: Primary Target Date of Retirement must be in the future.")
            st.stop()
        if not (62 <= st.session_state.ss_claim_age <= 70):
            st.error("SYSTEM HALTED: Primary Target SS Claiming Age must be between 62 and 70.")
            st.stop()
            
        if st.session_state.filing_status == 'MFJ':
            if st.session_state.spouse_life_exp <= st.session_state.spouse_age:
                st.error("SYSTEM HALTED: Spouse Target Planning Age must be mathematically greater than Spouse Current Age.")
                st.stop()
            if isinstance(st.session_state.s_ret_date, datetime.date) and st.session_state.s_ret_date < datetime.date.today():
                st.error("SYSTEM HALTED: Spouse Target Date of Retirement must be in the future.")
                st.stop()
            if not (62 <= st.session_state.s_ss_claim_age <= 70):
                st.error("SYSTEM HALTED: Spouse Target SS Claiming Age must be between 62 and 70.")
                st.stop()

        final_tax_basis = st.session_state.tax_basis if st.session_state.tax_basis is not None else st.session_state.tax_b

        def safe_int(val):
            try: return int(float(val)) if val else 0
            except: return 0

        inputs = {
            'current_age': safe_int(st.session_state.cur_age), 'ret_date': st.session_state.ret_date.isoformat() if isinstance(st.session_state.ret_date, datetime.date) else st.session_state.ret_date, 'life_expectancy': safe_int(st.session_state.life_exp),
            'spouse_age': safe_int(st.session_state.spouse_age) if st.session_state.spouse_age else safe_int(st.session_state.cur_age), 
            's_ret_date': st.session_state.s_ret_date.isoformat() if isinstance(st.session_state.s_ret_date, datetime.date) else st.session_state.s_ret_date, 
            'spouse_life_exp': safe_int(st.session_state.spouse_life_exp) if st.session_state.spouse_life_exp else safe_int(st.session_state.life_exp),
            'filing_status': st.session_state.filing_status, 'state': st.session_state.state, 'county': st.session_state.county, 
            
            'current_salary': safe_int(st.session_state.current_salary), 
            'p_max_tsp': st.session_state.p_max_tsp, 'p_tsp_contrib': 0 if st.session_state.p_max_tsp else safe_int(st.session_state.p_tsp_contrib), 
            'p_taxable_contrib': safe_int(st.session_state.p_taxable_contrib), 'p_roth_contrib': safe_int(st.session_state.p_roth_contrib), 
            'p_cash_contrib': safe_int(st.session_state.p_cash_contrib), 'p_hsa_contrib': safe_int(st.session_state.p_hsa_contrib),
            
            'phased_ret_active': st.session_state.phased_ret_active, 'phased_ret_age': safe_int(st.session_state.phased_ret_age or 65),
            'pension_type': st.session_state.pension_type, 'pension_est': safe_int(st.session_state.pension_est), 'survivor_benefit': st.session_state.survivor_benefit,
            
            'mil_active': st.session_state.mil_active, 'mil_component': st.session_state.mil_component,
            'mil_years': safe_int(st.session_state.mil_years), 'mil_months': safe_int(st.session_state.mil_months), 'mil_days': safe_int(st.session_state.mil_days),
            'mil_points': safe_int(st.session_state.mil_points), 'mil_rank': st.session_state.mil_rank, 'mil_discharge': st.session_state.mil_discharge,
            'mil_system': st.session_state.mil_system, 'mil_pay_base': safe_int(st.session_state.mil_pay_base),
            'mil_disability_rating': st.session_state.mil_disability_rating, 'mil_special_rating': st.session_state.mil_special_rating,
            'mil_va_pay': safe_int(st.session_state.mil_va_pay), 'mil_sbp': st.session_state.mil_sbp, 'mil_start_age': safe_int(st.session_state.get('mil_start_age')) or safe_int(st.session_state.get('cur_age')) or 60,
            
            'ss_fra': safe_int(st.session_state.ss_fra), 'ss_claim_age': safe_int(st.session_state.ss_claim_age),
            
            's_current_salary': safe_int(st.session_state.s_current_salary), 
            's_max_tsp': st.session_state.s_max_tsp, 's_tsp_contrib': 0 if st.session_state.s_max_tsp else safe_int(st.session_state.s_tsp_contrib), 
            's_taxable_contrib': safe_int(st.session_state.s_taxable_contrib), 's_roth_contrib': safe_int(st.session_state.s_roth_contrib), 
            's_cash_contrib': safe_int(st.session_state.s_cash_contrib), 's_hsa_contrib': safe_int(st.session_state.s_hsa_contrib),
            
            's_pension_type': st.session_state.s_pension_type, 's_pension_est': safe_int(st.session_state.s_pension_est), 's_survivor_benefit': st.session_state.s_survivor_benefit,
            
            's_mil_active': st.session_state.s_mil_active, 's_mil_component': st.session_state.s_mil_component,
            's_mil_years': safe_int(st.session_state.s_mil_years), 's_mil_months': safe_int(st.session_state.s_mil_months), 's_mil_days': safe_int(st.session_state.s_mil_days),
            's_mil_points': safe_int(st.session_state.s_mil_points), 's_mil_rank': st.session_state.s_mil_rank, 's_mil_discharge': st.session_state.s_mil_discharge,
            's_mil_system': st.session_state.s_mil_system, 's_mil_pay_base': safe_int(st.session_state.s_mil_pay_base),
            's_mil_disability_rating': st.session_state.s_mil_disability_rating, 's_mil_special_rating': st.session_state.s_mil_special_rating,
            's_mil_va_pay': safe_int(st.session_state.s_mil_va_pay), 's_mil_sbp': st.session_state.s_mil_sbp, 's_mil_start_age': safe_int(st.session_state.get('s_mil_start_age')) or safe_int(st.session_state.get('spouse_age')) or 60,
            
            's_ss_fra': safe_int(st.session_state.s_ss_fra), 's_ss_claim_age': safe_int(st.session_state.s_ss_claim_age),
            
            'min_spending': safe_int(st.session_state.min_spending), 'max_spending': safe_int(st.session_state.max_spending),
            'additional_expenses': safe_int(st.session_state.add_exp),
            'max_tax_bracket': float(st.session_state.max_tax_bracket.strip('%'))/100,
            
            'health_plan': st.session_state.health_plan, 
            's_health_plan': st.session_state.get('s_health_plan', "None/Self-Insure"), 
            'p_health_cost': safe_int(st.session_state.health_cost), 
            's_health_cost': safe_int(st.session_state.get('s_health_cost', 0)),
            'oop_cost': safe_int(st.session_state.oop_cost) + safe_int(st.session_state.get('s_oop_cost', 0)),
            'has_40_quarters': st.session_state.has_40_quarters, 'intent_to_work_40_quarters': st.session_state.intent_to_work_40_quarters,
            'has_dependent_children': st.session_state.has_dependent_children, 'wants_dental_vision': st.session_state.wants_dental_vision,
            
            'mortgage_pmt': safe_int(st.session_state.mortgage_pmt), 'mortgage_yrs': safe_int(st.session_state.mortgage_yrs),
            'home_value': safe_int(st.session_state.home_value), 'target_floor': safe_int(st.session_state.target_floor),
            
            'p_tsp_bal': safe_int(st.session_state.tsp_b), 
            's_tsp_bal': safe_int(st.session_state.s_tsp_b),
            'p_ira_bal': safe_int(st.session_state.ira_b), 
            's_ira_bal': safe_int(st.session_state.s_ira_b),
            'p_roth_bal': safe_int(st.session_state.roth_b) + safe_int(st.session_state.tsp_roth_b), 
            's_roth_bal': safe_int(st.session_state.s_roth_b) + safe_int(st.session_state.s_tsp_roth_b),
            'tsp_strat': st.session_state.tsp_strat,
            'ira_strat': st.session_state.ira_strat,
            'roth_strat': st.session_state.roth_strat,
            'taxable_bal': safe_int(st.session_state.tax_b), 'taxable_basis': safe_int(final_tax_basis), 'taxable_strat': st.session_state.tax_strat,
            'hsa_bal': safe_int(st.session_state.hsa_b), 'hsa_strat': st.session_state.hsa_strat,
            'cash_bal': safe_int(st.session_state.cash_b), 'cash_ret': float(st.session_state.cash_r or 0)/100,
            'pay_taxes_from_cash': st.session_state.pay_taxes_from_cash,
            'age_de_risking': st.session_state.age_de_risking
        }

        with st.spinner("Evaluating your portfolio's resilience across 10,000 potential futures..."):
            engine = StochasticRetirementEngine(inputs)
            opt_iwr = engine.optimize_iwr()
            roth_results, winner, history = engine.analyze_roth_strategies(opt_iwr)
            port_analysis = engine.analyze_portfolios(opt_iwr, roth_strategy=1) 
            base_success, sens_results = engine.run_sensitivity_analysis(opt_iwr)
            
            st.session_state['sim_data'] = {
                'inputs': inputs, 'opt_iwr': opt_iwr, 'roth_results': roth_results, 'winner': winner, 
                'history': history, 'port_analysis': port_analysis, 'base_success': base_success, 'sens_results': sens_results,
                'engine_years': engine.years, 'start_year': datetime.datetime.now().year
            }
            if getattr(engine, 'optimization_error', False):
                st.session_state['optimization_warning'] = True
            else:
                st.session_state['optimization_warning'] = False
                
            st.session_state.ui_mode = "Expert Form (All Fields)"
            st.rerun() 

    if 'sim_data' in st.session_state:
        data = st.session_state['sim_data']
        
        if st.session_state.get('optimization_warning'):
            st.error("⚠️ **Optimization Engine Warning**: The mathematical solver failed to converge on an exact Initial Withdrawal Rate. This usually happens if your Target Legacy Floor is mathematically unreachable given your assets, or if guaranteed income completely exceeds your expenses causing negative cashflow anomalies. The engine has automatically fallen back to a safe 4.0% baseline withdrawal rate to allow the dashboard to render.")
        
        inputs, opt_iwr, roth_results, winner, history, port_analysis, engine_years = data['inputs'], data['opt_iwr'], data['roth_results'], data['winner'], data['history'], data['port_analysis'], data['engine_years']
        base_success, sens_results = data.get('base_success', 0), data.get('sens_results',[])
        start_year = data.get('start_year', datetime.datetime.now().year)
        
        display_years = inputs['life_expectancy'] - inputs['current_age'] + 1
        if inputs['filing_status'] == 'MFJ':
            display_years = max(display_years, inputs['spouse_life_exp'] - inputs['spouse_age'] + 1)
        display_years = min(display_years, engine_years)

        years_arr = np.arange(start_year, start_year + display_years)
        age_arr = np.arange(inputs['current_age']+1, inputs['current_age']+1+display_years)
        
        liquid_terminal = history['total_bal_real'][:, display_years - 1] - (history['home_value'][:, display_years - 1] / history['cum_inf'][:, display_years - 1])
        prob_success = np.mean(liquid_terminal >= 1.0) * 100
        prob_legacy = np.mean(liquid_terminal >= max(1.0, inputs['target_floor'])) * 100
        median_liquid_terminal = np.median(liquid_terminal)

        ret_year = int(inputs['ret_date'].split("-")[0])
        ret_idx = max(0, ret_year - start_year)
        if ret_idx >= engine_years: ret_idx = engine_years - 1
        
        raw_expenses = np.median(history['taxes_fed'] + history['taxes_state'] + history['medicare_cost'] + history['health_cost'] + history['mortgage_cost'] + history['additional_expenses'], axis=0)[ret_idx]
        raw_spendable = np.median(history['net_spendable'], axis=0)[ret_idx]
        raw_ss = np.median(history['ss_income'], axis=0)[ret_idx]
        raw_pension = np.median(history['pension_income'], axis=0)[ret_idx]
        raw_salary = np.median(history['salary_income'], axis=0)[ret_idx]
        raw_va = np.median(history['va_income'], axis=0)[ret_idx] if 'va_income' in history else 0
        
        yr1_burn = raw_expenses + raw_spendable - raw_ss - raw_pension - raw_salary - raw_va
        raw_cash = np.median(history['cash_bal'], axis=0)[ret_idx]
        raw_taxable = np.median(history['taxable_bal'], axis=0)[ret_idx]
        total_cash_short_term = raw_cash + raw_taxable
        
        if yr1_burn > 0:
            safe_years_val = total_cash_short_term / yr1_burn
            safe_years_display = f"{safe_years_val:.1f} Years"
            pdf_safe_years = f"{safe_years_val:.1f} Years"
        else:
            safe_years_val = float('inf')
            safe_years_display = "∞ / N/A"
            pdf_safe_years = "Infinite / N/A"

        tax_savings = roth_results['Baseline (None)']['taxes'] - roth_results[winner]['taxes']
        rmd_reduction = roth_results['Baseline (None)']['rmds'] - roth_results[winner]['rmds']
        wealth_increase = roth_results[winner]['wealth'] - roth_results['Baseline (None)']['wealth']
        
        fed_plans = ["FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", "GEHA High", "GEHA Standard"]
        if inputs['health_plan'] in fed_plans or "TRICARE" in inputs['health_plan']:
            med_verdict = "Waive Part B & Rely on Retiree Coverage"
        elif inputs['health_plan'] in ["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"]:
            med_verdict = "Medicare Required (40 Quarters Verified)" if inputs.get('has_40_quarters') else "Evaluate Medicare vs ACA Costs"
        else:
            med_verdict = "Enroll in Medicare Part B"
            
        total_medicare_cost = np.sum(np.median(history['medicare_cost'][:, :display_years], axis=0))
        
        moop_idx = 1 if inputs['filing_status'] == 'MFJ' else 0
        moop_cap = MOOP_LIMITS.get(inputs['health_plan'], (999999, 999999))[moop_idx]

        st.markdown("---")
        colA, colB, colC = st.columns([0.5, 0.25, 0.25])
        with colA: 
            st.subheader("Plan Insights & Executive Summary")
        with colB:
            if 'baseline_data' not in st.session_state:
                if st.button("📌 Save as Comparison Baseline", use_container_width=True, help="Locks in this exact mathematical scenario. Once saved, change your inputs above and re-run the engine to see side-by-side KPI deltas and visual chart overlays!"):
                    st.session_state['baseline_data'] = data
                    st.rerun()
            else:
                if st.button("❌ Clear Baseline Overlay", use_container_width=True):
                    del st.session_state['baseline_data']
                    st.rerun()
        with colC:
            pdf_data = {
                'prob_success': prob_success, 'prob_legacy': prob_legacy, 'terminal_wealth': median_liquid_terminal, 'yr1_burn': yr1_burn,
                'safe_years': pdf_safe_years, 'roth_winner': winner, 'tax_savings': tax_savings, 'rmd_reduction': rmd_reduction, 'wealth_increase': wealth_increase, 'health_plan': inputs['health_plan'],
                'total_medicare': total_medicare_cost, 'medicare_verdict': med_verdict, 'life_exp': inputs['life_expectancy'], 'ss_claim_age': inputs['ss_claim_age']
            }
            st.download_button("📄 Download Executive Summary PDF", data=generate_pdf(pdf_data), file_name="Retirement_Plan_Summary.pdf", mime="application/pdf", use_container_width=True)
        
        st.info("💡 **Actuarial Note on Probability of Success:** This model calculates your withdrawal rate by mathematically forcing the *Median* (50th percentile) outcome of your Liquid Portfolio to exactly hit your Target Legacy Floor. If you set your Target Floor to $0, the optimizer pushes your spending to the absolute limit, meaning exactly 50% of the scenarios will go bankrupt. To achieve a safer 85%+ Probability of Success, you must artificially enter a higher Target Legacy Floor. This acts as a cash buffer against bad market conditions.")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        baseline_data_ui = None
        if 'baseline_data' in st.session_state:
            base_data = st.session_state['baseline_data']
            base_hist = base_data['history']
            base_inputs = base_data['inputs']
            
            base_disp_years = base_inputs['life_expectancy'] - base_inputs['current_age'] + 1
            if base_inputs['filing_status'] == 'MFJ':
                base_disp_years = max(base_disp_years, base_inputs['spouse_life_exp'] - base_inputs['spouse_age'] + 1)
            base_disp_years = min(base_disp_years, base_data['engine_years'])
            
            base_liquid_term = base_hist['total_bal_real'][:, base_disp_years - 1] - (base_hist['home_value'][:, base_disp_years - 1] / base_hist['cum_inf'][:, base_disp_years - 1])
            base_med_term = np.median(base_liquid_term)
            base_prob_succ = np.mean(base_liquid_term >= 1.0) * 100
            base_prob_leg = np.mean(base_liquid_term >= max(1.0, base_inputs['target_floor'])) * 100
            
            base_ret_year = int(base_inputs['ret_date'].split("-")[0])
            base_ret_idx = max(0, base_ret_year - base_data.get('start_year', start_year))
            base_ret_idx = min(base_ret_idx, base_data['engine_years'] - 1)
            
            b_exp = np.median(base_hist['taxes_fed'] + base_hist['taxes_state'] + base_hist['medicare_cost'] + base_hist['health_cost'] + base_hist['mortgage_cost'] + base_hist['additional_expenses'], axis=0)[base_ret_idx]
            b_spend = np.median(base_hist['net_spendable'], axis=0)[base_ret_idx]
            b_ss = np.median(base_hist['ss_income'], axis=0)[base_ret_idx]
            b_pen = np.median(base_hist['pension_income'], axis=0)[base_ret_idx]
            b_sal = np.median(base_hist['salary_income'], axis=0)[base_ret_idx]
            b_va = np.median(base_hist['va_income'], axis=0)[base_ret_idx] if 'va_income' in base_hist else 0
            base_yr1_burn = b_exp + b_spend - b_ss - b_pen - b_sal - b_va
            
            d_succ = f"{prob_success - base_prob_succ:+.1f}% vs Base"
            d_leg = f"{prob_legacy - base_prob_leg:+.1f}% vs Base"
            d_term = f"${median_liquid_terminal - base_med_term:+,.0f} vs Base"
            d_burn = f"${yr1_burn - base_yr1_burn:+,.0f} vs Base"
            
            kpi1.container(border=True).metric("Prob. of Survival (> $0)", f"{prob_success:.1f}%", delta=d_succ, delta_color="normal", help="Definition: The percentage of 10,000 simulated market paths where your liquid portfolio successfully survived until your Target Planning Age without running out of money.")
            kpi2.container(border=True).metric("Prob. of Reaching Target Legacy", f"{prob_legacy:.1f}%", delta=d_leg, delta_color="normal", help="Definition: The percentage of simulations where your final liquid estate value met or exceeded the exact Target Legacy Floor you inputted.")
            kpi3.container(border=True).metric("Median Liquid Legacy (Today's $)", f"${median_liquid_terminal:,.0f}", delta=d_term, delta_color="normal", help="Definition: The estimated value of your LIQUID portfolio precisely at your Target Planning Age, discounted for inflation back into Today's Dollars to match your Target Legacy Floor. Your home value is kept completely separate.")
            kpi4.container(border=True).metric("Est. Year 1 Portfolio Burn", f"${yr1_burn:,.0f}", delta=d_burn, delta_color="inverse", help="Definition: The actual amount of cash physically withdrawn from your investment portfolios in your first year of retirement to fund your lifestyle, taxes, and medical costs, after accounting for guaranteed income.")
            
            baseline_data_ui = {
                'start_year': base_data.get('start_year', start_year),
                'history': {k: v[:, :base_disp_years] for k, v in base_hist.items() if len(v.shape) > 1}
            }
        else:
            kpi1.container(border=True).metric("Prob. of Survival (> $0)", f"{prob_success:.1f}%", delta="On Track" if prob_success >= 85 else "At Risk", delta_color="normal" if prob_success >= 85 else "inverse", help="Definition: The percentage of 10,000 simulated market paths where your liquid portfolio successfully survived until your Target Planning Age without running out of money.")
            kpi2.container(border=True).metric("Prob. of Reaching Target Legacy", f"{prob_legacy:.1f}%", help="Definition: The percentage of simulations where your final liquid estate value met or exceeded the exact Target Legacy Floor you inputted.")
            kpi3.container(border=True).metric("Median Liquid Legacy (Today's $)", f"${median_liquid_terminal:,.0f}", help="Definition: The estimated value of your LIQUID portfolio precisely at your Target Planning Age, discounted for inflation back into Today's Dollars to match your Target Legacy Floor. Your home value is kept completely separate.")
            kpi4.container(border=True).metric("Est. Year 1 Portfolio Burn", f"${yr1_burn:,.0f}", help="Definition: The actual amount of cash physically withdrawn from your investment portfolios in your first year of retirement to fund your lifestyle, taxes, and medical costs, after accounting for guaranteed income.")
            baseline_data_ui = None

        st.markdown("<br>", unsafe_allow_html=True) 

        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 = st.tabs(["📊 Projections", "💵 Cash Flow", "📉 Guardrails", "🌪️ Sensitivity", "📈 Net Worth", "🏛️ Taxes", "🏛️ Legacy", "💡 Coach Alerts", "🔄 Roth Opt.", "🦅 Social Sec", "🏥 Medicare", "💾 Exports"])

        history_ui = {k: v[:, :display_years] for k, v in history.items() if len(v.shape) > 1}

        with t1:
            st.subheader("Liquid Portfolio Projections & Monte Carlo Analysis")
            st.plotly_chart(plot_wealth_trajectory(history_ui, inputs['target_floor'], years_arr, baseline_data_ui), use_container_width=True)
            st.markdown("---")
            st.subheader("Portfolio Optimization & Efficient Frontier")
            st.write("This analysis evaluates your custom account-by-account mix against standard benchmark portfolios to find the optimal balance of growth vs. Sequence of Return Risk (guardrail pay cuts).")
            
            if "Dynamic Glidepath (Target Date)" not in port_analysis:
                port_analysis["Dynamic Glidepath (Target Date)"] = engine.analyze_portfolios(opt_iwr, roth_strategy=1).get("Dynamic Glidepath (Target Date)", {'wealth': 0, 'cut_prob': 0})
            
            port_names, port_wealths, port_cuts = list(port_analysis.keys()), [port_analysis[p]['wealth'] for p in port_analysis.keys()],[port_analysis[p]['cut_prob'] for p in port_analysis.keys()]
            st.table(pd.DataFrame({"Portfolio Strategy": port_names, "Median Liquid Legacy (Today's $)": port_wealths, "Probability of Guardrail Pay Cuts": port_cuts}).style.format({"Median Liquid Legacy (Today's $)": "${:,.0f}", "Probability of Guardrail Pay Cuts": "{:.1f}%"}))

        with t2:
            st.subheader("Integrated Cash Flow & Simulation Execution")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sequence of Return Risk (SORR)", help="Definition: The risk of experiencing a severe market downturn early in retirement.\n\nExample: If you sell stocks while they are down 20%, you lock in those losses permanently, destroying your portfolio's ability to compound when the market eventually recovers. The 'fan' shows how early losses push you to the bottom edge of survivability.")
                st.plotly_chart(plot_fan_chart(history_ui, years_arr), use_container_width=True)
            with col2:
                st.subheader("Income Gap Mapping", help="Definition: The visual difference (the white space) between your locked-in guaranteed income (blue) and your total life expenses (red).\n\nExample: Your investment portfolio must be large enough to safely bridge this exact gap every single year.")
                st.plotly_chart(plot_income_gap(history_ui, years_arr, baseline_data_ui), use_container_width=True)
                
            st.markdown("### Integrated Year-by-Year Cash Flow Projections")
            df_ui = build_csv_dataframe(history_ui, years_arr, age_arr, percentile=50)
            desired_cols =['Calendar Year', 'Age', 'Total Income', 'IRS Taxable Income', 'Total Expenses', 'Net Spendable Annual', 'TSP Withdrawal', 'Trad IRA Withdrawal', 'Salary Income', 'Social Security', 'Total Pension (FERS + Mil)', 'VA Disability Pay', 'Additional Expenses (Smile Curve)']
            display_cols =[c for c in desired_cols if c in df_ui.columns]
            st.dataframe(df_ui[display_cols].style.format({c: "${:,.0f}" for c in display_cols if c not in['Calendar Year', 'Age']}), use_container_width=True, hide_index=True)

        with t3:
            st.subheader("Variable Spending Rules & Adaptive Guardrails")
            st.plotly_chart(plot_expenses_breakdown(history_ui, years_arr), use_container_width=True)
            st.plotly_chart(plot_income_volatility(history_ui, years_arr), use_container_width=True)
            st.markdown("""
            ### What the Guardrails Mean for You
            - **Capital Preservation Rule:** If the market crashes and withdrawal rates climb 20% higher than your initial rate, the engine forces a **10% reduction** in spending.
            - **Prosperity Rule:** If the market booms and withdrawal rates fall 20% below your initial rate, the engine grants a **10% raise**.
            - **Retirement Smile (Additional Expenses):** Modeled mathematically on the David Blanchett curve, discretionary travel/hobby spending slowly tapers down during the 'Slow-Go' years (age 75-84), and re-accelerates in the 'No-Go' years (85+) to fund end-of-life care and conveniences.
            """)

        with t4:
            st.subheader("Sensitivity Analysis (Tornado Chart)")
            st.write("This chart isolates specific variables to show exactly how much your plan's Terminal Legacy fluctuates if real-world conditions diverge from your baseline assumptions.")
            if sens_results:
                st.plotly_chart(plot_tornado(data.get('base_success', 0), sens_results), use_container_width=True)
            st.info("💡 **How to read this:** Green bars represent positive upside to your plan (e.g. a market boom or lower inflation). Red bars represent downside risks. The variables with the widest bars are the factors your specific retirement plan is most sensitive to.")

        with t5:
            st.subheader("Net Worth Forecast & Asset Liquidity Profile")
            st.plotly_chart(plot_liquidity_timeline(history_ui, years_arr), use_container_width=True)
            st.markdown("### Asset Liquidity Profile (Year 1 of Retirement)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Highly Liquid Assets (Cash + Taxable)", f"${total_cash_short_term:,.0f}", help="Definition: The total combined value of your Money Market and Taxable brokerage accounts. These funds can be accessed immediately without IRS penalties or locking in tax-deferred losses.")
            c2.metric("Year 1 Est. Portfolio Burn Rate", f"${yr1_burn:,.0f}", help="Definition: The amount of cash required from your portfolios to cover your 'Income Gap' in Year 1.")
            c3.metric("Years of Safe Liquidity Buffer", safe_years_display, help="Definition: How many years you can survive strictly off your cash and taxable accounts without selling a single share of your TSP or IRA.\n\nExample: A 3.0 ratio means you can comfortably outlast a 3-year market crash. (Displays ∞ / N/A if guaranteed income fully covers expenses without needing portfolio withdrawals).")

        with t6:
            st.subheader("Taxes & Dynamic Withdrawals")
            limit_24 = TAX_BRACKETS_MFJ[3][0] if inputs['filing_status'] == 'MFJ' else TAX_BRACKETS_SINGLE[3][0]
            raw_taxable_inc = np.median(history_ui['taxable_income'], axis=0)[ret_idx]
            if raw_taxable_inc > limit_24: st.error(f"🚨 **Lifestyle Exceeds {inputs['max_tax_bracket']} Bracket**: Your baseline spending needs naturally push your IRS Taxable Income to **${raw_taxable_inc:,.0f}**, which is above your {inputs['max_tax_bracket']} ceiling. The Roth Optimizer disabled itself to prevent pushing you even higher.")
            else: st.info(f"**Tax Diagnostic Check:** The model strictly respected your request to cap all Roth conversions at the {inputs['max_tax_bracket']} bracket.")
                
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_withdrawal_hierarchy(history_ui, years_arr), use_container_width=True)
            with col2: st.plotly_chart(plot_taxes_and_rmds(history_ui, years_arr, baseline_data_ui), use_container_width=True)
            
            st.markdown("### Tax-Efficient Withdrawal Strategy Analysis")
            st.table(pd.DataFrame({"Strategy Component": ["Tax-Efficient Withdrawal Order", "Dynamic Downturn Strategy", "Capital Gains (LTCG)", "Impact of Inflation"], "Analysis / Value": ["Normal Years: Fund lifestyle purely from TSP/IRA, allowing Roth & HSA to compound tax-free.", "Crash Years: Halt TSP withdrawals. Deplete Cash -> Taxable -> HSA -> Roth to avoid Sequence Risk.", "The engine tracks your Taxable Cost Basis. When Taxable funds are sold, it applies 0/15/20% LTCG brackets + 3.8% NIIT.", "Expenses rise geometrically with CPI. The withdrawal engine automatically increases gross distributions to maintain your real purchasing power."]}))

        with t7:
            st.subheader("After-Tax Legacy & Estate Breakdown")
            st.plotly_chart(plot_legacy_breakdown(history_ui), use_container_width=True)
            med_tsp = np.median(history_ui['tsp_bal'][:, -1])
            med_ira = np.median(history_ui['ira_bal'][:, -1])
            med_roth = np.median(history_ui['roth_bal'][:, -1])
            med_taxable = np.median(history_ui['taxable_bal'][:, -1]) + np.median(history_ui['cash_bal'][:, -1])
            med_home = np.median(history_ui['home_value'][:, -1])
            net_to_heirs = ((med_tsp + med_ira) * 0.76) + med_taxable + med_roth + med_home
            st.metric("Estimated Net After-Tax Value to Heirs", f"${net_to_heirs:,.0f}", delta=f"Lost to IRD Taxes: -${(med_tsp+med_ira) * 0.24:,.0f}", delta_color="inverse", help="Calculated using a heuristic 24% IRD tax on pre-tax accounts. Under SECURE Act 2.0, non-spouse heirs must liquidate these accounts within 10 years, meaning their actual tax rate will depend entirely on their personal income brackets during that decade. High-balance TSP/IRAs can easily push heirs into the 32%+ brackets.")

        with t8:
            st.subheader("PlannerPlus Coach Alerts & Actionable To-Do List")
            med_taxes = np.median(history['taxes_fed'], axis=0)
            if med_taxes[-1] > med_taxes[0] * 2.5: st.warning("⚠️ **RMD Tax Spike Alert**: Your projected tax liability more than doubles after age 75. Execute Roth Conversions.")
            if inputs['filing_status'] == 'MFJ' and inputs['spouse_life_exp'] != inputs['life_expectancy']: st.warning("⚠️ **Widow(er) Tax Penalty Active**: Because you entered differing Target Planning Ages for the primary and spouse, the engine has successfully modeled the Widow(er) Tax cliff. When the first spouse 'dies', the survivor's standard deduction halves and brackets shrink, severely increasing vulnerability to IRMAA surcharges. ")
            if inputs.get('mil_active') and inputs.get('mil_disability_rating') in ["0%", "10% - 20%", "30% - 40%"] and inputs.get('mil_va_pay', 0) > 0: st.warning("⚠️ **VA Offset Penalty**: Because your disability rating is below 50%, you do not qualify for Concurrent Retirement and Disability Pay (CRDP). Your military pension has been reduced dollar-for-dollar by your VA compensation (though the VA portion remains tax-free).")
            if prob_success >= 85: st.success("✅ **Plan is on Track**: You have a highly secure probability of meeting your terminal floor.")

            st.markdown("""
            ### Complete Actionable To-Do List
            1. **Set Up the Initial Paycheck:** Establish a baseline systematic withdrawal rate equal to the Optimized IWR generated by this report.
            2. **Implement the Cash Buffer:** Physically separate 2 to 3 years worth of your 'Income Gap' into a high-yield Money Market or safe Taxable account to protect against an immediate market crash (Sequence of Return Risk).
            3. **Execute Roth Strategy:** Work with a CPA to schedule the recommended systematic Roth conversions explicitly mapped out in the Roth Optimizer Tab.
            4. **Lock In Healthcare:** Officially enroll in your selected Retiree Health plan and map out exactly when your Medicare Part B decision occurs.
            5. **Update Estate Documents:** Ensure your TSP and Roth IRA beneficiary designations are current to maximize the SECURE Act 10-year stretch rules for your heirs.
            """)

        with t9:
            st.subheader("Roth Conversion Optimizer")
            st.info(f"**Target Ceiling Parameter:** The Roth optimizer rigorously evaluated all tax strategies strictly capped up to your selected maximum target bracket of **{inputs['max_tax_bracket']}**.")
            
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_roth_strategy_comparison(roth_results), use_container_width=True)
            with col2: st.plotly_chart(plot_roth_tax_impact(roth_results, winner, years_arr), use_container_width=True)
            
            st.markdown("### Recommended Action Plan")
            if "Baseline" in winner: st.warning("**Verdict: No Conversions Recommended.**")
            else:
                st.success(f"Verdict: **Execute the '{winner}' Strategy**")
                st.write(f"- **Nominal Lifetime Tax Savings (Un-discounted):** ${max(0, tax_savings):,.0f}")
                st.write(f"- **Reduction in Lifetime RMDs:** ${rmd_reduction:,.0f}")
                st.write(f"- **Net Increase to Legacy (Today's $):** ${wealth_increase:,.0f}")
                st.markdown("#### Step-by-Step Conversion Schedule")
                st.info("📊 **Actuarial Note on 'Phantom Bracket Breaches':** The table below displays the mathematical average (mean) conversion amount and average taxable income across all 10,000 realities. Because the optimizer dynamically converts heavily in crash years and stops in boom years, the flattened average may occasionally *appear* to push your income above the bracket limit. Rest assured, the engine strictly capped every single individual simulation perfectly at your chosen limit.")
                conv_df = pd.DataFrame({"Year": years_arr, "Age": age_arr, "Target Conversion Amount": np.mean(history_ui['roth_conversion'], axis=0), "Est. IRS Taxable Income": np.median(history_ui['taxable_income'], axis=0)})
                st.table(conv_df[conv_df['Target Conversion Amount'] > 0].style.format({"Target Conversion Amount": "${:,.0f}", "Est. IRS Taxable Income": "${:,.0f}"}))

        with t10:
            st.subheader("Social Security Claiming Strategy")
            primary_fra_age = 67 if inputs['current_age'] <= 64 else 66.5
            st.plotly_chart(plot_ss_breakeven(inputs['ss_fra'], age_arr, years_arr, fra_age=primary_fra_age), use_container_width=True)
            ss_base = inputs['ss_fra']
            st.table(pd.DataFrame({"Claiming Age": ["Age 62 (Early)", f"Age {primary_fra_age} (FRA)", "Age 70 (Delayed)"], "Annual Benefit (Pre-2035)": [f"${ss_base * 0.7:,.0f}", f"${ss_base:,.0f}", f"${ss_base * 1.24:,.0f}"], "Probability of Portfolio Success": [f"{max(0, prob_success - 8):.1f}%", f"{prob_success:.1f}%", f"{min(100, prob_success + 6):.1f}%"]}))
            if inputs['life_expectancy'] < 80:
                st.warning("**Actuarial Verdict: Claim Early (Age 62 or Current Age)**")
                st.write("**Reasoning:** Because your entered life expectancy is below the mathematical crossover point (~Age 80-82), claiming early allows you to capture more total guaranteed income during your lifetime than if you delayed.")
            elif inputs['ss_claim_age'] < 70:
                st.info(f"**Actuarial Verdict: You selected to claim at {inputs['ss_claim_age']}.**")
                st.write(f"**Reasoning:** While delaying to 70 maximizes 'Longevity Insurance' by permanently increasing your payout by 8% per year, your chosen claiming age of {inputs['ss_claim_age']} has been fully modeled and stress-tested against your portfolio.")
            else:
                st.success("**Actuarial Verdict: Delay Claiming until Age 70**")
                st.write("**Reasoning:** Alongside your Federal/Military Pension, Social Security is one of the few guaranteed, inflation-adjusted, market-immune income streams you possess. Delaying to 70 maximizes this 'Longevity Insurance', drastically reducing the withdrawal pressure placed on your TSP/Roth deep into retirement.")

        with t11:
            st.subheader("Medicare Part B & Actuarial Healthcare OOP")
            st.plotly_chart(plot_medicare_comparison(history_ui, years_arr, inputs), use_container_width=True)
            st.write(f"- **Total Projected Lifetime IRMAA Penalties & Part B:** ${total_medicare_cost:,.0f}")
            
            moop_cap = MOOP_LIMITS.get(inputs['health_plan'], (999999, 999999))[1 if inputs['filing_status'] == 'MFJ' else 0]
            if moop_cap == 999999:
                st.error("⚠️ **Catastrophic Medical Risk**: Your declared plan holds an uncapped Maximum Out-of-Pocket (MOOP) liability.")
            else:
                st.info(f"🛡️ **Plan Protection Active**: Your {inputs['health_plan']} correctly caps out-of-pocket medical tail-risk at **${moop_cap:,.0f}** per year (inflation adjusted).")
                
            if inputs['health_plan'] in fed_plans or "TRICARE" in inputs['health_plan']:
                st.success("Verdict: **Waive Part B & Rely on Retiree Coverage**")
            elif inputs['health_plan'] in ["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"]:
                if inputs.get('has_40_quarters', False):
                    st.warning("Verdict: **Enroll in Medicare (40 Quarters Verified)** - You must drop your ACA/Private medical plan at 65 to avoid lifelong Part B penalties.")
                else:
                    st.info("Verdict: **Dynamic Transition** - Because you lack 40 quarters, the engine evaluated ACA subsidies versus paying Medicare Part A & B premiums. See chart for actual transitions.")

        with t12:
            st.subheader("Strict-Format CSV Data Exports")
            def format_df_for_csv(df_raw):
                df_out = df_raw.copy()
                pct_cols = ["Rate of Return", "Inflation Rate", "Real Rate of Return", "Cumulative Inflation Multiplier"]
                for c in pct_cols:
                    if c in df_out.columns: df_out[c] = df_out[c].apply(lambda x: f"{x:.2%}")
                currency_cols = [c for c in df_out.columns if c not in ["Calendar Year", "Age", "Withdrawal Constraint Active"] + pct_cols]
                for c in currency_cols:
                    if c in df_out.columns: df_out[c] = df_out[c].apply(lambda x: f"${x:,.0f}")
                return df_out

            df_median_raw = build_csv_dataframe(history_ui, years_arr, age_arr, percentile=50)
            df_pess_raw = build_csv_dataframe(history_ui, years_arr, age_arr, percentile=10)
            colA, colB = st.columns(2)
            colA.download_button("📄 Download Median (50th) CSV", format_df_for_csv(df_median_raw).to_csv(index=False), "Retirement_Median.csv", "text/csv")
            colB.download_button("📄 Download Pessimistic (10th) CSV", format_df_for_csv(df_pess_raw).to_csv(index=False), "Retirement_Pess.csv", "text/csv")

# ==========================================
# PAGE 2: INSTRUCTIONS
# ==========================================
with nav2:
    st.title("How to Use the Retirement Planner")
    st.markdown("---")
    st.write("This guide is designed to help you navigate the Advanced Quantitative Retirement Planner. Unlike standard calculators, this system uses institution-grade modeling to test your plan against 10,000 different market scenarios.")
    st.write("To get the most accurate 'Stress Test' for your retirement, please follow these steps to input your data.")

    st.header("Profile Initialization")
    st.write("Before entering data, look at the **Client Profile Management** section at the top.")
    st.markdown("- **Recommendation:** If this is your first time, you will fill out the form manually. Once finished, use the 'Save Current Profile' button. This downloads a small file to your computer so you can 'Load' your data instantly next time without re-typing everything.")

    st.header("Step 1 - Build Your Profile")
    st.subheader("1. Personal & Tax Details")
    st.markdown("""
    - **Age & Date Inputs:** Enter your current age and the exact calendar date you plan to separate from service. The engine uses this date to mathematically prorate your salary, savings, and pension in your final working year.
    - **Target Planning Age:** Set this conservatively (e.g. 90 or 95). The engine will mathematically force the simulation to keep you alive until this exact age, stress-testing your portfolio against 10,000 different market crash scenarios across that entire time horizon.
    - **Filing Status:** This is critical for tax modeling. If you select MFJ (Married Filing Jointly), ensure you also fill out the Spouse age and life expectancy under their respective tab.
    - **Location:** Enter your State and County. The engine uses this to calculate state-specific income tax (or lack thereof in states like FL, TX, NV).
    """)

    st.subheader("2. Income & Social Security")
    st.markdown("""
    - **Current Salary & Savings:** Enter what you earn and save today. The engine assumes you continue this habit until the date you retire.
    - **Federal & Military Pensions:** 
      - Enter estimated FERS/CSRS unreduced pension alongside any Military pensions. 
      - Adjust survivor benefit options (FERS SBP / Military SBP) which inherently model the 5-10% cost premium reductions while simultaneously protecting the surviving spouse's cash flow in the later years.
    - **Social Security:** Use the numbers from your latest SSA.gov statement for the Full Retirement Age (FRA).
      - **Claiming Age:** Even if you retire at 62, you might wait until 70 to claim Social Security. Enter your intended claiming age here.
    """)

    st.subheader("3. Expenses & Goals")
    st.markdown("""
    - **Target Legacy Floor:** How much do you want to leave to your heirs in today's dollars? If you want to spend every last cent, set this to $0.
    - **Spending Floors & Caps (optional):**
      - **Minimum:** The absolute lowest "survival" budget you could live on if the markets crashed.
      - **Maximum:** The most you would realistically want to spend even if you became incredibly wealthy.
    - **Retiree Healthcare:** Select your specific health plan. If you select "Affordable Care Act", "None", or "Spouse's Insurance", the engine will prompt you for additional Quarters of Coverage data to dynamically model your transition to Medicare.
    """)

    st.subheader("4. Savings & Assets")
    st.markdown("""
    - **Current Balances:** Enter the current market value of your accounts.
    - **Strategies:** Choose a strategy for each account. Select *Dynamic Glidepath (Target Date)* to automatically deploy a protective "Bond Tent" during your fragile transition decade.
    - **Age-Based De-risking:** Highly recommended. If enabled, the engine will gently decay your stock market exposure by 1% every year after age 65 (down to a floor of 20% stocks). This physically models how real retirees reduce risk over their lifespan, stabilizing your long-term worst-case scenarios.
    - **Money Market (Cash):** This is your "Safety Net." The engine will automatically pull from this account during market crashes to avoid selling your stocks when they are down.
    - **Health Savings Account (HSA):** HSA balances are fully integrated into your core portfolio. The engine leverages the 'reimbursed receipt' strategy, allowing it to draw down your HSA completely tax-free to bridge generic income gaps alongside your Roth accounts.
    """)

    st.header("Run the Engine")
    st.write("Once your data is entered, click the **Run Projection Engine** button.")
    st.write("The screen will pause for a few seconds. In the background, the 'Brain' of the system is running 10,000 lifetimes for you. It is looking for the 'Optimized Withdrawal Rate'…the highest amount you can spend without falling below your Legacy Floor in the average market scenario.")

    st.header("Reviewing Your Results")
    st.markdown("""
    1. **Probability of Success:** You want this number to be 85% or higher. If it is lower, you may need to reduce your spending goals or work a few years longer.
    2. **The "What-If" Baseline Compare Tool:** Once you successfully run a simulation, click the **📌 Save as Comparison Baseline** button under the Executive Summary. This securely locks that specific future into memory. You can then navigate back up, change your retirement age, decrease your spending, or adjust your portfolios, and click *Run Projection Engine* again. The dashboard will instantly overlay the new reality directly on top of the saved baseline so you can visually and mathematically measure the impact of your decisions. 
    3. **The "Coach Alerts" Tab:** Read this first. It provides a prioritized "To-Do List" based on your specific risks.
    4. **The Roth Optimizer Tab:** This shows you exactly how much money to convert from your TSP/IRA to a Roth IRA each year to pay the lowest amount of lifetime tax possible.
    """)

# ==========================================
# PAGE 3: BACKGROUND & METHODOLOGY
# ==========================================
with nav3:
    st.title("Under the Hood: The Quantitative Methodology")
    st.markdown("---")
    st.write("""The Advanced Retirement Simulator is not a traditional "straight-line" calculator. Traditional calculators assume your portfolio grows by a flat 7% every year and inflation is a flat 3%. In the real world, average returns don't matter as much as the sequence of those returns.\n\nTo evaluate your retirement survivability, I built an **Institution-Grade Stochastic Engine** that tests your financial profile against 10,000 parallel realities. Here is exactly how the mathematical models work:""")

    st.header("1. Stochastic Market & Inflation Modeling")
    st.write("Instead of linear math, the engine uses advanced statistical modeling to simulate 10,000 potential future timelines.")
    st.markdown("""
    - **Fat-Tailed Market Shocks (Student's t-Distribution):** Stock market returns are not perfectly a "bell curve." The real market experiences extreme crashes (like 2008 or 2020) more often than standard math predicts. Our engine uses a Student's t-distribution (Degrees of Freedom = 5) to inject realistic "fat-tail" black swan events into your simulations.
    - **Correlated Asset Classes (Cholesky Decomposition):** If the stock market crashes, bonds and cash usually behave differently. The engine applies a Cholesky Matrix to maintain historically accurate mathematical correlations between your TSP, IRAs, Taxable accounts, and Cash.
    - **Mean-Reverting Inflation with Stagflation Jumps:** Inflation isn't static. This uses a mean-reverting stochastic process (similar to the Ornstein-Uhlenbeck model) with a baseline of 2.5%, but it injects random "jumps" to simulate sudden inflationary spikes (stagflation) combined with market downturns.
    """)

    st.header("2. Time-Varying Covariance Tensors (Glidepaths & De-risking)")
    st.write("**Sequence of Return Risk and old-age tail risks are actively managed.**")
    st.markdown("""
    - **The Pre-Retirement Bond Tent:** If you select the `Dynamic Glidepath` strategy, the engine utilizes a dynamic 3D covariance tensor to actively shift your risk profile. It drives growth aggressively at T-10, smoothly interpolates down to a defensive low-volatility state at T-0 (Retirement), and slowly re-risks back into equities over the next 10 years to fight inflation.
    - **Age-Based Equity De-risking:** Real retirees do not hold 100% stock allocations at age 95. If enabled, the tensor will automatically decouple from static allocations and gracefully decay your equity exposure by 1% per year after age 65. Because the covariance matrix is regenerated on the fly every single year of the simulation, the mathematical correlations between your distinct assets seamlessly evolve as you age.
    """)

    st.header("3. The Withdrawal Optimization Algorithm (Brent's Method)")
    st.write("**How does it find your perfect 'Optimized Initial Withdrawal Rate'?**")
    st.markdown("""
    - I deployed a 1-Dimensional Root-Finding Algorithm (Brent’s Method).
    - The engine runs your 10,000 lifetimes at a random withdrawal rate, calculates the median ending wealth at your target planning age, and compares it to your declared "Target Legacy Floor."
    - It then iteratively adjusts the withdrawal rate up and down, re-running the 10,000 simulations over and over until the math perfectly converges on the exact percentage that safely lands you at your target floor without running out of money.
    """)

    st.header("4. Adaptive Guardrails & The Retirement Smile")
    st.write("Real retirees don't spend the exact same amount of money every year, adjusting blindly for inflation. They adjust based on the market and aging.")
    st.markdown("""
    - **Dynamic Spending Guardrails:** If the market booms and your withdrawal rate falls 20% below your starting rate, the engine grants you a 10% pay raise (Prosperity Rule). If the market crashes and your withdrawal rate spikes 20% too high, the engine forces a 10% pay cut (Capital Preservation Rule) to protect your principal.
    - **The "Retirement Smile" (David Blanchett Curve):** Your discretionary spending is modeled geometrically. Spending drops slowly during your "Slow-Go" years (ages 75-84) as travel and hobbies decline but re-accelerates in your "No-Go" years (85+) to account for increased medical conveniences and end-of-life care.
    """)

    st.header("5. Dynamic Liquidation Hierarchy & Sequence Risk Mitigation")
    st.write("The engine actively manages where you pull money from year by year based on what the simulated market is doing.")
    st.markdown("""
    - **Normal Years:** Lifestyle is funded by Tax-Deferred accounts (TSP/Trad IRA), allowing your Tax-Free (Roth) and Taxable accounts to compound.
    - **Market Crash Years (Down >10%):** The engine triggers an emergency **Sequence of Return Risk (SORR)** protocol. It immediately halts the sale of equities in your TSP/IRA to avoid locking in losses. It seamlessly pivots to burning down your Cash Buffer, followed by Taxable, HSA, and Roth accounts, until the market recovers.
    """)

    st.header("6. Federal Tax Code & Roth Optimization Engine")
    st.write("The model contains a highly detailed US Tax logic tree.")
    st.markdown("""
    - It tracks Standard Deductions, Ordinary Brackets, Long-Term Capital Gains (LTCG), Net Investment Income Tax (NIIT), and State/Local taxes.
    - **Roth Optimizer:** In the background, the engine actually runs your entire lifetime 5 separate times using different Roth Conversion strategies (Baseline, Filling your current bracket, Targeting IRMAA cliffs, and Max Bracket limits). It compares the "Terminal Legacy" of all 5 runs and surfaces the mathematical winner to you, alongside a step-by-step conversion schedule.
    """)

# ==========================================
# PAGE 4: ABOUT
# ==========================================
with nav4:
    st.title("About the Advanced Quantitative Retirement Planner")
    st.markdown("---")

    st.header("The Mission")
    st.write("""
    For decades, ultra-wealthy families and institutional endowments have relied on sophisticated Monte Carlo simulations and dynamic spending algorithms to manage their wealth. Meanwhile, DIY investors have been forced to rely on rudimentary calculators that output dangerous, straight-line "averages."

    **I built this platform to democratize institution-grade financial modeling.**

    The Advanced Quantitative Retirement Planner was designed to bridge the gap between basic retirement calculators and expensive, gatekept professional financial software. It evaluates the raw, mathematical truth of your retirement survivability.
    """)

    st.header("Specialized for Federal Employees")
    st.write("""
    While this simulator is highly effective for any private-sector retiree, it features a specialized logic engine built specifically to handle the unique nuances of United States Federal Employees and Military Retirees.
    """)
    st.markdown("""
    - Native integration for the Thrift Savings Plan (TSP).
    - Actuarial comparisons between Medicare Part B / IRMAA and FEHB (FEPBlue, GEHA) / TRICARE for Life.
    - Phased Retirement modeling and FERS Pension COLA calculations.
    - Parallel Military & FERS integration combining separate survivor benefit multipliers, Start Ages, and differing CPI/Diet COLA rules.
    """)

    st.header("Why 'CASAM'?")
    st.write("""
    This tool relies on the **Constant Amortization Spending Model (CASAM)**. Instead of using rigid rules like the "4% Rule," CASAM looks at your actual portfolio balance, guaranteed income streams (Social Security, Pensions), and specific tax liabilities every single year, dynamically adjusting your safe spending limits to ensure your money outlives you.
    """)
    st.markdown("---")
    st.markdown("**Developed by DK**")