import numpy as np
import scipy.optimize as optimize
from scipy.stats import t
import datetime
from config import *
import gc

# Simplified SSA Period Life Table (Unisex Actuarial Probabilities of Death: q_x)
SSA_MORTALITY = {
    **{a: 0.001 for a in range(0, 30)},
    **{a: 0.0015 for a in range(30, 40)},
    **{a: 0.0025 for a in range(40, 50)},
    **{a: 0.005 for a in range(50, 60)},
    60: 0.007, 61: 0.008, 62: 0.009, 63: 0.010, 64: 0.011,
    65: 0.013, 66: 0.014, 67: 0.016, 68: 0.017, 69: 0.019,
    70: 0.021, 71: 0.024, 72: 0.027, 73: 0.030, 74: 0.034,
    75: 0.038, 76: 0.043, 77: 0.048, 78: 0.054, 79: 0.061,
    80: 0.069, 81: 0.077, 82: 0.087, 83: 0.098, 84: 0.111,
    85: 0.125, 86: 0.141, 87: 0.158, 88: 0.177, 89: 0.198,
    90: 0.222, 91: 0.247, 92: 0.274, 93: 0.303, 94: 0.334,
    95: 0.365, 96: 0.398, 97: 0.431, 98: 0.466, 99: 0.500,
    100: 0.535, 101: 0.570, 102: 0.605, 103: 0.640, 104: 0.675,
    105: 0.710, 106: 0.745, 107: 0.780, 108: 0.815, 109: 0.850,
    110: 0.885, 111: 0.920, 112: 0.955, 113: 0.990
}
for a in range(114, 121):
    SSA_MORTALITY[a] = 1.0

class StochasticRetirementEngine:
    def __init__(self, inputs):
        self.inputs = inputs
        self.iterations = 10000
        self.optimization_failed = False
        
        self.ret_year = int(self.inputs['ret_date'].split("-")[0])
        self.ret_month = int(self.inputs['ret_date'].split("-")[1])
        self.s_ret_year = int(self.inputs['s_ret_date'].split("-")[0])
        self.s_ret_month = int(self.inputs['s_ret_date'].split("-")[1])

        base_years = inputs['life_expectancy'] - inputs['current_age']
        if inputs['filing_status'] == 'MFJ':
            spouse_years = inputs['spouse_life_exp'] - inputs['spouse_age']
            self.years = max(1, max(base_years, spouse_years))
        else:
            self.years = max(1, base_years)
            
        self.n_assets = 6 

    def get_yr_port_params(self, asset_key, yr, override_port=None):
        strat = override_port if override_port else self.inputs[asset_key]
        if strat == "Dynamic Glidepath (Target Date)":
            current_sim_year = datetime.datetime.now().year + yr
            yrs_to_ret = self.ret_year - current_sim_year
            
            agg_r, agg_v = 0.080, 0.150
            cons_r, cons_v = 0.045, 0.060
            mod_r, mod_v = 0.065, 0.100
            
            if yrs_to_ret >= 10:
                base_E = 1.0
            elif yrs_to_ret > 0:
                base_E = 0.20 + (yrs_to_ret / 10.0) * 0.80
            elif yrs_to_ret > -10:
                base_E = 0.20 + (abs(yrs_to_ret) / 10.0) * 0.40
            else:
                base_E = 0.60
        else:
            if "Aggressive" in strat:
                base_E = 1.0
            elif "Moderate" in strat:
                base_E = 0.60
            elif "Conservative" in strat:
                base_E = 0.20
            else:
                base_E = 0.60
                
        if self.inputs.get('age_de_risking', False):
            current_age = self.inputs['current_age'] + yr
            if current_age > 65:
                years_past_65 = current_age - 65
                drop = years_past_65 * 0.01
                base_E = max(0.20, base_E - drop)
                
        if base_E <= 0.60:
            pct = (base_E - 0.20) / 0.40
            ret = 0.045 + pct * (0.070 - 0.045)
            vol = 0.060 + pct * (0.100 - 0.060)
        else:
            pct = (base_E - 0.60) / 0.40
            ret = 0.070 + pct * (0.095 - 0.070)
            vol = 0.100 + pct * (0.150 - 0.100)
            
        return ret, vol

    def get_covariance_and_drifts(self, yr, override_port=None):
        params =[
            self.get_yr_port_params('tsp_strat', yr, override_port),
            self.get_yr_port_params('ira_strat', yr, override_port),
            self.get_yr_port_params('roth_strat', yr, override_port),
            self.get_yr_port_params('taxable_strat', yr, override_port),
            self.get_yr_port_params('hsa_strat', yr, override_port)
        ]
        
        corr = np.array([[1.00, -0.15, -0.15, -0.15, -0.15, -0.15],[-0.15, 1.00,  0.85,  0.85,  0.85,  0.85],[-0.15, 0.85,  1.00,  0.85,  0.85,  0.85],[-0.15, 0.85,  0.85,  1.00,  0.85,  0.85],[-0.15, 0.85,  0.85,  0.85,  1.00,  0.85],[-0.15, 0.85,  0.85,  0.85,  0.85,  1.00]
        ])
        
        vols = np.array([0.020, params[0][1], params[1][1], params[2][1], params[3][1], params[4][1]])
        cov = np.outer(vols, vols) * corr
        L = np.linalg.cholesky(cov)
        
        drifts = np.array([
            0.03, 
            params[0][0] - (params[0][1]**2)/2, 
            params[1][0] - (params[1][1]**2)/2, 
            params[2][0] - (params[2][1]**2)/2, 
            params[3][0] - (params[3][1]**2)/2, 
            params[4][0] - (params[4][1]**2)/2
        ])
        return L, drifts

    def generate_stochastic_paths(self, seed=None, override_port=None, sensitivity_mode=None):
        if seed is not None:
            np.random.seed(seed)
            
        shocks = t.rvs(df=5, size=(self.iterations, self.years, self.n_assets)) * np.sqrt(3.0 / 5.0)
        
        returns = np.zeros((self.iterations, self.years, self.n_assets))
        inf_paths = np.zeros((self.iterations, self.years))
        
        inf_base = 0.021
        if sensitivity_mode == 'inf_up': inf_base += 0.01
        elif sensitivity_mode == 'inf_down': inf_base -= 0.01
            
        kappa = 0.25
        jump_prob = 0.04
        dt = 1.0            
        current_inf = np.full(self.iterations, inf_base)
        
        for yr in range(self.years):
            L_yr, drifts_yr = self.get_covariance_and_drifts(yr, override_port)
            
            if sensitivity_mode == 'market_up': drifts_yr += 0.01
            elif sensitivity_mode == 'market_down': drifts_yr -= 0.01
                
            corr_shocks_yr = np.einsum('ij,kj->ki', L_yr, shocks[:, yr, :])
            
            dW = corr_shocks_yr[:, 0] * np.sqrt(dt)
            jumps = np.where(np.random.rand(self.iterations) < jump_prob, np.random.uniform(0.03, 0.05, self.iterations), 0)
            current_inf = current_inf + kappa * (inf_base - current_inf) * dt + dW + jumps
            inf_paths[:, yr] = np.clip(current_inf, -0.01, 0.15) 
            stagflation_shock = np.where(inf_paths[:, yr] > 0.05, -1.5 * (inf_paths[:, yr] - 0.05), 0)
            
            returns[:, yr, :] = np.exp(drifts_yr + corr_shocks_yr) - 1
            returns[:, yr, 1:] += stagflation_shock[:, None]
            
        return returns, inf_paths

    def calc_tax_vectorized(self, income, brackets, cum_inf):
        tax = np.zeros(self.iterations)
        for i in range(len(brackets)):
            if i > 0:
                prev_limit = brackets[i-1][0] * cum_inf
            else:
                prev_limit = np.zeros(self.iterations)
                
            limit = brackets[i][0] * cum_inf
            in_bracket = np.clip(income, prev_limit, limit) - prev_limit
            tax += np.maximum(0, in_bracket) * brackets[i][1]
        return tax

    def run_mc(self, iwr, seed=None, roth_strategy=0, override_port=None, sensitivity_mode=None):
        returns, inf_paths = self.generate_stochastic_paths(seed=seed, override_port=override_port, sensitivity_mode=sensitivity_mode)
        cash_ret = float(self.inputs.get('cash_ret', 0.04))
        
        p_death_ages = np.full(self.iterations, self.inputs['life_expectancy'])
        if self.inputs['filing_status'] == 'MFJ':
            s_death_ages = np.full(self.iterations, self.inputs.get('spouse_life_exp', self.inputs['life_expectancy']))
        else:
            s_death_ages = np.zeros(self.iterations, dtype=int)
            
        history = {
            'total_bal': np.zeros((self.iterations, self.years)),
            'total_bal_real': np.zeros((self.iterations, self.years)),
            'cum_inf': np.zeros((self.iterations, self.years)),
            'tsp_bal': np.zeros((self.iterations, self.years)),
            'ira_bal': np.zeros((self.iterations, self.years)),
            'roth_bal': np.zeros((self.iterations, self.years)),
            'taxable_bal': np.zeros((self.iterations, self.years)),
            'cash_bal': np.zeros((self.iterations, self.years)),
            'hsa_bal': np.zeros((self.iterations, self.years)),
            'home_value': np.zeros((self.iterations, self.years)),
            'tsp_withdrawal': np.zeros((self.iterations, self.years)),
            'ira_withdrawal': np.zeros((self.iterations, self.years)),
            'roth_withdrawal': np.zeros((self.iterations, self.years)),
            'hsa_withdrawal': np.zeros((self.iterations, self.years)),
            'taxable_withdrawal': np.zeros((self.iterations, self.years)),
            'cash_withdrawal': np.zeros((self.iterations, self.years)),
            'rmds': np.zeros((self.iterations, self.years)),
            'extra_rmd': np.zeros((self.iterations, self.years)),
            'taxes_fed': np.zeros((self.iterations, self.years)),
            'taxes_state': np.zeros((self.iterations, self.years)),
            'taxable_income': np.zeros((self.iterations, self.years)),
            'magi': np.zeros((self.iterations, self.years)),
            'medicare_cost': np.zeros((self.iterations, self.years)),
            'health_cost': np.zeros((self.iterations, self.years)),
            'mortgage_cost': np.zeros((self.iterations, self.years)),
            'additional_expenses': np.zeros((self.iterations, self.years)),
            'net_spendable': np.zeros((self.iterations, self.years)),
            'salary_income': np.zeros((self.iterations, self.years)),
            'port_return': np.zeros((self.iterations, self.years)),
            'real_return': np.zeros((self.iterations, self.years)),
            'inflation': inf_paths,
            'constraint_active': np.zeros((self.iterations, self.years)),
            'ss_income': np.zeros((self.iterations, self.years)),
            'pension_income': np.zeros((self.iterations, self.years)),
            'va_income': np.zeros((self.iterations, self.years)),
            'roth_conversion': np.zeros((self.iterations, self.years)),
            'roth_taxes_from_cash': np.zeros((self.iterations, self.years)),
            'income_gap': np.zeros((self.iterations, self.years)),
            'guaranteed_income': np.zeros((self.iterations, self.years)),
            'tax_paid': np.zeros((self.iterations, self.years)),
            'terminal_year': np.zeros(self.iterations, dtype=int)
        }

        p_tsp = np.full(self.iterations, float(self.inputs.get('p_tsp_bal', 0.0)))
        s_tsp = np.full(self.iterations, float(self.inputs.get('s_tsp_bal', 0.0)))
        p_ira = np.full(self.iterations, float(self.inputs.get('p_ira_bal', 0.0)))
        s_ira = np.full(self.iterations, float(self.inputs.get('s_ira_bal', 0.0)))
        p_roth = np.full(self.iterations, float(self.inputs.get('p_roth_bal', 0.0)))
        s_roth = np.full(self.iterations, float(self.inputs.get('s_roth_bal', 0.0)))

        taxable = np.full(self.iterations, float(self.inputs['taxable_bal']))
        hsa = np.full(self.iterations, float(self.inputs['hsa_bal']))
        cash = np.full(self.iterations, float(self.inputs['cash_bal']))
        home_value = np.full(self.iterations, float(self.inputs.get('home_value', 0.0)))
        taxable_basis = np.full(self.iterations, float(self.inputs.get('taxable_basis', self.inputs['taxable_bal'])))
        
        p_base_salary = float(self.inputs.get('current_salary', 0))
        s_base_salary = float(self.inputs.get('s_current_salary', 0))
        p_tsp_c = float(self.inputs.get('p_tsp_contrib', 0))
        p_tax_c = float(self.inputs.get('p_taxable_contrib', 0))
        p_roth_c = float(self.inputs.get('p_roth_contrib', 0))
        p_cash_c = float(self.inputs.get('p_cash_contrib', 0))
        p_hsa_c = float(self.inputs.get('p_hsa_contrib', 0))

        s_tsp_c = float(self.inputs.get('s_tsp_contrib', 0))
        s_tax_c = float(self.inputs.get('s_taxable_contrib', 0))
        s_roth_c = float(self.inputs.get('s_roth_contrib', 0))
        s_cash_c = float(self.inputs.get('s_cash_contrib', 0))
        s_hsa_c = float(self.inputs.get('s_hsa_contrib', 0))

        p_base_pension = float(self.inputs.get('pension_est', 0))
        s_base_pension = float(self.inputs.get('s_pension_est', 0))
        
        p_ss_claim = self.inputs.get('ss_claim_age', 67)
        p_months_early = max(0, (67 - p_ss_claim) * 12)
        p_months_late = max(0, (p_ss_claim - 67) * 12)
        p_ss_modifier = 1.0 - ((min(36, p_months_early) * (5/900)) + (max(0, p_months_early - 36) * (5/1200))) + (p_months_late * (8/1200))
        p_spousal_modifier = 1.0 - ((min(36, p_months_early) * (25/3600)) + (max(0, p_months_early - 36) * (5/1200)))
        p_base_ss = float(self.inputs.get('ss_fra', 0))
        
        s_ss_claim = self.inputs.get('s_ss_claim_age', 67)
        s_months_early = max(0, (67 - s_ss_claim) * 12)
        s_months_late = max(0, (s_ss_claim - 67) * 12)
        s_ss_modifier = 1.0 - ((min(36, s_months_early) * (5/900)) + (max(0, s_months_early - 36) * (5/1200))) + (s_months_late * (8/1200))
        s_spousal_modifier = 1.0 - ((min(36, s_months_early) * (25/3600)) + (max(0, s_months_early - 36) * (5/1200)))
        s_base_ss = float(self.inputs.get('s_ss_fra', 0))
        
        if sensitivity_mode == 'income_up':
            p_base_ss *= 1.10
            s_base_ss *= 1.10
            p_base_pension *= 1.10
            s_base_pension *= 1.10
        elif sensitivity_mode == 'income_down':
            p_base_ss *= 0.90
            s_base_ss *= 0.90
            p_base_pension *= 0.90
            s_base_pension *= 0.90
        
        p_surv_choice = self.inputs.get('survivor_benefit', 'No Survivor Benefit')
        if self.inputs.get('pension_type', 'FERS') == "FERS":
            if p_surv_choice == 'Full Survivor Benefit':
                p_pension_mult = 0.90
                p_fers_survivor_mult = 0.50
            elif p_surv_choice == 'Partial Survivor Benefit':
                p_pension_mult = 0.95
                p_fers_survivor_mult = 0.25
            else:
                p_pension_mult = 1.0
                p_fers_survivor_mult = 0.0
        else:
            if "100%" in p_surv_choice:
                p_pension_mult = 0.85
                p_fers_survivor_mult = 1.0
            elif "50%" in p_surv_choice:
                p_pension_mult = 0.925
                p_fers_survivor_mult = 0.50
            elif "Present Value" in p_surv_choice:
                p_pension_mult = 0.965
                p_fers_survivor_mult = 0.0
            else:
                p_pension_mult = 1.0
                p_fers_survivor_mult = 0.0

        s_surv_choice = self.inputs.get('s_survivor_benefit', 'No Survivor Benefit')
        if self.inputs.get('s_pension_type', 'FERS') == "FERS":
            if s_surv_choice == 'Full Survivor Benefit':
                s_pension_mult = 0.90
                s_fers_survivor_mult = 0.50
            elif s_surv_choice == 'Partial Survivor Benefit':
                s_pension_mult = 0.95
                s_fers_survivor_mult = 0.25
            else:
                s_pension_mult = 1.0
                s_fers_survivor_mult = 0.0
        else:
            if "100%" in s_surv_choice:
                s_pension_mult = 0.85
                s_fers_survivor_mult = 1.0
            elif "50%" in s_surv_choice:
                s_pension_mult = 0.925
                s_fers_survivor_mult = 0.50
            elif "Present Value" in s_surv_choice:
                s_pension_mult = 0.965
                s_fers_survivor_mult = 0.0
            else:
                s_pension_mult = 1.0
                s_fers_survivor_mult = 0.0

        p_mil_active = self.inputs.get('mil_active', False)
        p_base_mil_gross = 0
        p_base_va = 0
        p_crdp = False
        p_mil_sbp = False
        p_mil_start_age = self.inputs.get('mil_start_age', 60)
        
        if p_mil_active:
            if self.inputs['mil_discharge'] not in["Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge"]:
                if self.inputs['mil_component'] in["National Guard / Reserve", "Mixed (Active + Guard/Reserve)"]:
                    eq_years = self.inputs['mil_points'] / 360.0
                else:
                    eq_years = self.inputs['mil_years'] + (self.inputs['mil_months'] / 12.0) + (self.inputs['mil_days'] / 360.0)
                
                sys = self.inputs['mil_system']
                if "BRS" in sys:
                    mult = eq_years * 0.02
                elif "REDUX" in sys:
                    mult = (eq_years * 0.025) - max(0, 30 - eq_years) * 0.01
                else:
                    mult = eq_years * 0.025
                    
                p_base_mil_gross = self.inputs['mil_pay_base'] * mult * 12
                p_mil_sbp = "Full SBP" in self.inputs['mil_sbp']
                p_base_va = self.inputs['mil_va_pay'] * 12
                
                if sensitivity_mode == 'income_up':
                    p_base_mil_gross *= 1.10
                    p_base_va *= 1.10
                elif sensitivity_mode == 'income_down':
                    p_base_mil_gross *= 0.90
                    p_base_va *= 0.90
                
                is_crdp_rating = self.inputs['mil_disability_rating'] in["50% - 60%", "70% - 90%", "100%"]
                is_smc_rating = self.inputs['mil_special_rating'] in["TDIU (Unemployability)", "SMC (Special Monthly Comp)"]
                p_crdp = is_crdp_rating or is_smc_rating

        s_mil_active = self.inputs.get('s_mil_active', False)
        s_base_mil_gross = 0
        s_base_va = 0
        s_crdp = False
        s_mil_sbp = False
        s_mil_start_age = self.inputs.get('s_mil_start_age', 60)
        
        if s_mil_active:
            if self.inputs['s_mil_discharge'] not in["Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge"]:
                if self.inputs['s_mil_component'] in["National Guard / Reserve", "Mixed (Active + Guard/Reserve)"]:
                    s_eq_years = self.inputs['s_mil_points'] / 360.0
                else:
                    s_eq_years = self.inputs['s_mil_years'] + (self.inputs['s_mil_months'] / 12.0) + (self.inputs['s_mil_days'] / 360.0)
                
                s_sys = self.inputs['s_mil_system']
                if "BRS" in s_sys:
                    s_mult = s_eq_years * 0.02
                elif "REDUX" in s_sys:
                    s_mult = (s_eq_years * 0.025) - max(0, 30 - s_eq_years) * 0.01
                else:
                    s_mult = s_eq_years * 0.025
                    
                s_base_mil_gross = self.inputs['s_mil_pay_base'] * s_mult * 12
                s_mil_sbp = "Full SBP" in self.inputs['s_mil_sbp']
                s_base_va = self.inputs['s_mil_va_pay'] * 12
                
                if sensitivity_mode == 'income_up':
                    s_base_mil_gross *= 1.10
                    s_base_va *= 1.10
                elif sensitivity_mode == 'income_down':
                    s_base_mil_gross *= 0.90
                    s_base_va *= 0.90
                
                s_is_crdp_rating = self.inputs['s_mil_disability_rating'] in["50% - 60%", "70% - 90%", "100%"]
                s_is_smc_rating = self.inputs['s_mil_special_rating'] in["TDIU (Unemployability)", "SMC (Special Monthly Comp)"]
                s_crdp = s_is_crdp_rating or s_is_smc_rating

        age = self.inputs['current_age']
        spouse_age = self.inputs.get('spouse_age', age)
        current_year = datetime.datetime.now().year
        
        base_filing_status = self.inputs['filing_status']
        pay_taxes_from_cash = self.inputs.get('pay_taxes_from_cash', True)
        min_spending = float(self.inputs.get('min_spending', 0))
        max_spending = float(self.inputs.get('max_spending', 0))
        base_add_exp = float(self.inputs.get('additional_expenses', 0))
        user_max_bracket = float(self.inputs.get('max_tax_bracket', '0.24'))
        base_oop_cost = float(self.inputs.get('oop_cost', 0))
        health_plan = self.inputs.get('health_plan', "None/Self-Insure")
        mortgage_pmt = float(self.inputs.get('mortgage_pmt', 0))
        mortgage_yrs = int(self.inputs.get('mortgage_yrs', 0))
        has_40_quarters = self.inputs.get('has_40_quarters', True)
        has_dependent_children = self.inputs.get('has_dependent_children', False)
        wants_dental_vision = self.inputs.get('wants_dental_vision', False)

        state_str = self.inputs.get('state', '').strip().upper()
        county_str = self.inputs.get('county', '').strip().upper()
        
        if state_str in NO_INCOME_TAX_STATES:
            state_tax_rate = 0.0
        else:
            state_tax_rate = STATE_TAX_RATES.get(state_str, 0.045)
            
        if county_str != "" and state_str in["MD", "IN", "PA", "OH", "NY"]:
            local_tax_rate = 0.025
        elif county_str != "":
            local_tax_rate = 0.010
        else:
            local_tax_rate = 0.0
            
        combined_state_local_rate = state_tax_rate + local_tax_rate
        
        cum_inf = np.ones(self.iterations)
        med_cpi_cum = np.ones(self.iterations)
        target_lifestyle = np.zeros(self.iterations)
        ref_draw = np.zeros(self.iterations)

        for yr in range(self.years):
            age += 1
            spouse_age += 1
            current_year += 1
            
            p_alive = age <= p_death_ages
            
            if base_filing_status == 'MFJ':
                s_alive = spouse_age <= s_death_ages
            else:
                s_alive = np.zeros(self.iterations, dtype=bool)
                
            household_alive = p_alive | s_alive
            
            if base_filing_status == 'MFJ':
                is_mfj = p_alive & s_alive
            else:
                is_mfj = np.zeros(self.iterations, dtype=bool)
                
            deduction = np.where(is_mfj, STD_DED_MFJ, STD_DED_SINGLE)
            deduction += np.where(is_mfj & (age >= 65), EXTRA_DED_65_MFJ_PER_PERSON, 0)
            deduction += np.where(is_mfj & (spouse_age >= 65), EXTRA_DED_65_MFJ_PER_PERSON, 0)
            deduction += np.where(~is_mfj & p_alive & (age >= 65), EXTRA_DED_65_SINGLE, 0)
            deduction += np.where(~is_mfj & ~p_alive & s_alive & (spouse_age >= 65), EXTRA_DED_65_SINGLE, 0)

            if yr > 0:
                cum_inf *= (1 + np.maximum(0, inf_paths[:, yr]))
                
            history['cum_inf'][:, yr] = cum_inf
            
            limit_max_pct = np.full(self.iterations, np.inf)
            for i in range(len(TAX_BRACKETS_MFJ)):
                if np.isclose(TAX_BRACKETS_MFJ[i][1], user_max_bracket, atol=0.01):
                    limit_max_pct = TAX_BRACKETS_MFJ[i][0] * cum_inf
                    break
            
            home_value = np.where(household_alive, home_value * 1.03, home_value)

            inf_floor = np.maximum(0, inf_paths[:, yr])
            
            if self.inputs.get('pension_type', 'FERS') == 'FERS':
                p_cola = np.where(inf_floor <= 0.02, inf_floor, np.where(inf_floor <= 0.03, 0.02, inf_floor - 0.01))
            else:
                p_cola = np.minimum(inf_floor, 0.03)
            p_cola = np.where(age >= 62, p_cola, 0.0) 
            
            if self.inputs.get('s_pension_type', 'FERS') == 'FERS':
                s_cola = np.where(inf_floor <= 0.02, inf_floor, np.where(inf_floor <= 0.03, 0.02, inf_floor - 0.01))
            else:
                s_cola = np.minimum(inf_floor, 0.03)
            s_cola = np.where(spouse_age >= 62, s_cola, 0.0)

            p_base_pension *= (1 + p_cola)
            s_base_pension *= (1 + s_cola)
            p_base_mil_gross *= (1 + inf_floor)
            s_base_mil_gross *= (1 + inf_floor)
            p_base_va *= (1 + inf_floor)
            s_base_va *= (1 + inf_floor)
            p_base_ss *= (1 + inf_floor)
            s_base_ss *= (1 + inf_floor)
            
            p_salary_inf = p_base_salary * cum_inf
            s_salary_inf = s_base_salary * cum_inf
            
            if current_year >= 2035:
                ss_haircut = 0.79
            else:
                ss_haircut = 1.0
            
            if current_year < self.ret_year:
                p_work_frac = 1.0
            elif current_year == self.ret_year:
                p_work_frac = self.ret_month / 12.0
            else:
                p_work_frac = 0.0
                
            p_ret_frac = 1.0 - p_work_frac
            
            if current_year < self.s_ret_year:
                s_work_frac = 1.0
            elif current_year == self.s_ret_year:
                s_work_frac = self.s_ret_month / 12.0
            else:
                s_work_frac = 0.0
                
            s_ret_frac = 1.0 - s_work_frac

            yr_salary = np.where(p_alive, p_salary_inf * p_work_frac, 0) + np.where(s_alive, s_salary_inf * s_work_frac, 0)
            
            p_tsp += np.where(p_alive, p_tsp_c * cum_inf * p_work_frac, 0)
            s_tsp += np.where(s_alive, s_tsp_c * cum_inf * s_work_frac, 0)
            
            p_roth += np.where(p_alive, p_roth_c * cum_inf * p_work_frac, 0)
            s_roth += np.where(s_alive, s_roth_c * cum_inf * s_work_frac, 0)
            
            taxable += np.where(p_alive, p_tax_c * cum_inf * p_work_frac, 0) + np.where(s_alive, s_tax_c * cum_inf * s_work_frac, 0)
            taxable_basis += np.where(p_alive, p_tax_c * cum_inf * p_work_frac, 0) + np.where(s_alive, s_tax_c * cum_inf * s_work_frac, 0)
            cash += np.where(p_alive, p_cash_c * cum_inf * p_work_frac, 0) + np.where(s_alive, s_cash_c * cum_inf * s_work_frac, 0)
            hsa += np.where(p_alive, p_hsa_c * cum_inf * p_work_frac, 0) + np.where(s_alive, s_hsa_c * cum_inf * s_work_frac, 0)

            p_civ_pen = np.where(p_alive, p_base_pension * p_pension_mult * p_ret_frac, np.where(s_alive, p_base_pension * p_fers_survivor_mult, 0))
            
            p_mil_pen_active = np.maximum(0, p_base_mil_gross - (p_base_va if not p_crdp else 0) - (p_base_mil_gross * 0.065 if p_mil_sbp and s_alive.any() else 0))
            p_mil_pen = np.where(p_alive & (age >= p_mil_start_age), p_mil_pen_active, np.where(~p_alive & s_alive & p_mil_sbp, p_base_mil_gross * 0.55, 0))
            
            yr_va_p = np.where(p_alive & (age >= p_mil_start_age), p_base_va, 0)
            
            s_civ_pen = np.where(s_alive, s_base_pension * s_pension_mult * s_ret_frac, np.where(p_alive, s_base_pension * s_fers_survivor_mult, 0))
            
            s_mil_pen_active = np.maximum(0, s_base_mil_gross - (s_base_va if not s_crdp else 0) - (s_base_mil_gross * 0.065 if s_mil_sbp and p_alive.any() else 0))
            s_mil_pen = np.where(s_alive & (spouse_age >= s_mil_start_age), s_mil_pen_active, np.where(~s_alive & p_alive & s_mil_sbp, s_base_mil_gross * 0.55, 0))
            
            yr_va_s = np.where(s_alive & (spouse_age >= s_mil_start_age), s_base_va, 0)
            
            yr_civ_pension = p_civ_pen + s_civ_pen
            yr_mil_pension = p_mil_pen + s_mil_pen
            yr_pension = yr_civ_pension + yr_mil_pension
            yr_va = yr_va_p + yr_va_s

            p_active_ss = p_base_ss * p_ss_modifier * ss_haircut
            s_active_ss = s_base_ss * s_ss_modifier * ss_haircut
            
            p_spousal_ss = np.where(spouse_age >= s_ss_claim, s_base_ss * 0.5 * p_spousal_modifier * ss_haircut, 0)
            s_spousal_ss = np.where(age >= p_ss_claim, p_base_ss * 0.5 * s_spousal_modifier * ss_haircut, 0)

            p_ss_val = np.where(age >= p_ss_claim, np.maximum(p_active_ss, p_spousal_ss), 0)
            s_ss_val = np.where(spouse_age >= s_ss_claim, np.maximum(s_active_ss, s_spousal_ss), 0)
            
            p_survivor_base = np.maximum(p_active_ss, p_base_ss * 0.825 * ss_haircut)
            s_survivor_base = np.maximum(s_active_ss, s_base_ss * 0.825 * ss_haircut)
            
            p_surv_penalty = np.clip(1.0 - ((67 - age) * (0.285 / 7.0)), 0.715, 1.0)
            s_surv_penalty = np.clip(1.0 - ((67 - spouse_age) * (0.285 / 7.0)), 0.715, 1.0)
            
            inherited_ss_from_s = np.where(age >= 60, s_survivor_base * p_surv_penalty, 0)
            inherited_ss_from_p = np.where(spouse_age >= 60, p_survivor_base * s_surv_penalty, 0)

            yr_ss = np.where(
                is_mfj, 
                p_ss_val + s_ss_val, 
                np.where(
                    p_alive, 
                    np.maximum(np.where(age >= p_ss_claim, p_ss_val, 0), inherited_ss_from_s), 
                    np.where(
                        s_alive, 
                        np.maximum(np.where(spouse_age >= s_ss_claim, s_ss_val, 0), inherited_ss_from_p), 
                        0
                    )
                )
            )

            history['salary_income'][:, yr] = yr_salary
            history['pension_income'][:, yr] = yr_pension
            history['va_income'][:, yr] = yr_va
            history['ss_income'][:, yr] = yr_ss
            
            total_guaranteed = yr_pension + yr_va + yr_ss + yr_salary
            history['guaranteed_income'][:, yr] = total_guaranteed
            
            tsp_pre = p_tsp + s_tsp
            ira_pre = p_ira + s_ira
            roth_pre = p_roth + s_roth
            hsa_pre = hsa.copy()
            prev_total_port = tsp_pre + ira_pre + roth_pre + taxable + cash + hsa_pre
            
            p_tsp = np.where(household_alive, p_tsp * (1 + returns[:, yr, 1]), p_tsp)
            s_tsp = np.where(household_alive, s_tsp * (1 + returns[:, yr, 1]), s_tsp)
            p_ira = np.where(household_alive, p_ira * (1 + returns[:, yr, 2]), p_ira)
            s_ira = np.where(household_alive, s_ira * (1 + returns[:, yr, 2]), s_ira)
            p_roth = np.where(household_alive, p_roth * (1 + returns[:, yr, 3]), p_roth)
            s_roth = np.where(household_alive, s_roth * (1 + returns[:, yr, 3]), s_roth)
            taxable = np.where(household_alive, taxable * (1 + returns[:, yr, 4]), taxable)
            hsa = np.where(household_alive, hsa * (1 + returns[:, yr, 5]), hsa)
            cash = np.where(household_alive, cash * (1 + cash_ret), cash)
            
            tsp = p_tsp + s_tsp
            ira = p_ira + s_ira
            roth = p_roth + s_roth
            
            current_total_port = tsp + ira + roth + taxable + cash + hsa
            history['port_return'][:, yr] = (current_total_port - prev_total_port) / np.maximum(prev_total_port, 1)
            history['real_return'][:, yr] = history['port_return'][:, yr] - inf_paths[:, yr]
            
            history['total_bal'][:, yr] = current_total_port + home_value
            history['total_bal_real'][:, yr] = (current_total_port + home_value) / cum_inf
            
            w_needed = np.zeros(self.iterations)
            constraint_flag = np.zeros(self.iterations)
            
            if current_year == self.ret_year or (yr == 0 and current_year > self.ret_year):
                target_lifestyle = (current_total_port * iwr) + total_guaranteed
                ref_draw = current_total_port * iwr 
                
                if sensitivity_mode == 'spend_up':
                    target_lifestyle *= 1.10
                    ref_draw *= 1.10
                elif sensitivity_mode == 'spend_down':
                    target_lifestyle *= 0.90
                    ref_draw *= 0.90
                
            if current_year >= self.ret_year:
                if yr > 0 and current_year > self.ret_year:
                    port_ret_prev = history['port_return'][:, yr-1]
                    inf_adj = np.where(port_ret_prev <= -0.10, 0, inf_paths[:, yr])
                    target_lifestyle *= (1 + inf_adj)
                    
                    w_needed = np.maximum(0, target_lifestyle - total_guaranteed)
                    ref_gap = ref_draw * cum_inf
                    
                    prosperity = w_needed < (ref_gap * 0.8)
                    target_lifestyle = np.where(prosperity, target_lifestyle * 1.05, target_lifestyle)
                    
                    preservation = w_needed > (ref_gap * 1.2)
                    target_lifestyle = np.where(preservation, target_lifestyle * 0.90, target_lifestyle)
                    constraint_flag = np.where(preservation, 1, constraint_flag)
                    
                    sweep_mask = port_ret_prev > 0.08
                    target_cash = w_needed * 2.0
                    cash_deficit = np.maximum(0, target_cash - cash)
                    excess_gains = np.maximum(0, port_ret_prev - 0.08) * prev_total_port
                    sweep_allowance = excess_gains * 0.50
                    actual_sweep = np.where(sweep_mask, np.minimum(sweep_allowance, cash_deficit), 0)
                    
                    sweep_from_taxable = np.minimum(actual_sweep, taxable)
                    taxable -= sweep_from_taxable
                    
                    sweep_from_tsp = np.minimum(actual_sweep - sweep_from_taxable, tsp)
                    ratio_p_tsp = np.where(tsp > 0, p_tsp / tsp, 0.0)
                    p_tsp -= sweep_from_tsp * ratio_p_tsp
                    s_tsp -= sweep_from_tsp * (1.0 - ratio_p_tsp)
                    tsp = p_tsp + s_tsp
                    
                    cash += (sweep_from_taxable + sweep_from_tsp)

                inflated_min_spend = min_spending * cum_inf
                
                if max_spending > 0:
                    inflated_max_spend = max_spending * cum_inf
                else:
                    inflated_max_spend = np.full(self.iterations, np.inf)
                    
                target_lifestyle = np.clip(target_lifestyle, inflated_min_spend + total_guaranteed, inflated_max_spend + total_guaranteed)
                w_needed = np.where(household_alive, np.maximum(0, target_lifestyle - total_guaranteed), 0)

            history['constraint_active'][:, yr] = constraint_flag
            history['income_gap'][:, yr] = w_needed
            
            # --- SPOUSAL INHERITANCE ROLLOVER ---
            roll_p_to_s = (~p_alive) & s_alive & ((p_tsp + p_ira + p_roth) > 0)
            s_tsp = np.where(roll_p_to_s, s_tsp + p_tsp, s_tsp)
            p_tsp = np.where(roll_p_to_s, 0.0, p_tsp)
            s_ira = np.where(roll_p_to_s, s_ira + p_ira, s_ira)
            p_ira = np.where(roll_p_to_s, 0.0, p_ira)
            s_roth = np.where(roll_p_to_s, s_roth + p_roth, s_roth)
            p_roth = np.where(roll_p_to_s, 0.0, p_roth)
            
            roll_s_to_p = (~s_alive) & p_alive & ((s_tsp + s_ira + s_roth) > 0)
            p_tsp = np.where(roll_s_to_p, p_tsp + s_tsp, p_tsp)
            s_tsp = np.where(roll_s_to_p, 0.0, s_tsp)
            p_ira = np.where(roll_s_to_p, p_ira + s_ira, p_ira)
            s_ira = np.where(roll_s_to_p, 0.0, s_ira)
            p_roth = np.where(roll_s_to_p, p_roth + s_roth, p_roth)
            s_roth = np.where(roll_s_to_p, 0.0, s_roth)
            
            tsp = p_tsp + s_tsp
            ira = p_ira + s_ira
            roth = p_roth + s_roth
            
            # --- TARGETED RMD LOGIC ---
            rmd_divisor_p = np.array([IRS_RMD_DIVISORS.get(a, 1.9 if a > 120 else 0.0) for a in np.clip(np.full(self.iterations, age), 0, 125)])
            rmd_divisor_s = np.array([IRS_RMD_DIVISORS.get(a, 1.9 if a > 120 else 0.0) for a in np.clip(np.full(self.iterations, spouse_age), 0, 125)])
            
            p_retired = current_year >= self.ret_year
            s_retired = current_year >= self.s_ret_year
            
            rmd_p_tsp = np.where(p_alive & p_retired & (rmd_divisor_p > 0), p_tsp * (1.0 / rmd_divisor_p), 0.0)
            rmd_s_tsp = np.where(s_alive & s_retired & (rmd_divisor_s > 0), s_tsp * (1.0 / rmd_divisor_s), 0.0)
            rmd_p_ira = np.where(p_alive & (rmd_divisor_p > 0), p_ira * (1.0 / rmd_divisor_p), 0.0)
            rmd_s_ira = np.where(s_alive & (rmd_divisor_s > 0), s_ira * (1.0 / rmd_divisor_s), 0.0)
            
            rmd_tsp = np.where(household_alive, rmd_p_tsp + rmd_s_tsp, 0.0)
            rmd_ira = np.where(household_alive, rmd_p_ira + rmd_s_ira, 0.0)
            
            rmds = rmd_tsp + rmd_ira
            history['rmds'][:, yr] = rmds
            
            p_tsp -= rmd_p_tsp
            s_tsp -= rmd_s_tsp
            p_ira -= rmd_p_ira
            s_ira -= rmd_s_ira
            
            tsp = p_tsp + s_tsp
            ira = p_ira + s_ira
            
            w_remaining = np.maximum(0, w_needed - rmds)
            excess_rmd = np.maximum(0, rmds - w_needed)
            history['extra_rmd'][:, yr] = excess_rmd
            
            w_tsp = np.zeros(self.iterations)
            w_ira = np.zeros(self.iterations)
            w_cash = np.zeros(self.iterations)
            w_taxable = np.zeros(self.iterations)
            w_roth = np.zeros(self.iterations)
            pull_hsa = np.zeros(self.iterations)
            
            if current_year >= self.ret_year:
                if yr > 0:
                    downturn_state = history['port_return'][:, yr-1] <= -0.10
                else:
                    downturn_state = np.zeros(self.iterations, dtype=bool)
                    
                normal_state = ~downturn_state
                
                pi_base_pre = rmds + yr_pension + yr_salary + (0.5 * yr_ss)
                
                t1_val = np.where(is_mfj, 32000, 25000)
                t2_val = np.where(is_mfj, 44000, 34000)
                calc_ss_base_pre = 0.5 * np.clip(pi_base_pre - t1_val, 0, t2_val - t1_val) + 0.85 * np.maximum(0, pi_base_pre - t2_val)
                taxable_ss_base_pre = np.minimum(0.85 * yr_ss, calc_ss_base_pre)
                
                base_taxable_pre = np.maximum(0, rmds + yr_pension + taxable_ss_base_pre + yr_salary - (deduction * cum_inf))
                bracket_space = np.maximum(0, limit_max_pct - base_taxable_pre)
                
                pull_tsp_1 = np.where(household_alive & normal_state, np.minimum(w_remaining, np.minimum(tsp, bracket_space)), 0)
                w_tsp += pull_tsp_1
                w_remaining -= pull_tsp_1
                ratio_p_tsp = np.where(tsp > 0, p_tsp / tsp, 0.0)
                p_tsp -= pull_tsp_1 * ratio_p_tsp
                s_tsp -= pull_tsp_1 * (1.0 - ratio_p_tsp)
                tsp = p_tsp + s_tsp
                bracket_space -= pull_tsp_1
                
                pull_ira_1 = np.where(household_alive & normal_state, np.minimum(w_remaining, np.minimum(ira, bracket_space)), 0)
                w_ira += pull_ira_1
                w_remaining -= pull_ira_1
                ratio_p_ira = np.where(ira > 0, p_ira / ira, 0.0)
                p_ira -= pull_ira_1 * ratio_p_ira
                s_ira -= pull_ira_1 * (1.0 - ratio_p_ira)
                ira = p_ira + s_ira
                
                pull_cash_1 = np.where(household_alive & downturn_state, np.minimum(w_remaining, cash), 0)
                w_cash += pull_cash_1
                w_remaining -= pull_cash_1
                cash -= pull_cash_1
                
                pull_tax_1 = np.where(household_alive, np.minimum(w_remaining, taxable), 0)
                w_taxable += pull_tax_1
                w_remaining -= pull_tax_1
                taxable -= pull_tax_1
                
                pull_cash_2 = np.where(household_alive & normal_state, np.minimum(w_remaining, cash), 0)
                w_cash += pull_cash_2
                w_remaining -= pull_cash_2
                cash -= pull_cash_2
                
                pull_hsa = np.where(household_alive, np.minimum(w_remaining, hsa), 0)
                w_remaining -= pull_hsa
                hsa -= pull_hsa
                
                pull_roth_1 = np.where(household_alive, np.minimum(w_remaining, roth), 0)
                w_roth += pull_roth_1
                w_remaining -= pull_roth_1
                ratio_p_roth = np.where(roth > 0, p_roth / roth, 0.0)
                p_roth -= pull_roth_1 * ratio_p_roth
                s_roth -= pull_roth_1 * (1.0 - ratio_p_roth)
                roth = p_roth + s_roth
                
                pull_tsp_2 = np.where(household_alive, np.minimum(w_remaining, tsp), 0)
                w_tsp += pull_tsp_2
                w_remaining -= pull_tsp_2
                ratio_p_tsp = np.where(tsp > 0, p_tsp / tsp, 0.0)
                p_tsp -= pull_tsp_2 * ratio_p_tsp
                s_tsp -= pull_tsp_2 * (1.0 - ratio_p_tsp)
                tsp = p_tsp + s_tsp
                
                pull_ira_2 = np.where(household_alive, np.minimum(w_remaining, ira), 0)
                w_ira += pull_ira_2
                w_remaining -= pull_ira_2
                ratio_p_ira = np.where(ira > 0, p_ira / ira, 0.0)
                p_ira -= pull_ira_2 * ratio_p_ira
                s_ira -= pull_ira_2 * (1.0 - ratio_p_ira)
                ira = p_ira + s_ira
            
            actual_portfolio_withdrawal = w_tsp + w_ira + w_cash + w_taxable + w_roth + pull_hsa + rmds - excess_rmd
            
            history['tsp_withdrawal'][:, yr] = w_tsp
            history['ira_withdrawal'][:, yr] = w_ira
            history['roth_withdrawal'][:, yr] = w_roth
            history['taxable_withdrawal'][:, yr] = w_taxable
            history['cash_withdrawal'][:, yr] = w_cash
            
            total_w_taxable = w_taxable
            gains_ratio = np.maximum(0, 1.0 - (taxable_basis / np.maximum(taxable, 1.0)))
            realized_gains = np.where(household_alive, total_w_taxable * gains_ratio, 0)
            taxable_basis = np.where(household_alive, taxable_basis - (total_w_taxable - realized_gains), taxable_basis)
            
            taxable = np.where(household_alive, taxable + excess_rmd, taxable)
            taxable_basis = np.where(household_alive, taxable_basis + excess_rmd, taxable_basis)
            
            pi_base = rmds + w_tsp + w_ira + yr_pension + yr_salary + realized_gains + (0.5 * yr_ss)
            
            t1_val = np.where(is_mfj, 32000, 25000)
            t2_val = np.where(is_mfj, 44000, 34000)
            calc_ss_base = 0.5 * np.clip(pi_base - t1_val, 0, t2_val - t1_val) + 0.85 * np.maximum(0, pi_base - t2_val)
            taxable_ss_base = np.minimum(0.85 * yr_ss, calc_ss_base)
            
            gross_income = rmds + w_tsp + w_ira + yr_pension + taxable_ss_base + yr_salary
            magi = gross_income.copy() 
            taxable_income = np.maximum(0, gross_income - (deduction * cum_inf)) 
            
            mfj_tax_fed = self.calc_tax_vectorized(taxable_income, TAX_BRACKETS_MFJ, cum_inf)
            single_tax_fed = self.calc_tax_vectorized(taxable_income, TAX_BRACKETS_SINGLE, cum_inf)
            base_tax_fed = np.where(is_mfj, mfj_tax_fed, single_tax_fed)
                
            mfj_ltcg = self.calc_tax_vectorized(taxable_income + realized_gains, LTCG_BRACKETS_MFJ, cum_inf) - self.calc_tax_vectorized(taxable_income, LTCG_BRACKETS_MFJ, cum_inf)
            single_ltcg = self.calc_tax_vectorized(taxable_income + realized_gains, LTCG_BRACKETS_SINGLE, cum_inf) - self.calc_tax_vectorized(taxable_income, LTCG_BRACKETS_SINGLE, cum_inf)
            ltcg_tax = np.where(is_mfj, mfj_ltcg, single_ltcg)
                
            niit_threshold_val = np.where(is_mfj, NIIT_THRESHOLD_MFJ, NIIT_THRESHOLD_SINGLE)
            niit_tax = np.where(magi > (niit_threshold_val * cum_inf), realized_gains * 0.038, 0.0)
            base_tax_fed += (ltcg_tax + niit_tax)
            
            # --- START COMPLEX STATE TAX MATRIX (PRE-CONVERSION) ---
            state_taxable_base = np.where(
                np.isin(state_str, NO_INCOME_TAX_STATES), 
                0.0, 
                np.where(
                    np.isin(state_str, STATES_TAXING_SS), 
                    taxable_income, 
                    np.maximum(0, taxable_income - taxable_ss_base)
                )
            )
            
            if state_str in MILITARY_PENSION_EXEMPT_STATES:
                state_taxable_base = np.maximum(0, state_taxable_base - yr_mil_pension)
                    
            is_fed_pension = self.inputs.get('pension_type', 'FERS') == 'FERS'
            ret_income = rmds + w_tsp + w_ira
            total_ret_income = ret_income + yr_civ_pension
            
            if state_str in FULL_RETIREMENT_EXEMPT_STATES:
                state_taxable_base = np.maximum(0, state_taxable_base - total_ret_income)
            elif state_str in FEDERAL_PENSION_EXEMPT_STATES and is_fed_pension:
                state_taxable_base = np.maximum(0, state_taxable_base - yr_civ_pension)
                state_excl_val = np.where(is_mfj, STATE_EXCLUSIONS_65_MFJ.get(state_str, 0.0), STATE_EXCLUSIONS_65_SINGLE.get(state_str, 0.0)) * cum_inf
                state_exclusion = np.where((age >= 65) | (spouse_age >= 65), state_excl_val, 0.0)
                if state_str == "NJ":
                    state_exclusion = np.where(magi > (150000 * cum_inf), 0.0, state_exclusion)
                allowed_exclusion = np.minimum(state_exclusion, ret_income)
                state_taxable_base = np.maximum(0, state_taxable_base - allowed_exclusion)
            else:
                state_excl_val = np.where(is_mfj, STATE_EXCLUSIONS_65_MFJ.get(state_str, 0.0), STATE_EXCLUSIONS_65_SINGLE.get(state_str, 0.0)) * cum_inf
                state_exclusion = np.where((age >= 65) | (spouse_age >= 65), state_excl_val, 0.0)
                if state_str == "NJ":
                    state_exclusion = np.where(magi > (150000 * cum_inf), 0.0, state_exclusion)
                allowed_exclusion = np.minimum(state_exclusion, total_ret_income)
                state_taxable_base = np.maximum(0, state_taxable_base - allowed_exclusion)
                        
            base_tax_state_local = state_taxable_base * combined_state_local_rate
            
            total_tax_fed = np.where(household_alive, base_tax_fed, 0)
            total_tax_state = np.where(household_alive, base_tax_state_local, 0)
            conv_amt = np.zeros(self.iterations)
            w_tax_cash_roth = np.zeros(self.iterations)
            final_taxable_income = np.where(household_alive, taxable_income, 0)
            final_magi = np.where(household_alive, magi, 0)
            
            if roth_strategy > 0 and current_year >= self.ret_year and age < 75:
                space = np.zeros(self.iterations)
                if roth_strategy in[1, 4]: 
                    mfj_space = np.zeros(self.iterations)
                    single_space = np.zeros(self.iterations)
                    
                    for i in range(len(TAX_BRACKETS_MFJ)):
                        mfj_space = np.where((taxable_income < TAX_BRACKETS_MFJ[i][0] * cum_inf) & (mfj_space == 0), (TAX_BRACKETS_MFJ[i][0] * cum_inf) - taxable_income - 1, mfj_space)
                    
                    for i in range(len(TAX_BRACKETS_SINGLE)):
                        single_space = np.where((taxable_income < TAX_BRACKETS_SINGLE[i][0] * cum_inf) & (single_space == 0), (TAX_BRACKETS_SINGLE[i][0] * cum_inf) - taxable_income - 1, single_space)
                    
                    space = np.where(is_mfj, mfj_space, single_space)
                    space = np.where(space > 1e6, 0, space)
                    
                    if roth_strategy == 1:
                        mfj_irmaa = np.zeros(self.iterations)
                        single_irmaa = np.zeros(self.iterations)
                        
                        for i in range(len(IRMAA_BRACKETS_MFJ)):
                            mfj_irmaa = np.where((magi < IRMAA_BRACKETS_MFJ[i][0] * cum_inf) & ((magi + space) >= IRMAA_BRACKETS_MFJ[i][0] * cum_inf), (IRMAA_BRACKETS_MFJ[i][0] * cum_inf) - magi - 1, mfj_irmaa)
                        
                        for i in range(len(IRMAA_BRACKETS_SINGLE)):
                            single_irmaa = np.where((magi < IRMAA_BRACKETS_SINGLE[i][0] * cum_inf) & ((magi + space) >= IRMAA_BRACKETS_SINGLE[i][0] * cum_inf), (IRMAA_BRACKETS_SINGLE[i][0] * cum_inf) - magi - 1, single_irmaa)
                        
                        space = np.where(is_mfj, np.where(mfj_irmaa > 0, mfj_irmaa, space), np.where(single_irmaa > 0, single_irmaa, space))
                        
                elif roth_strategy == 2: 
                    space_mfj = np.maximum(0, (IRMAA_BRACKETS_MFJ[0][0] * cum_inf) - magi - 1)
                    space_single = np.maximum(0, (IRMAA_BRACKETS_SINGLE[0][0] * cum_inf) - magi - 1)
                    space = np.where(is_mfj, space_mfj, space_single)
                    
                elif roth_strategy == 3:
                    space_mfj = np.maximum(0, (IRMAA_BRACKETS_MFJ[1][0] * cum_inf) - magi - 1)
                    space_single = np.maximum(0, (IRMAA_BRACKETS_SINGLE[1][0] * cum_inf) - magi - 1)
                    space = np.where(is_mfj, space_mfj, space_single)
                
                space = np.minimum(space, np.maximum(0, limit_max_pct - taxable_income - 1))
                
                conv_from_ira = np.where(household_alive, np.minimum(space, ira), 0)
                ratio_p_ira = np.where(ira > 0, p_ira / ira, 0.0)
                p_ira -= conv_from_ira * ratio_p_ira
                s_ira -= conv_from_ira * (1.0 - ratio_p_ira)
                ira = p_ira + s_ira
                
                conv_from_tsp = np.where(household_alive, np.minimum(space - conv_from_ira, tsp), 0)
                ratio_p_tsp = np.where(tsp > 0, p_tsp / tsp, 0.0)
                p_tsp -= conv_from_tsp * ratio_p_tsp
                s_tsp -= conv_from_tsp * (1.0 - ratio_p_tsp)
                tsp = p_tsp + s_tsp
                
                conv_amt = conv_from_ira + conv_from_tsp
                
                pi_conv = pi_base + conv_amt
                
                t1_val = np.where(is_mfj, 32000, 25000)
                t2_val = np.where(is_mfj, 44000, 34000)
                calc_ss_conv = 0.5 * np.clip(pi_conv - t1_val, 0, t2_val - t1_val) + 0.85 * np.maximum(0, pi_conv - t2_val)
                taxable_ss_conv = np.minimum(0.85 * yr_ss, calc_ss_conv)
                
                final_gross_income = rmds + w_tsp + w_ira + yr_pension + taxable_ss_conv + yr_salary + conv_amt
                final_taxable_income = np.where(household_alive, np.maximum(0, final_gross_income - (deduction * cum_inf)), 0)
                final_magi = np.where(household_alive, final_gross_income.copy(), 0)
                
                new_mfj_tax_fed = self.calc_tax_vectorized(final_taxable_income, TAX_BRACKETS_MFJ, cum_inf)
                new_single_tax_fed = self.calc_tax_vectorized(final_taxable_income, TAX_BRACKETS_SINGLE, cum_inf)
                new_tax_fed = np.where(is_mfj, new_mfj_tax_fed, new_single_tax_fed)
                
                new_mfj_ltcg = self.calc_tax_vectorized(final_taxable_income + realized_gains, LTCG_BRACKETS_MFJ, cum_inf) - self.calc_tax_vectorized(final_taxable_income, LTCG_BRACKETS_MFJ, cum_inf)
                new_single_ltcg = self.calc_tax_vectorized(final_taxable_income + realized_gains, LTCG_BRACKETS_SINGLE, cum_inf) - self.calc_tax_vectorized(final_taxable_income, LTCG_BRACKETS_SINGLE, cum_inf)
                new_ltcg_tax = np.where(is_mfj, new_mfj_ltcg, new_single_ltcg)
                    
                new_niit_tax = np.where(final_magi > (np.where(is_mfj, NIIT_THRESHOLD_MFJ, NIIT_THRESHOLD_SINGLE) * cum_inf), realized_gains * 0.038, 0.0)
                extra_tax_fed = (new_tax_fed + new_ltcg_tax + new_niit_tax) - base_tax_fed
                
                # --- START COMPLEX STATE TAX MATRIX (POST-CONVERSION) ---
                new_state_taxable_base = np.where(
                    np.isin(state_str, NO_INCOME_TAX_STATES), 
                    0.0, 
                    np.where(
                        np.isin(state_str, STATES_TAXING_SS), 
                        final_taxable_income, 
                        np.maximum(0, final_taxable_income - taxable_ss_conv)
                    )
                )

                if state_str in MILITARY_PENSION_EXEMPT_STATES:
                    new_state_taxable_base = np.maximum(0, new_state_taxable_base - yr_mil_pension)
                
                ret_income_conv = rmds + w_tsp + w_ira + conv_amt
                total_ret_income_conv = ret_income_conv + yr_civ_pension
                
                if state_str in FULL_RETIREMENT_EXEMPT_STATES:
                    new_state_taxable_base = np.maximum(0, new_state_taxable_base - total_ret_income_conv)
                elif state_str in FEDERAL_PENSION_EXEMPT_STATES and is_fed_pension:
                    new_state_taxable_base = np.maximum(0, new_state_taxable_base - yr_civ_pension)
                    state_excl_val = np.where(is_mfj, STATE_EXCLUSIONS_65_MFJ.get(state_str, 0.0), STATE_EXCLUSIONS_65_SINGLE.get(state_str, 0.0)) * cum_inf
                    state_exclusion = np.where((age >= 65) | (spouse_age >= 65), state_excl_val, 0.0)
                    if state_str == "NJ":
                        state_exclusion = np.where(final_magi > (150000 * cum_inf), 0.0, state_exclusion)
                    allowed_exclusion_conv = np.minimum(state_exclusion, ret_income_conv)
                    new_state_taxable_base = np.maximum(0, new_state_taxable_base - allowed_exclusion_conv)
                else:
                    state_excl_val = np.where(is_mfj, STATE_EXCLUSIONS_65_MFJ.get(state_str, 0.0), STATE_EXCLUSIONS_65_SINGLE.get(state_str, 0.0)) * cum_inf
                    state_exclusion = np.where((age >= 65) | (spouse_age >= 65), state_excl_val, 0.0)
                    if state_str == "NJ":
                        state_exclusion = np.where(final_magi > (150000 * cum_inf), 0.0, state_exclusion)
                    allowed_exclusion_conv = np.minimum(state_exclusion, total_ret_income_conv)
                    new_state_taxable_base = np.maximum(0, new_state_taxable_base - allowed_exclusion_conv)

                extra_state_taxable = np.maximum(0, new_state_taxable_base - state_taxable_base)

                extra_tax_state = extra_state_taxable * combined_state_local_rate
                extra_tax_total = extra_tax_fed + extra_tax_state
                
                if pay_taxes_from_cash:
                    w_tax_cash_roth = np.minimum(cash, extra_tax_total)
                    cash -= w_tax_cash_roth
                    rem_tax = extra_tax_total - w_tax_cash_roth
                    w_tax_taxable = np.minimum(taxable, rem_tax)
                    taxable -= w_tax_taxable
                    gains_ratio_tax = np.maximum(0, 1.0 - (taxable_basis / np.maximum(taxable + w_tax_taxable, 1.0)))
                    taxable_basis -= (w_tax_taxable - (w_tax_taxable * gains_ratio_tax))
                    net_to_roth = np.maximum(0, conv_amt - (rem_tax - w_tax_taxable))
                else:
                    net_to_roth = np.maximum(0, conv_amt - extra_tax_total)
                    
                p_roth += np.where(p_alive, net_to_roth, 0.0)
                s_roth += np.where(~p_alive & s_alive, net_to_roth, 0.0)
                roth = p_roth + s_roth
                
                total_tax_fed = np.where(household_alive, new_tax_fed + new_ltcg_tax + new_niit_tax, 0)
                total_tax_state = np.where(household_alive, base_tax_state_local + extra_tax_state, 0)
            
            history['roth_conversion'][:, yr] = conv_amt
            history['taxes_fed'][:, yr] = total_tax_fed
            history['taxes_state'][:, yr] = total_tax_state
            history['tax_paid'][:, yr] = total_tax_fed + total_tax_state
            history['taxable_income'][:, yr] = final_taxable_income
            history['magi'][:, yr] = final_magi
            history['roth_taxes_from_cash'][:, yr] = w_tax_cash_roth 
            
            base_p_health = float(self.inputs.get('p_health_cost', 0))
            base_s_health = float(self.inputs.get('s_health_cost', 0))
            base_oop_cost_yr = base_oop_cost
            
            if sensitivity_mode == 'health_up':
                base_p_health *= 1.20
                base_s_health *= 1.20
                base_oop_cost_yr *= 1.20
            elif sensitivity_mode == 'health_down':
                base_p_health *= 0.80
                base_s_health *= 0.80
                base_oop_cost_yr *= 0.80
                
            MEDICARE_PART_A_BASE = 505.0

            age_morbidity = 1.025 ** max(0, age - self.inputs['current_age'])
            if yr > 0:
                med_cpi_cum *= (1 + (np.maximum(0, inf_paths[:, yr]) * 1.5))
                
            raw_oop = base_oop_cost_yr * med_cpi_cum * age_morbidity
            
            moop_mfj = MOOP_LIMITS.get(health_plan, (999999, 999999))[1]
            moop_single = MOOP_LIMITS.get(health_plan, (999999, 999999))[0]
            
            inflated_oop_mfj = np.minimum(raw_oop, moop_mfj * cum_inf)
            inflated_oop_single = np.minimum(raw_oop, moop_single * cum_inf)
            inflated_oop = np.where(is_mfj, inflated_oop_mfj, inflated_oop_single)
            
            p_med_cost = np.zeros(self.iterations)
            s_med_cost = np.zeros(self.iterations)
            p_health_prem = base_p_health * cum_inf
            s_health_prem = base_s_health * cum_inf

            intent_delay = 3 if self.inputs.get('intent_to_work_40_quarters', False) else 0
            primary_medicare_age = 65 + intent_delay

            if age >= primary_medicare_age:
                if health_plan in["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"]:
                    if has_40_quarters or intent_delay > 0:
                        p_med_cost += MEDICARE_PART_B_BASE * cum_inf
                        if intent_delay > 0:
                            p_med_cost *= (1.0 + (0.10 * intent_delay))
                        
                        mfj_irmaa_add = np.zeros(self.iterations)
                        single_irmaa_add = np.zeros(self.iterations)
                        
                        for i in range(len(IRMAA_BRACKETS_MFJ)):
                            mfj_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_MFJ[i][0] * cum_inf), IRMAA_BRACKETS_MFJ[i][1] * cum_inf, mfj_irmaa_add)
                        
                        for i in range(len(IRMAA_BRACKETS_SINGLE)):
                            single_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_SINGLE[i][0] * cum_inf), IRMAA_BRACKETS_SINGLE[i][1] * cum_inf, single_irmaa_add)
                            
                        p_med_cost += np.where(is_mfj, mfj_irmaa_add, single_irmaa_add)
                        
                        p_health_prem = np.zeros(self.iterations)
                        if wants_dental_vision:
                            p_health_prem += 600 * cum_inf
                        if has_dependent_children:
                            p_health_prem += (base_p_health * 0.5) * cum_inf
                    else:
                        aca_sub_cost = np.minimum(base_p_health * cum_inf, final_magi * 0.085)
                        med_tot_cost = (MEDICARE_PART_A_BASE + MEDICARE_PART_B_BASE) * cum_inf
                        
                        mfj_irmaa_add = np.zeros(self.iterations)
                        single_irmaa_add = np.zeros(self.iterations)
                        
                        for i in range(len(IRMAA_BRACKETS_MFJ)):
                            mfj_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_MFJ[i][0] * cum_inf), IRMAA_BRACKETS_MFJ[i][1] * cum_inf, mfj_irmaa_add)
                            
                        for i in range(len(IRMAA_BRACKETS_SINGLE)):
                            single_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_SINGLE[i][0] * cum_inf), IRMAA_BRACKETS_SINGLE[i][1] * cum_inf, single_irmaa_add)
                            
                        med_tot_cost += np.where(is_mfj, mfj_irmaa_add, single_irmaa_add)

                        stay_on_aca = aca_sub_cost < med_tot_cost
                        p_med_cost = np.where(stay_on_aca, 0, med_tot_cost)
                        dep_cost = (base_p_health * 0.5) * cum_inf if has_dependent_children else 0
                        den_cost = 600 * cum_inf if wants_dental_vision else 0
                        p_health_prem = np.where(stay_on_aca, aca_sub_cost, den_cost + dep_cost)
                
                elif "FEHB" not in health_plan and "TRICARE" not in health_plan:
                    p_med_cost += MEDICARE_PART_B_BASE * cum_inf
                    mfj_irmaa_add = np.zeros(self.iterations)
                    single_irmaa_add = np.zeros(self.iterations)
                    
                    for i in range(len(IRMAA_BRACKETS_MFJ)):
                        mfj_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_MFJ[i][0] * cum_inf), IRMAA_BRACKETS_MFJ[i][1] * cum_inf, mfj_irmaa_add)
                        
                    for i in range(len(IRMAA_BRACKETS_SINGLE)):
                        single_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_SINGLE[i][0] * cum_inf), IRMAA_BRACKETS_SINGLE[i][1] * cum_inf, single_irmaa_add)
                        
                    p_med_cost += np.where(is_mfj, mfj_irmaa_add, single_irmaa_add)

            s_health_plan = self.inputs.get('s_health_plan', "None/Self-Insure")
            if spouse_age >= 65:
                if s_health_plan in["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"]:
                    if has_40_quarters:
                        s_med_cost += MEDICARE_PART_B_BASE * cum_inf
                        mfj_irmaa_add = np.zeros(self.iterations)
                        single_irmaa_add = np.zeros(self.iterations)
                        
                        for i in range(len(IRMAA_BRACKETS_MFJ)):
                            mfj_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_MFJ[i][0] * cum_inf), IRMAA_BRACKETS_MFJ[i][1] * cum_inf, mfj_irmaa_add)
                            
                        for i in range(len(IRMAA_BRACKETS_SINGLE)):
                            single_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_SINGLE[i][0] * cum_inf), IRMAA_BRACKETS_SINGLE[i][1] * cum_inf, single_irmaa_add)
                            
                        s_med_cost += np.where(is_mfj, mfj_irmaa_add, single_irmaa_add)
                        
                        s_health_prem = np.zeros(self.iterations)
                        if wants_dental_vision:
                            s_health_prem += 600 * cum_inf
                
                elif "FEHB" not in s_health_plan and "TRICARE" not in s_health_plan:
                    s_med_cost += MEDICARE_PART_B_BASE * cum_inf
                    mfj_irmaa_add = np.zeros(self.iterations)
                    single_irmaa_add = np.zeros(self.iterations)
                    
                    for i in range(len(IRMAA_BRACKETS_MFJ)):
                        mfj_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_MFJ[i][0] * cum_inf), IRMAA_BRACKETS_MFJ[i][1] * cum_inf, mfj_irmaa_add)
                        
                    for i in range(len(IRMAA_BRACKETS_SINGLE)):
                        single_irmaa_add = np.where(final_magi > (IRMAA_BRACKETS_SINGLE[i][0] * cum_inf), IRMAA_BRACKETS_SINGLE[i][1] * cum_inf, single_irmaa_add)
                        
                    s_med_cost += np.where(is_mfj, mfj_irmaa_add, single_irmaa_add)

            history['medicare_cost'][:, yr] = np.where(p_alive, p_med_cost, 0) + np.where(s_alive, s_med_cost, 0)

            w_hsa_oop = np.minimum(hsa, inflated_oop)
            hsa -= w_hsa_oop
            
            health_cost_base = np.where(p_alive, p_health_prem, 0) + np.where(s_alive, s_health_prem, 0) + (inflated_oop - w_hsa_oop)
            history['health_cost'][:, yr] = np.where(household_alive, health_cost_base, 0)
            
            history['hsa_withdrawal'][:, yr] = pull_hsa + w_hsa_oop
            
            current_mortgage = np.where(household_alive, np.full(self.iterations, mortgage_pmt if yr < mortgage_yrs else 0.0), 0)
            history['mortgage_cost'][:, yr] = current_mortgage
            
            if current_year >= self.ret_year:
                years_in_ret = current_year - self.ret_year
                smile_mult = 1.0 - (0.015 * years_in_ret) + (0.0005 * (years_in_ret ** 2))
                current_add_exp = np.where(household_alive, base_add_exp * cum_inf * smile_mult, 0)
            else:
                current_add_exp = np.zeros(self.iterations)
                
            history['additional_expenses'][:, yr] = current_add_exp
            
            history['tsp_bal'][:, yr] = tsp
            history['ira_bal'][:, yr] = ira
            history['roth_bal'][:, yr] = roth
            history['taxable_bal'][:, yr] = taxable
            history['cash_bal'][:, yr] = cash
            history['hsa_bal'][:, yr] = hsa
            
            if current_year >= self.ret_year:
                total_deductions = total_tax_fed + total_tax_state + history['medicare_cost'][:, yr] + history['health_cost'][:, yr] + current_mortgage + current_add_exp
                spendable = actual_portfolio_withdrawal + yr_pension + yr_va + yr_ss + yr_salary - total_deductions
                history['net_spendable'][:, yr] = np.where(household_alive, spendable, 0)
            else:
                history['net_spendable'][:, yr] = 0.0
            
            just_died_mask = (~household_alive) & (history['terminal_year'] == 0)
            history['terminal_year'][just_died_mask] = yr

        history['terminal_year'][history['terminal_year'] == 0] = self.years - 1
            
        return history

    def objective_function(self, iwr_test):
        history = self.run_mc(iwr_test, seed=42, roth_strategy=0)
        
        liquid_real_path = history['total_bal_real'][np.arange(self.iterations), history['terminal_year']] - (history['home_value'][np.arange(self.iterations), history['terminal_year']] / history['cum_inf'][np.arange(self.iterations), history['terminal_year']])
        median_real_path = np.median(liquid_real_path)
        
        target_floor = self.inputs.get('target_floor', 0.0)
        return median_real_path - target_floor

    def optimize_iwr(self):
        self.optimization_error = False
        try:
            return optimize.brentq(self.objective_function, a=0.001, b=0.40, xtol=1e-4, maxiter=40)
        except Exception as e:
            print(f"IWR Optimization Failed: {e}")
            self.optimization_error = True
            return 0.04
            
    def analyze_portfolios(self, opt_iwr, roth_strategy=0):
        results = {}
        hist_custom = self.run_mc(opt_iwr, seed=42, roth_strategy=roth_strategy, override_port=None)
        
        liquid_custom = hist_custom['total_bal_real'][np.arange(self.iterations), hist_custom['terminal_year']] - (hist_custom['home_value'][np.arange(self.iterations), hist_custom['terminal_year']] / hist_custom['cum_inf'][np.arange(self.iterations), hist_custom['terminal_year']])
        
        results["Your Custom Mix"] = {'wealth': np.median(liquid_custom), 'cut_prob': np.mean(np.any(hist_custom['constraint_active'] == 1, axis=1)) * 100}
        del hist_custom
        gc.collect()
        
        for port in["Conservative (20% Stock / 80% Bond)", "Moderate (60% Stock / 40% Bond)", "Aggressive (100% Stock)", "Dynamic Glidepath (Target Date)"]:
            hist = self.run_mc(opt_iwr, seed=42, roth_strategy=roth_strategy, override_port=port)
            liquid_port = hist['total_bal_real'][np.arange(self.iterations), hist['terminal_year']] - (hist['home_value'][np.arange(self.iterations), hist['terminal_year']] / hist['cum_inf'][np.arange(self.iterations), hist['terminal_year']])
            results[port] = {'wealth': np.median(liquid_port), 'cut_prob': np.mean(np.any(hist['constraint_active'] == 1, axis=1)) * 100}
            del hist
            gc.collect()
            
        return results

    def analyze_roth_strategies(self, opt_iwr):
        user_max = float(self.inputs.get("max_tax_bracket", 0.24)) * 100
        strats =[(0, 'Baseline (None)'), (1, 'Fill Current Bracket (IRMAA Protected)'), (2, 'Target IRMAA Tier 1'), (3, 'Target IRMAA Tier 2'), (4, f'Max User Bracket Fill ({user_max:.0f}%)')]
        
        results, best_wealth, winner_name, winner_hist = {}, -np.inf, 'Baseline (None)', None
        
        for s_idx, s_name in strats:
            hist = self.run_mc(opt_iwr, seed=42, roth_strategy=s_idx)
            
            liquid_wealth = hist['total_bal_real'][np.arange(self.iterations), hist['terminal_year']] - (hist['home_value'][np.arange(self.iterations), hist['terminal_year']] / hist['cum_inf'][np.arange(self.iterations), hist['terminal_year']])
            wealth = np.median(liquid_wealth)
            
            results[s_name] = {'wealth': wealth, 'taxes': np.sum(np.median(hist['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist['rmds'], axis=0)), 'tax_path': np.median(hist['taxes_fed'], axis=0), 'conv_path': np.median(hist['roth_conversion'], axis=0), 'taxable_inc_path': np.median(hist['taxable_income'], axis=0)}
            
            if wealth > best_wealth:
                best_wealth, winner_name, winner_hist = wealth, s_name, hist 
            else:
                del hist 
                gc.collect()
                
        return results, winner_name, winner_hist

    def run_sensitivity_analysis(self, opt_iwr):
        modes =[
            ('Market Returns (±1%)', 'market_up', 'market_down'),
            ('Inflation Rate (±1%)', 'inf_down', 'inf_up'), 
            ('Discretionary Spend (±10%)', 'spend_down', 'spend_up'),
            ('Healthcare Costs (±20%)', 'health_down', 'health_up'),
            ('Guaranteed Income (±10%)', 'income_up', 'income_down')
        ]
        
        hist_base = self.run_mc(opt_iwr, seed=42)
        liquid_base = hist_base['total_bal_real'][np.arange(self.iterations), hist_base['terminal_year']] - (hist_base['home_value'][np.arange(self.iterations), hist_base['terminal_year']] / hist_base['cum_inf'][np.arange(self.iterations), hist_base['terminal_year']])
        base_legacy = np.median(liquid_base)
        base_success = np.mean(liquid_base >= 1.0) * 100
        del hist_base
        gc.collect()
        
        results =[]
        for label, mode_pos, mode_neg in modes:
            h_pos = self.run_mc(opt_iwr, seed=42, sensitivity_mode=mode_pos)
            liquid_pos = h_pos['total_bal_real'][np.arange(self.iterations), h_pos['terminal_year']] - (h_pos['home_value'][np.arange(self.iterations), h_pos['terminal_year']] / h_pos['cum_inf'][np.arange(self.iterations), h_pos['terminal_year']])
            legacy_pos = np.median(liquid_pos)
            
            h_neg = self.run_mc(opt_iwr, seed=42, sensitivity_mode=mode_neg)
            liquid_neg = h_neg['total_bal_real'][np.arange(self.iterations), h_neg['terminal_year']] - (h_neg['home_value'][np.arange(self.iterations), h_neg['terminal_year']] / h_neg['cum_inf'][np.arange(self.iterations), h_neg['terminal_year']])
            legacy_neg = np.median(liquid_neg)
            
            results.append({
                'Factor': label,
                'Positive Impact': legacy_pos - base_legacy,
                'Negative Impact': legacy_neg - base_legacy
            })
            
            del h_pos, h_neg
            gc.collect()
            
        return base_success, results