import numpy as np

# --- 2025 IRS TAX BRACKETS & LIMITS ---
STD_DED_SINGLE = 15000
STD_DED_MFJ = 30000

# IRS Extra Standard Deduction for Age 65+ or Blind (2024/2025 estimates)
EXTRA_DED_65_SINGLE = 1950
EXTRA_DED_65_MFJ_PER_PERSON = 1550

TAX_BRACKETS_SINGLE =[
    (11925, 0.10), (48475, 0.12), (103350, 0.22), 
    (197300, 0.24), (250525, 0.32), (626350, 0.35), (np.inf, 0.37)
]
TAX_BRACKETS_MFJ =[
    (23850, 0.10), (96950, 0.12), (206700, 0.22), 
    (394600, 0.24), (501050, 0.32), (751600, 0.35), (np.inf, 0.37)
]

# --- LONG-TERM CAPITAL GAINS (LTCG) BRACKETS ---
LTCG_BRACKETS_SINGLE =[(47025, 0.0), (518900, 0.15), (np.inf, 0.20)]
LTCG_BRACKETS_MFJ =[(94050, 0.0), (583750, 0.15), (np.inf, 0.20)]

# NIIT (Net Investment Income Tax) Thresholds
NIIT_THRESHOLD_SINGLE = 200000
NIIT_THRESHOLD_MFJ = 250000

# 2024/2025 Estimated IRMAA Thresholds
MEDICARE_PART_B_BASE = 2096.40
IRMAA_BRACKETS_SINGLE =[(103000, 0), (129000, 838.8), (161000, 2101.2), (193000, 3362.4), (499999, 4624.8), (np.inf, 5043.6)]
IRMAA_BRACKETS_MFJ =[(206000, 0), (258000, 838.8), (322000, 2101.2), (386000, 3362.4), (749999, 4624.8), (np.inf, 5043.6)]

# --- COMPLEX STATE EXCLUSION MATRIX ---

# States with NO individual income tax
NO_INCOME_TAX_STATES =["AK", "FL", "NV", "NH", "SD", "TN", "TX", "WA", "WY"]

# States that fully exempt ALL retirement income (TSP, IRA, Pensions) but tax W2 Salary
FULL_RETIREMENT_EXEMPT_STATES = ["IL", "MS", "PA", "IA"]

# States that fully exempt Federal/Civil Service Pensions (but may tax IRA/TSP)
FEDERAL_PENSION_EXEMPT_STATES = ["AL", "HI", "LA", "MA", "NY", "KS"]

# States that fully exempt Military Pensions
MILITARY_PENSION_EXEMPT_STATES =[
    "AL", "AZ", "AR", "CT", "HI", "IL", "IN", "IA", "KS", "LA", "ME", "MA", 
    "MI", "MN", "MS", "MO", "NE", "NJ", "NY", "NC", "ND", "OH", "OK", "PA", 
    "RI", "SC", "UT", "WV", "WI"
]

# States that fully tax Social Security (Updated for recent phase-outs)
STATES_TAXING_SS =["CO", "CT", "KS", "MN", "MT", "NM", "RI", "UT", "VT"]

# Flat dollar exemptions deducted from state taxable income for Retirees Age 65+
STATE_EXCLUSIONS_65_SINGLE = {
    "MD": 34300, "GA": 65000, "NY": 20000, "CO": 24000, 
    "MI": 61518, "SC": 15000, "OK": 10000, "NJ": 100000,
    "VA": 12000, "AR": 6000, "DE": 12500
}
STATE_EXCLUSIONS_65_MFJ = {
    "MD": 68600, "GA": 130000, "NY": 40000, "CO": 48000, 
    "MI": 123036, "SC": 30000, "OK": 20000, "NJ": 100000,
    "VA": 24000, "AR": 12000, "DE": 25000
}

# Statutory / Effective marginal rates applied AFTER exemptions
STATE_TAX_RATES = {
    "AL": 0.050, "AK": 0.000, "AZ": 0.025, "AR": 0.039, "CA": 0.080, 
    "CO": 0.044, "CT": 0.050, "DE": 0.050, "FL": 0.000, "GA": 0.054, 
    "HI": 0.072, "ID": 0.058, "IL": 0.0495, "IN": 0.030, "IA": 0.057, 
    "KS": 0.057, "KY": 0.040, "LA": 0.042, "ME": 0.071, "MD": 0.047, 
    "MA": 0.050, "MI": 0.042, "MN": 0.068, "MS": 0.040, "MO": 0.049, 
    "MT": 0.059, "NE": 0.058, "NV": 0.000, "NH": 0.000, "NJ": 0.055, 
    "NM": 0.049, "NY": 0.055, "NC": 0.045, "ND": 0.025, "OH": 0.035, 
    "OK": 0.047, "OR": 0.087, "PA": 0.0307, "RI": 0.047, "SC": 0.064, 
    "SD": 0.000, "TN": 0.000, "TX": 0.000, "UT": 0.046, "VT": 0.066, 
    "VA": 0.057, "WA": 0.000, "WV": 0.042, "WI": 0.053, "WY": 0.000
}

# --- STATUTORY HEALTH INSURANCE MOOP LIMITS (Single, MFJ) ---
MOOP_LIMITS = {
    "FEHB FEPBlue Basic": (6500, 13000), "FEPBlue Standard": (6000, 12000), "FEPBlue Focus": (8500, 17000),
    "GEHA High": (6000, 12000), "GEHA Standard": (7000, 14000), "Aetna Open Access": (5000, 10000),
    "Aetna Direct": (5000, 10000), "Aetna Advantage": (5000, 10000), "Cigna": (6500, 13000),
    "TRICARE for Life": (3000, 3000), "None/Self-Insure": (999999, 999999)
}

# --- IRS UNIFORM LIFETIME TABLE (RMD Divisors) ---
IRS_RMD_DIVISORS = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8,
    85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5,
    95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 
    105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0
}

# --- PRE-SET ACTUARIAL PORTFOLIOS ---
PORTFOLIOS = {
    "Conservative (20% Stock / 80% Bond)": {"ret": 0.045, "vol": 0.06},
    "Moderate (60% Stock / 40% Bond)": {"ret": 0.070, "vol": 0.10},
    "Aggressive (100% Stock)": {"ret": 0.095, "vol": 0.15}
}