from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 131, 123) # Boldin Teal
        self.cell(0, 10, 'Advanced Quantitative Retirement Plan', 0, 1, 'C')
        self.set_font('Arial', 'I', 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Institution-Grade Actuarial Analysis & Executive Summary', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.date.today().strftime("%Y-%m-%d")}', 0, 0, 'C')

def generate_pdf(data):
    pdf = PDF()
    pdf.add_page()
    
    # 1. PLAN INSIGHTS
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, '1. Core Plan Insights', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    status = "ON TRACK" if data['prob_success'] >= 85 else "AT RISK"
    pdf.multi_cell(0, 8, f"Probability of Survival (> $0): {data['prob_success']:.1f}% ({status})\n"
                         f"Probability of Meeting Target Legacy: {data['prob_legacy']:.1f}%\n"
                         f"Median Terminal Legacy (Today's Dollars): ${data['terminal_wealth']:,.0f}\n"
                         f"Estimated Year 1 Portfolio Burn Rate: ${data['yr1_burn']:,.0f}\n"
                         f"Years of Safe Liquidity Buffer: {data['safe_years']}")
    
    pdf.ln(5)

    # 2. ROTH OPTIMIZATION
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Roth Conversion Optimizer', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, f"Recommended Strategy: {data['roth_winner']}\n"
                         f"Projected Nominal Lifetime Tax Savings (Un-discounted): ${max(0, data['tax_savings']):,.0f}\n"
                         f"Projected RMD Reduction: ${data['rmd_reduction']:,.0f}\n"
                         f"Net Increase to Terminal Legacy (Today's $): ${data['wealth_increase']:,.0f}")
    pdf.ln(5)

    # 3. SOCIAL SECURITY
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Social Security Strategy', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # NEW: Dynamic SS Logic
    if data['life_exp'] < 80:
        ss_verdict = "Actuarial Verdict: Claim Early (Age 62 or Current Age)\nReasoning: Because your entered life expectancy is below the mathematical crossover point (~Age 80-82), claiming early allows you to capture more total guaranteed income during your lifetime than if you delayed."
    elif data['ss_claim_age'] < 70:
        ss_verdict = f"Actuarial Verdict: You selected to claim at {data['ss_claim_age']}.\nReasoning: While delaying to 70 maximizes 'Longevity Insurance' by permanently increasing your payout by 8% per year, your chosen claiming age of {data['ss_claim_age']} has been fully modeled and stress-tested against your portfolio."
    else:
        ss_verdict = "Actuarial Verdict: Delay Claiming until Age 70\nReasoning: Alongside your Federal/Military Pension, Social Security is one of the few guaranteed, inflation-adjusted, market-immune income streams you possess. Delaying to 70 maximizes this 'Longevity Insurance', drastically reducing the withdrawal pressure placed on your TSP/Roth deep into retirement."
        
    pdf.multi_cell(0, 8, ss_verdict)
    pdf.ln(5)

    # 4. HEALTHCARE & MEDICARE
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '4. Healthcare & Medicare Part B', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, f"Declared Retiree Health Plan: {data['health_plan']}\n"
                         f"Projected Lifetime Medicare Part B + IRMAA Costs: ${data['total_medicare']:,.0f}\n"
                         f"Actuarial Verdict: {data['medicare_verdict']}")
    pdf.ln(5)

    # 5. ACTIONABLE TO-DO LIST
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '5. Coach Alerts & Actionable To-Do List', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    todos = (
        "1. Set Up Initial Paycheck: Establish a baseline systematic withdrawal rate equal to your Optimized IWR.\n\n"
        "2. Implement Cash Buffer: Physically separate 2 to 3 years worth of your 'Income Gap' into a high-yield Money Market or Taxable account to protect against Sequence of Return Risk.\n\n"
        "3. Execute Roth Strategy: Work with a CPA to schedule the recommended systematic Roth conversions mapped out in this report.\n\n"
        "4. Lock In Healthcare: Officially enroll in your selected Retiree Health plan and execute your Medicare Part B decision.\n\n"
        "5. Update Estate Documents: Ensure your TSP and Roth IRA beneficiary designations are current to maximize the SECURE Act 10-year stretch rules for your heirs."
    )
    pdf.multi_cell(0, 8, todos)
    
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "*Note: This is an actuarial simulation based on Monte Carlo stochastic modeling. For visual charts, full cash flow mapping, and step-by-step conversion tables, please refer to your interactive web dashboard.")

    return bytes(pdf.output())