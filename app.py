1. SYSTEM ROLE
You are a Senior Quantitative Actuary and Advanced Financial Planning Engineer.
Your task is to design and implement an institution-grade accurate Python-based application that:
	Builds an interactive web interface that can be run on a website independently 
	Collects structured user input 
	Runs a stochastic retirement simulation 
	Produces structured outputs (CSV + report with visualizations) 
	runs the Python code using Streamlit
	Embeds the App into Canva Free Website
Do not fabricate assumptions. Use user provided input and reliable external data sources where required.

2. PROGRAM OBJECTIVE
Develop a Python application that determines:
	Maximum Optimal Initial Withdrawal Rate (IWR) 
	Sustainable inflation-adjusted retirement income 
	Probability of meeting a terminal wealth floor 
Terminal Constraint:
The median (50th percentile) simulation must meet or exceed:
	Target Ending Total Balance at Life Expectancy Age 
Secondary Objective:
Maintain stable real net spendable income during ages 63–85 (“Go-Go years”).
________________________________________
3. APPLICATION ARCHITECTURE
The program must include:
A. Frontend (Web UI)
	Collect structured user inputs via form 
	Validate all inputs 
	Do NOT use example values as defaults 
B. Backend (Python Engine)
	Monte Carlo simulation (10,000 iterations) 
	Tax engine (federal + state + IRMAA) 
	Withdrawal optimization engine 
	Roth conversion analysis module 
	Medicare decision module 
	Social Security decision module
C. Outputs
	Two CSV datasets (strict format required) 
	Comprehensive client report with Visual analytics (charts/graphs) 
________________________________________
4. USER INPUTS
Collect ALL inputs explicitly.
Rule: Any text labeled “e.g.” is an example only and must NEVER be used as a default value.
Personal Data
	Current Age 
	Retirement Age 
	Life Expectancy Age 
	State and County 
	Tax Filing Status 
	Federal Employment Details (Grade, Step, Years of Service) 
	Salary 
	Pension Estimate 
	Social Security (at FRA) 
	Monthly Mortgage Amount (annual payment, years remaining, payoff amount) 
	Annual Health Insurance (policy + plan type) 
	Target Estate Floor Amount 
Account Balances (at retirement start)
	TSP 
	HSA (if available)
	Roth IRA
	Money Market 
	Taxable Investments 
________________________________________
5. ECONOMIC MODEL
To accurately capture geometric compounding and strictly prevent negative asset prices, market returns must NOT be modeled using a standard Normal distribution.
Monte Carlo
	Iterations: 10,000 
	Distribution Type: Multivariate Lognormal Distribution (simulated via Geometric Brownian Motion).
	Process: dS_t=μS_t dt+σS_t dW_t
	Volatility Drag Adjustments: The model must automatically convert the user’s "Historical Mean" (Arithmetic Mean) into a Geometric Mean for the simulation using the formula: GeometricMean≈ArithmeticMean-(σ^2/2)
.
	To properly stress-test the Guyton-Klinger and SORR guardrails, apply a heavy-tailed adjustment (e.g., a Student's t-distribution for the random shocks dW_t with degrees of freedom ν=5) to simulate realistic market crashes, rather than pure Gaussian shocks.
Cross-Asset Correlation Matrix
Market returns and inflation cannot be simulated as independent variables. High inflation historically exerts downward pressure on real asset returns. The model must enforce correlation between the continuous part of the Ornstein-Uhlenbeck inflation process and the market return processes.
	Implementation: Utilize a Cholesky Decomposition matrix applied to the standard normal random variables (Z) before generating the paths.
	Baseline Correlation Assumptions (if user does not override):
	Equities vs. Inflation: -0.15 (Slightly negative; equities offer partial long-term inflation protection but suffer in short-term inflationary shocks).
	Bonds vs. Inflation: -0.30 (Strongly negative; rising inflation drives up yields and crushes bond prices).
	Equities vs. Bonds: +0.10	to +0.20 (Moderate positive correlation in moderate regimes).
	Jump Impact: The jump-diffusion component of the inflation model (λ=0.1, jump mean = 5%) acts as an exogenous shock. When an inflation jump occurs, the model should optionally apply a proportional negative shock to the equity/bond returns for that specific period to simulate sudden stagflation.
________________________________________
6. RETIREMENT INCOME RULES
Pension (FERS)
	Multiplier: 
	1.1% if retirement age ≥ 62 
	Otherwise 1.0% 
	Apply FERS Diet COLA: 
	CPI ≤ 2% → full CPI 
	2–3% → capped at 2% 
	3% → CPI - 1%
Social Security
	Start at age 67 
	Apply 21% haircut at trust depletion (2035) 
________________________________________
7. WITHDRAWAL STRATEGY (MANDATORY LOGIC)
Core Method: CASAM (Constant Amortization Spending Model) + Guardrails
The withdrawal engine utilizes a dynamic amortization framework smoothed by path-dependent guardrails to maximize income while preventing premature portfolio depletion.
	Baseline Withdrawal (Explicit CASAM Definition)
	Definition: The CASAM method calculates the theoretical baseline withdrawal by treating the retirement portfolio as an amortizing asset over the client's remaining lifetime, accounting for the terminal estate floor.
	Mathematical Formula: The baseline withdrawal for a given year 
	t  is calculated using a standard annuity formula (PMT): Wbase,t=PMT(r, n, −Vt, F)

	r = Assumed real rate of return (In Year 1, the optimizer solves for the effective r that yields the Maximum IWR. In subsequent years, this baseline rate is held constant).
	n = Remaining years (Life Expectancy Age - Current Age).
	V_t = Current Total Portfolio Balance at the start of year t.
	F = Target Estate Floor Amount (Future Value constraint).
	Optimizer Integration: For Year 1, the deterministic optimizer solves for the exact Initial Withdrawal Amount (W_(base,1)) that satisfies the condition: Median Terminal Wealth ≥ Target Estate Floor. The resulting percentage (W_(base,1)/V_1) becomes the Maximum IWR.
	Inflation Adjustment
	Rule: Following Year 1, the actual scheduled withdrawal (W_(scheduled,t)) defaults to the prior year's actual withdrawal adjusted for inflation to maintain real purchasing power. 
	Formula: W_(scheduled,t)=W_(actual,t-1)×(1+〖"Inflation" 〗_t)
	Exception: This standard adjustment is superseded if Guyton-Klinger or SORR rules (below) are triggered.
	Guyton-Klinger Rules (Dynamic Guardrails)
These rules calculate the Current Withdrawal Rate (CWR_t=W_(scheduled,t)/V_t) and compare it to the Initial Withdrawal Rate (IWR) to enforce boundaries:
	Capital Preservation Rule (Ceiling):
	Trigger: If CWR_t>"IWR"×1.20 (i.e., current rate exceeds initial rate by +20%).
	Action: Reduce the scheduled withdrawal amount by 10%.
	Prosperity Rule (Floor):
	Trigger: If CWR_t<"IWR"×0.80 (i.e., current rate falls below initial rate by -20%).
	Action: Increase the scheduled withdrawal amount by 10%.
	Inflation Freeze Rule:
	Trigger: If the portfolio's nominal return in year t-1 was negative (<0%┤).
	Action: Forfeit the annual inflation adjustment for year t (withdrawal amount remains flat, barring other cuts).
	SORR Protection (Sequence of Return Risk) 
An explicit overlay to protect the portfolio against severe market drawdowns during the "Go-Go years."
	Trigger: If the Total Portfolio Balance drops by ≥10% year-over-year (V_t≤V_(t-1)×0.90).
	Action: Apply an immediate 10% reduction to the scheduled withdrawal amount (W_(scheduled,t)×0.90).
	Priority: This rule is evaluated independently of the Guyton-Klinger ceiling rule and ensures rapid income adjustment during severe market shocks.
________________________________________
8. LIQUIDATION ORDER
To precisely model Sequence of Return Risk (SORR) mitigation and tax consequences, the engine must coordinate cash flows across multiple accounts using a strict IF/THEN hierarchy.
 Normal Year Liquidation (Default Logic)
	Rule: In any year where the TSP/401(k) balance did not drop by 10% or more in the preceding year, the TSP is the exclusive funding source for all portfolio withdrawals.
	Ambiguity Check: The Money Market, Taxable Investments, and Roth IRA accounts must remain 100% completely untouched and allowed to compound. No proportional withdrawals across accounts are permitted.
Downturn Year Liquidation (SORR Mitigation)
	Trigger: If the Ending TSP Balance dropped by ≥10% in the prior year.
	Rule: Halt discretionary withdrawals from the TSP to prevent locking in losses. The annual withdrawal must be funded sequentially using the following strict priority order:
	Money Market (Draw down to $0 before moving to step 2).
	Taxable Account (Draw down to $0 before moving to step 3).
	Roth IRA (Last resort buffer).
	Depletion Fallback: If all three buffer accounts are entirely depleted during a downturn year, the system must revert to liquidating from the TSP to meet the spending need.
Multi-Account Coordination (RMD Overrides & Reinvestment)
Statutory tax laws supersede discretionary liquidation rules. The engine must coordinate RMDs and spending needs as follows:
	Mandatory Distributions: Beginning at age 75, the TSP must distribute the IRS-mandated minimum amount, even if a downturn year rule has triggered a halt on TSP withdrawals.
	Net Spending Need vs. RMD:
	Scenario A (RMD < Spending Need): Apply the RMD to the spending need. Fund the remaining shortfall using the Normal or Downturn logic defined above.
	Scenario B (RMD > Spending Need): Apply the necessary portion of the RMD to fully satisfy the spending need. The excess, unspent RMD cash must be automatically reinvested into the Taxable Account at the end of the year.
	Taxes: All taxes (Federal, State, IRMAA) generated by TSP distributions, RMDs, or Taxable account dividends are paid as an expense out of the total required withdrawal for that year, prior to determining the Net Spendable Income.
________________________________________
9. TAX & EXPENSE ENGINE
Must include:
	Federal tax brackets 
	State + county tax (include specific state and county tax nuances)
	IRMAA Medicare adjustments 
	Health insurance inflation 
	SS taxation (provisional income formula) 
	Use current IRS tax brackets (inflation-adjusted annually)
- Apply standard deduction by filing status
- Include:
- Ordinary income tax
- Long-term capital gains stacking
- Net Investment Income Tax (NIIT)
- Social Security taxation via provisional income formula
- IRMAA based on MAGI (2-year lookback)
Calculations:
	Total Income 
	Total Expenses 
	Net Spendable Income 
________________________________________
10. OPTIMIZATION MODULES
10.A.1 Maximum IWR Deterministic Optimization Architecture
To solve for the optimal Maximum Initial Withdrawal Rate (IWR) within a high-variance stochastic environment (incorporating Ornstein-Uhlenbeck jump-diffusion and path-dependent Guyton-Klinger rules), the system utilizes a 1-Dimensional Root-Finding Algorithm. Because terminal wealth is strictly monotonically decreasing as the IWR increases, the constrained maximization problem is transformed into a derivative-free root-finding objective.
1. Objective Function Formulation
Instead of using gradient-based constrained maximization, the engine minimizes the absolute difference between the simulated Median Terminal Wealth and the Target Estate Floor.
	Target Function: f(IWR)="Median"(W_T∣IWR)-"Estate Floor" 
	Objective: Find the root where f(IWR)=0. Satisfying this root inherently yields the absolute maximum sustainable IWR without violating the floor constraint.
2. Constraint Handling & Variance Control (CRN)
Due to the non-differentiable "kinks" introduced by tax cliffs (IRMAA), Medicare rules, and 10% Guyton-Klinger guardrail adjustments, standard gradient-based solvers (e.g., SLSQP) will fail.
	Algorithm: The engine implements Brent’s Method (scipy.optimize.brentq), a derivative-free bracketed root-finding algorithm. It is highly robust against non-linear guardrail triggers and requires the fewest calls to the heavy 10,000-iteration Monte Carlo engine.
	Variance Reduction (Mandatory): The engine enforces Common Random Numbers (CRN). The Numpy random state (seed) is strictly frozen inside the objective function for each optimization run. This ensures that the Ornstein-Uhlenbeck inflation paths and market return paths remain identical across all optimizer iterations. Consequently, the deterministic solver observes a smooth, noiseless curve where changes in terminal wealth are driven solely by changes in the IWR.
3. Convergence Tolerance & Performance Limits
To ensure rapid execution suitable for a Streamlit web interface without timing out, the deterministic optimizer is constrained by tolerances that align with the Monte Carlo standard error:
	Variable Step Tolerance (xtol): Set to 10^(-4) (0.01% or 1 basis point). In the context of financial planning, optimizing a withdrawal rate beyond 1 basis point exceeds practical application and wastes compute cycles on stochastic noise.
	Maximum Iterations (maxiter): Capped at 15 iterations. Because Brent’s method brackets a strictly monotonic function, convergence to 10^(-4) precision is mathematically expected within 6 to 10 iterations.
	Boundary Limits: The algorithm searches within a hardcoded realistic boundary (e.g., lower bound: 1.0%, upper bound: 15.0%). If the root lies outside these bounds, the system returns a specific exception to the UI indicating the estate floor is either mathematically unachievable or effortlessly exceeded.
B. Roth Conversion Analysis
1. Core Objective
The Roth conversion optimizer evaluates discrete conversion strategies during the "Conversion Window" (from Current Age / Retirement Age up to Age 74, prior to mandatory RMDs at 75).
	Evaluation Metric: The optimal strategy is strictly defined as the scenario that produces the highest Median Ending Total Balance (Terminal Wealth) at Life Expectancy, not simply the lowest lifetime tax paid (as tax minimization often ignores the opportunity cost of lost compound growth on the tax dollars paid).
2. Dynamic Scenario Testing (The Algorithm)
Instead of blind bracket-filling or computationally impossible multi-variable gradient searches, the engine will iteratively run the Monte Carlo simulation against a defined set of strategic ceilings. For each year in the Conversion Window, the system calculates the client's baseline Provisional Income and tests the following conversion limits:
	Scenario 0 (Baseline): No Roth conversions.
	Scenario 1 (Current Bracket Max): Convert exactly enough to reach the top of the client’s current marginal Federal tax bracket, stopping $1 short of the next bracket.
	Scenario 2 (IRMAA Tier 1 Limit): Convert up to exactly $1 below the first IRMAA MAGI cliff (or the client's baseline IRMAA tier, if already higher).
	Scenario 3 (IRMAA Tier 2 Limit): Convert up to exactly $1 below the second IRMAA MAGI cliff.
	Scenario 4 (Next Bracket Max): Convert up to the top of the next marginal Federal tax bracket, deliberately absorbing the higher tax rate and potential IRMAA hit to rapidly reduce future RMDs.
3. Explicit Constraint Logic
During each scenario test, the following rules are strictly enforced:
	Hard Cap on Account Balance: The annual conversion amount cannot exceed the remaining balance of the TSP/Traditional IRA.
	Tax Funding Hierarchy: The tax liability generated by the Roth conversion must be paid from outside assets to maximize the mathematical benefit. The engine must deduct the conversion taxes from the Taxable Account or Money Market first. If non-qualified funds are depleted, the taxes must be withheld from the conversion amount itself (which reduces the net amount entering the Roth).
	IRMAA Cliff Detection: If a standard tax bracket ceiling (e.g., Scenario 1 or 4) naturally falls within $5,000 of an IRMAA cliff, the algorithm must dynamically truncate the conversion amount to stop $1 short of that IRMAA cliff. This prevents triggering a thousands-of-dollars Medicare surcharge for the sake of a marginal conversion.
4. Module Outputs
The engine compares the Median Terminal Wealth of Scenarios 1–4 against the Baseline (Scenario 0).
	If no scenario beats the Baseline, the output recommendation is: No Conversions Recommended.
	If a scenario yields a higher Terminal Wealth, the engine outputs the winning strategy, detailing:
	The recommended annual conversion target (e.g., "Convert up to IRMAA Tier 1").
	The total projected lifetime tax savings.
	The projected reduction in lifetime RMDs.
	The net increases to the Target Ending Total Balance.
C. Medicare Part B Decision
	Compare: 
	Premiums + IRMAA 
	Part B vs self-insure modeling 
	Lifetime cost comparison
	Out-of-pocket risk 
	Output recommendation 
D. Social Security
	Compare standard scenarios, such as filing at 62 vs. 67 vs. 70, and see how each affects total assets over time.
	Use FRA amount and adjust for delayed credits or early filing reductions
________________________________________
11. OUTPUT REQUIREMENTS
CSV OUTPUTS (STRICT FORMAT)
Provide:
	Median (50th percentile) 
	10th percentile 
Columns MUST match EXACTLY:
Calendar Year, Age, Rate of Return, Inflation Rate, Real Rate of Return, Taxable ETF Balance, Roth IRA Balance, HSA Balance, Money Market Balance, Annual 401(k)/TSP Withdrawal, Pension, Social Security, RMD Amount, Extra RMD Amount, Roth Conversion Amount, Federal Taxes, State Taxes, Medicare Cost, Health Insurance Cost, Total Income, Total Expenses, Net Spendable Annual, Net Monthly, Ending 401(k)/TSP Balance, Ending Total Balance (excluding HSA), Withdrawal Constraint Active
________________________________________
12. CLIENT REPORT STRUCTURE
Model after Boldin PlannerPlus Report:
Lifetime Projections: A high-level overview showing the sustainability of the plan through your projected lifespan, including "optimistic" and "pessimistic" scenarios.
Cash Flow Forecast: Detailed monthly and annual charts illustrating income sources versus recurring and one-time expenses throughout retirement.
Net Worth Forecast: Projections of total assets and liabilities over time, including real estate and debt payoff schedules.
Income Analysis: A breakdown of all revenue streams, such as Social Security, pensions, annuities, and work income.
Expense & Budget Details: Itemized sections for recurring costs, healthcare (including Medicare and long-term care), and special "milestone" purchases.
Taxes: Modeling for federal and state tax brackets, estimated tax liability, and potential deductions.
Withdrawal Strategy: A specialized section showing how withdrawals from different account types (Taxable, Roth, Tax-Deferred) will unfold to meet cash flow needs. 
Monte Carlo Analysis: A statistical section that runs thousands of simulations to determine the probability of plan success under varying market conditions.
Roth Conversion Opportunities: Analysis identifying specific years where converting traditional IRA funds to Roth might be tax-advantageous.
Required Minimum Distributions (RMDs): Forecasts for mandatory withdrawals starting at age 73 (or later, based on birth year) to help with tax planning.
Savings & Contribution Limits: A "Financial Wellness Snapshot" showing how current savings align with annual IRS contribution limits.
Estate & Legacy Planning: High-level summaries of legacy goals and how remaining assets may pass to heirs. 
PlannerPlus Coach Alerts: A summary of specific risks, oversights, or opportunities identified by the system’s automated monitoring.
Actionable To-Do List: A dynamic list of prioritized steps for the client to take to improve their financial security. 
