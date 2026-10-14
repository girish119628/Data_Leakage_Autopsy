"""Generate the small subscription-churn dataset for the data leakage webinar.

One row = one customer in one month. About 5,500 rows, 560 customers, 36 months.
Target: churned_next_8m = 1 if the customer cancels within the next 8 months.

Leaks planted on purpose
  1. Preprocessing: missing values + very different column scales, so imputing/scaling
     before the split is the natural thing to do (small effect, by design).
  2. Proxy feature: cancellation_ticket_opened is written once the customer is leaving.
  3. Same customer in train and test: every customer has 8-12 monthly rows.
  4. Time: a price change and a new dashboard launch after month 24 shift behaviour.
  5. Selection before cross-validation: shown on a separate noise dataset (see
     make_noise below), not in this file.

Usage:  python make_data.py     ->  writes churn_data.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N_CUST = 560
MONTHS = 36
CUT = 24               # price change / new dashboard launch happens after this month

PARAMS = dict(
    churn_frac=0.45, Lc=(9, 13), Ln=(8, 12), cust_sd=0.30,
    logins_eff=0.85, tick_eff=3.5, spend_eff=0.5, disc_eff=0.9,
    w_old_late=0.75, w_new_early=0.65, uplift=1.15, steep=2.5,
    hard_frac=0.15, hard_churn_scale=0.1, hard_fake=0.7, H=8,
)


def build(seed=SEED, n_cust=N_CUST, months=MONTHS, cut=CUT, **kw):
    P = {**PARAMS, **kw}
    rng = np.random.default_rng(seed)
    rows = []
    plans = np.array(["Basic", "Standard", "Premium"])
    regions = np.array(["North", "South", "East", "West"])
    pays = np.array(["Card", "UPI", "NetBanking"])
    plan_spend = {"Basic": 300, "Standard": 650, "Premium": 1200}

    for i in range(n_cust):
        cid = f"C{i + 1:04d}"
        plan = rng.choice(plans, p=[0.5, 0.3, 0.2])
        region = rng.choice(regions)
        pay = rng.choice(pays, p=[0.4, 0.45, 0.15])
        age = int(rng.integers(18, 66))
        base_spend = plan_spend[plan] * np.exp(rng.normal(0, P["cust_sd"]))
        base_logins = np.exp(rng.normal(2.6, P["cust_sd"]))
        base_tickets = np.exp(rng.normal(-0.3, 0.4))
        prior_tenure = int(rng.integers(0, 30))

        churner = rng.random() < P["churn_frac"]
        hard = rng.random() < P["hard_frac"]       # customers whose behaviour misleads
        lo, hi = P["Lc"] if churner else P["Ln"]
        L = int(rng.integers(lo, hi))
        start = int(rng.integers(1, months - L + 2))
        T = start + L - 1                          # last snapshot month

        for j in range(L):
            m = start + j
            k = T - m                              # months before churn
            d = 1.0 / (1.0 + np.exp((k - (P["H"] - 0.5)) * P["steep"])) if churner else 0.0
            if hard:
                d = d * P["hard_churn_scale"] if churner else P["hard_fake"]
            late = m > cut

            # before the change churn shows in logins/tickets, after it in spend/discounts
            w_old = P["w_old_late"] if late else 1.0
            w_new = 1.0 if late else P["w_new_early"]

            logins = rng.poisson(max(0.5, base_logins * (1 - P["logins_eff"] * d * w_old)))
            tickets = rng.poisson(max(0.05, base_tickets * (1 + P["tick_eff"] * d * w_old)))
            up = P["uplift"] if late else 1.0      # price change
            spend = base_spend * up * (1 - P["spend_eff"] * d * w_new) * np.exp(rng.normal(0, 0.06))
            discount = int(rng.random() < min(0.95, 0.12 + P["disc_eff"] * d * w_new))
            new_dash = int(late and rng.random() < 0.6)   # feature launched after the cut

            label = int(churner and k < P["H"])
            proxy = int(rng.random() < (0.93 if label else 0.02))

            rows.append(dict(
                customer_id=cid, snapshot_month=m,
                customer_age=age,
                tenure_months=prior_tenure + j + 1,
                monthly_spend=round(spend, 2),
                logins_last_30d=logins,
                support_tickets_last_30d=tickets,
                plan_type=plan, payment_method=pay, region=region,
                discount_flag=discount,
                uses_new_dashboard=new_dash,
                cancellation_ticket_opened=proxy,
                churned_next_8m=label,
            ))

    df = pd.DataFrame(rows)
    for col, p in [("monthly_spend", 0.04), ("logins_last_30d", 0.05)]:
        mask = rng.random(len(df)) < p
        df.loc[mask, col] = np.nan
    return df


def make_noise(seed=36, n=120, p=3000):
    """Pure noise for Leak 5: random columns, random labels."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng.integers(0, 2, n)


if __name__ == "__main__":
    df = build()
    df.to_csv("churn_data.csv", index=False)
    print(df.shape, "churn rate:", round(df.churned_next_8m.mean(), 3))
