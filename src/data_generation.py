import numpy as np
import pandas as pd


def generate_synthetic_data(n=10000, random_state=42):
    np.random.seed(random_state)

    # ----------------------------
    # Customer Features
    # ----------------------------
    tenure = np.random.exponential(scale=12, size=n)
    monthly_usage = np.random.normal(50, 15, n)
    support_tickets = np.random.poisson(2, n)
    engagement_score = np.clip(np.random.normal(0.5, 0.15, n), 0, 1)

    plan_type = np.random.choice([0, 1], size=n, p=[0.6, 0.4])  # 1 = premium
    region = np.random.choice([0, 1, 2], size=n)

    # ----------------------------
    # Confounded Treatment Assignment
    # ----------------------------
    treatment_logit = (
        -1
        + 0.02 * tenure
        + 1.5 * (engagement_score > 0.6)
        + 0.8 * plan_type
    )

    treatment_prob = 1 / (1 + np.exp(-treatment_logit))
    treatment = np.random.binomial(1, treatment_prob)

    # ----------------------------
    # Base Churn Mechanism
    # ----------------------------
    churn_logit = (
        1.2
        - 0.03 * tenure
        + 0.4 * support_tickets
        - 2.0 * engagement_score
    )

    # Heterogeneous Treatment Effect
    true_uplift = (
        0.6 * (engagement_score < 0.4)
        - 0.3 * (engagement_score > 0.75)
    )

    churn_logit = churn_logit - treatment * true_uplift
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_usage": monthly_usage,
        "support_tickets": support_tickets,
        "engagement_score": engagement_score,
        "plan_type": plan_type,
        "region": region,
        "treatment": treatment,
        "churn": churn
    })

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("../data/synthetic_retention_data.csv", index=False)
    print("Synthetic dataset generated successfully.")
