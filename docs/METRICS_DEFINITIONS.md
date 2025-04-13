# SaaS Metrics Definitions

## Core Metrics

| Metric | Formula | Interpretation | Ideal Target |
|--------|---------|----------------|--------------|
| **DAU (Daily Active Users)** | Unique users with ≥1 event/day | Measures daily engagement | Varies by product (≥20% MAU) |
| **MAU (Monthly Active Users)** | Unique users with ≥1 event/30d | Total active user base | Month-over-month growth |
| **Churn Rate** | `(Churned users ÷ Starting users) × 100` | User attrition | <5% monthly |
| **LTV (Lifetime Value)** | `Avg MRR × Avg customer lifespan` | Revenue per user | ≥3× CAC |

## Engagement Metrics

| Metric | Calculation | Significance |
|--------|-------------|--------------|
| **Feature Adoption** | `Users using feature ÷ Total active users` | Product value perception |
| **Stickiness** | `DAU ÷ MAU` | Daily usage frequency | >20% |
| **Session Duration** | `Sum(event timestamps) ÷ Session count` | Engagement depth | Product-dependent |

## Revenue Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MRR (Monthly Recurring Revenue)** | `Sum of all subscription fees` | Use simulated values in this project |
| **ARPU** | `MRR ÷ Active users` | Benchmark against industry |
| **Expansion MRR** | `Upsell revenue - Downgrade revenue` | Track in user_activity.csv |

## Technical Metrics

| Metric | Source Column | Data Type |
|--------|--------------|----------|
| **Churn Risk Score** | `churn_risk` (0-1) | Machine learning output |
| **Anomaly Flag** | `is_anomaly` (True/False) | Statistical detection |
| **User Segment** | `user_segment` (0-3) | K-Means cluster ID |
