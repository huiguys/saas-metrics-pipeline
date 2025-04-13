# Visualization Guide

## Tableau Setup

1. **Connect Data**:
   # Recommended data preparation:
   daily_summary['date'] = pd.to_datetime(daily_summary['date']).dt.strftime('%Y-%m-%d')
Key Dashboards:

Cohort Retention Heatmap
Data: cohort_retention.csv

Steps:

Drag cohort_month to Rows

Drag months_since_cohort to Columns

Drag retention_rate to Color

Set Marks to "Square"

Cohort Example

Churn Risk Analysis
python
Copy
# Calculated Field in Tableau:
IF [churn_risk] > 0.7 THEN "High"
ELSEIF [churn_risk] > 0.4 THEN "Medium"
ELSE "Low" END
Power BI Guide
Import Data:

Use "Get Data" → Text/CSV

Set date columns as DateTime

Sample Reports:

Funnel Visualization
Data: funnel_counts.csv

Use the "Funnel" custom visual

Map:

step to Category

user_count to Values

User Segments

Segment LTV = 
VAR CurrentSegment = SELECTEDVALUE(user_activity[user_segment])
RETURN
CALCULATE(
    AVERAGE(user_activity[simulated_ltv]),
    user_activity[user_segment] = CurrentSegment
)

Best Practices
Color Schemes:

Churn risk: Red (High) → Yellow → Green (Low)

Anomalies: Bright red with alert icons

Filters:

Add date range slicers to all dashboards

Segment filter for user-level views

Performance:

Use "Extract" mode in Tableau for large datasets

Power BI: Enable "Aggregation" on user_activity.csv
