
````markdown
# SaaS Metrics Analytics Pipeline
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/pandas-2.0%2B-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)



A data pipeline project for simulating, tracking, and analyzing key metrics for a SaaS product, including machine learning insights like churn prediction and user segmentation.

## Features

- **Realistic Data Simulation**: Generates synthetic user event logs including signups, logins, feature usage, upgrades, and cancellations.
- **ETL Processing**: Cleans and transforms raw logs into structured analytical datasets using Pandas.
- **Sessionization**: Groups events into user sessions based on inactivity time.
- **Cohort Analysis**: Tracks user retention based on signup cohorts.
- **Funnel Analysis**: Calculates conversion rates through key user actions.
- **ML-Powered Analytics**:
  - **Churn Prediction:** Trains a Logistic Regression model to predict user churn risk based on activity patterns. (Check script output for in-sample accuracy).
  - **User Segmentation:** Uses K-Means clustering to group users into behavioral segments.
  - **Anomaly Detection:** Applies time-series decomposition to detect anomalies in daily signup trends.
- **Automated Insights**: Generates a text file with simple, rule-based insights derived from the analysis.
- **BI Ready Outputs**: Produces CSV files suitable for connection to tools like Tableau or Power BI.

## Architecture Flow

```mermaid
graph TD
    A[Data Generation<br>(generate_logs.py)] --> B(Raw Event Logs<br>raw_event_logs.csv);
    B --> C{ETL & Analysis<br>(process_logs.py)};
    C --> D(Daily Summary<br>daily_summary.csv);
    C --> E(User Activity<br>user_activity_summary.csv);
    C --> F(Session Metrics<br>session_metrics.csv);
    C --> G(Cohort Retention<br>cohort_retention.csv);
    C --> H(Funnel Data<br>funnel_counts.csv);
    D --> I[Anomaly Detection];
    E --> J[Churn Prediction];
    E --> K[User Segmentation];
    E --> L[LTV Simulation];
    I --> M(Automated Insights<br>auto_insights.txt);
    J --> M;
    K --> M;
    G --> M;
    H --> M;
````

## Quick Start

### Prerequisites

  - Python (tested with 3.9+)
  - pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    # Replace 'yourusername' with your actual GitHub username
    git clone [https://github.com/huiguys/saas-metrics-pipeline.git](https://github.com/huiguys/saas-metrics-pipeline.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd saas-metrics-pipeline
    ```
3.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Create environment
    python -m venv venv
    # Activate (Windows)
    .\venv\Scripts\activate
    # Activate (macOS/Linux)
    # source venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Generate sample event data:**

      * Run the script from the project's root directory:
        ```bash
        python generate_logs.py
        ```
      * This will create `raw_event_logs.csv` in the same directory.

2.  **Run the ETL and analytics pipeline:**

      * Run the script from the project's root directory:
        ```bash
        python process_logs.py
        ```
      * This reads `raw_event_logs.csv` and generates the following output files in the same directory:

3.  **Key Output Files:**

      * `daily_summary.csv`: Daily aggregated KPIs (signups, DAU, errors, anomaly flags, etc.).
      * `user_activity_summary.csv`: User-level profiles with aggregated metrics, churn risk scores, LTV simulation, and segment assignments.
      * `cohort_retention.csv`: Pivot table showing monthly cohort retention rates.
      * `funnel_counts.csv`: User counts and conversion rates for defined funnel steps.
      * `session_metrics.csv`: Metrics calculated per user session (duration, event counts, etc.).
      * `auto_insights.txt`: Text file with automated rule-based insights from the analysis.

## Data Model

### Simulated Event Types

The `generate_logs.py` script simulates events such as:

  * `signup`, `login`, `session_end`
  * `feature_A_used`, `feature_B_used`
  * `profile_update`, `search_performed`, `help_accessed`
  * `error_occurred`
  * `subscription_cancelled`, `plan_upgraded`

### Calculated Metrics & Features

The `process_logs.py` script calculates numerous metrics stored in the output files, including:

  - **Activity:** Daily Active Users (DAU), events per day, sessions per day.
  - **Engagement:** Feature usage counts and frequency (Feature A, Feature B, Search, etc.).
  - **Retention:** Cohort retention matrix.
  - **Funnel Conversion:** Step-by-step user progression.
  - **Churn:** 30-day inactivity flag (`potentially_churned`), predicted churn probability (`churn_risk_score`).
  - **Value:** Simulated MRR and LTV per user.
  - **Segmentation:** Assigned user segment ID based on behavior.
  - **Sessions:** Session duration, events per session.
  - **Anomalies:** Flags for anomalous daily signup counts.

## Visualization

The generated CSV files can be easily connected to Business Intelligence tools like Tableau, Power BI, Google Looker Studio, etc.

Refer to the **[VISUALIZATION\_GUIDE.md](docs/VISUALIZATION_GUIDE.md)** file for detailed steps and best practices on building dashboards with these outputs in Tableau and Power BI.

*(Note: This project does not include pre-built dashboard template files like `.twb` or `.pbix`. You will build the dashboards by connecting to the CSV data sources as described in the guide.)*

## Configuration

Currently, most configuration parameters (number of users/events to simulate, churn definition days, anomaly thresholds, file names) are hardcoded within the `.py` scripts. Modify the configuration variables directly within the scripts for adjustments.

## Contributing

Contributions are welcome\! Please follow these steps:

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`)
3.  Commit your Changes (`git commit -m 'Add some NewFeature'`)
4.  Push to the Branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` file for more information.
*(Note: You need to add a file named `LICENSE` containing the actual text of the MIT license to the repository).*

## Contact

Srinivasa PM - pmsrinivasa65@gmail.com

Project Link: [https://github.com/huiguys/saas-metrics-pipeline](https://github.com/huiguys/saas-metrics-pipeline)
