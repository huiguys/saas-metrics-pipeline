# process_logs.py - Final Optimized Version
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# Configuration
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
CHURN_DEFINITION_DAYS = 30
ANOMALY_STD_THRESHOLD = 2.0
SESSION_INACTIVITY_MINUTES = 30
CLUSTERING_N_CLUSTERS = 4
MIN_DAYS_FOR_ANOMALY_DETECTION = 15

def initialize_paths():
    """Initialize all file paths with proper error handling"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return {
            'raw_logs': os.path.join(script_dir, 'raw_event_logs.csv'),
            'daily_summary': os.path.join(script_dir, 'daily_summary.csv'),
            'user_activity': os.path.join(script_dir, 'user_activity_summary.csv'),
            'cohort_retention': os.path.join(script_dir, 'cohort_retention.csv'),
            'funnel_counts': os.path.join(script_dir, 'funnel_counts.csv'),
            'session_metrics': os.path.join(script_dir, 'session_metrics.csv'),
            'auto_insights': os.path.join(script_dir, 'auto_insights.txt')
        }
    except Exception as e:
        print(f"Error initializing file paths: {e}")
        sys.exit(1)

def load_raw_data(file_path):
    """Load and validate raw event logs"""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Raw log file is empty")
            
        required_columns = {'timestamp', 'user_id', 'event_type'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().any():
            raise ValueError("Invalid timestamp values found")
            
        print(f"Successfully loaded {len(df)} raw events")
        return df
    except Exception as e:
        print(f"Error loading raw data: {e}")
        sys.exit(1)

def calculate_sessions(df):
    """Calculate session metrics with robust error handling"""
    try:
        df = df.sort_values(['user_id', 'timestamp'])
        df['time_diff'] = df.groupby('user_id')['timestamp'].diff()
        df['new_session'] = (df['time_diff'] > pd.Timedelta(minutes=SESSION_INACTIVITY_MINUTES)) | (df['time_diff'].isna())
        df['session_id'] = df.groupby('user_id')['new_session'].cumsum()
        df['session_key'] = df['user_id'] + '_s' + df['session_id'].astype(str)
        
        session_metrics = df.groupby('session_key').agg(
            user_id=('user_id', 'first'),
            session_start=('timestamp', 'min'),
            session_end=('timestamp', 'max'),
            event_count=('event_type', 'count'),
            feature_a_used=('event_type', lambda x: (x == 'feature_A_used').sum()),
            feature_b_used=('event_type', lambda x: (x == 'feature_B_used').sum()),
            errors=('event_type', lambda x: (x == 'error_occurred').sum())
        ).reset_index()
        
        session_metrics['duration_minutes'] = (
            (session_metrics['session_end'] - session_metrics['session_start']).dt.total_seconds() / 60
        ).clip(lower=0.01)  # Ensure minimum duration
        
        return df, session_metrics
    except Exception as e:
        print(f"Error calculating sessions: {e}")
        return df, pd.DataFrame()

def calculate_daily_metrics(df):
    """Calculate daily summary metrics with comprehensive validation"""
    try:
        df['date'] = df['timestamp'].dt.date
        
        daily_metrics = df.groupby('date').agg(
            total_events=('event_type', 'count'),
            signups=('event_type', lambda x: (x == 'signup').sum()),
            logins=('event_type', lambda x: (x == 'login').sum()),
            errors=('event_type', lambda x: (x == 'error_occurred').sum()),
            cancellations=('event_type', lambda x: (x == 'subscription_cancelled').sum()),
            unique_users=('user_id', 'nunique'),
            unique_sessions=('session_key', 'nunique')
        ).reset_index()
        
        # Add feature usage metrics
        features = ['feature_A_used', 'feature_B_used']
        for feature in features:
            col_name = feature.replace('_used', '') + '_count'
            feature_counts = df[df['event_type'] == feature].groupby('date').size().reset_index(name=col_name)
            daily_metrics = pd.merge(daily_metrics, feature_counts, on='date', how='left')
            daily_metrics[col_name] = daily_metrics[col_name].fillna(0).astype(int)
        
        daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
        return daily_metrics
    except Exception as e:
        print(f"Error calculating daily metrics: {e}")
        return pd.DataFrame()

def detect_anomalies(daily_metrics):
    """Perform anomaly detection with robust validation"""
    try:
        if len(daily_metrics) < MIN_DAYS_FOR_ANOMALY_DETECTION:
            print("Insufficient data for anomaly detection")
            daily_metrics['is_anomaly'] = False
            return daily_metrics
            
        ts_data = daily_metrics.set_index('date')['signups']
        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
        residuals = decomposition.resid
        
        threshold = ANOMALY_STD_THRESHOLD * residuals.std()
        daily_metrics['is_anomaly'] = (residuals.abs() > threshold).reindex(daily_metrics['date']).fillna(False)
        
        print(f"Found {daily_metrics['is_anomaly'].sum()} anomalies")
        return daily_metrics
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        daily_metrics['is_anomaly'] = False
        return daily_metrics

def calculate_user_activity(df):
    """Calculate comprehensive user activity metrics"""
    try:
        user_activity = df.groupby('user_id').agg(
            first_seen=('timestamp', 'min'),
            last_seen=('timestamp', 'max'),
            total_events=('event_type', 'count'),
            total_sessions=('session_key', 'nunique'),
            days_active=('date', 'nunique'),
            feature_a_used=('event_type', lambda x: (x == 'feature_A_used').sum()),
            feature_b_used=('event_type', lambda x: (x == 'feature_B_used').sum()),
            has_cancelled=('event_type', lambda x: (x == 'subscription_cancelled').any()),
            has_upgraded=('event_type', lambda x: (x == 'plan_upgraded').any())
        ).reset_index()
        
        # Calculate derived metrics
        user_activity['days_since_last_seen'] = (df['timestamp'].max() - user_activity['last_seen']).dt.days.clip(lower=0)
        user_activity['potentially_churned'] = user_activity['days_since_last_seen'] > CHURN_DEFINITION_DAYS
        user_activity['tenure_days'] = (user_activity['last_seen'] - user_activity['first_seen']).dt.days.clip(lower=1)
        
        # Calculate rates and frequencies
        user_activity['events_per_day'] = user_activity['total_events'] / user_activity['days_active']
        user_activity['feature_a_freq'] = user_activity['feature_a_used'] / user_activity['tenure_days']
        user_activity['feature_b_freq'] = user_activity['feature_b_used'] / user_activity['tenure_days']
        
        # Handle potential division issues
        user_activity.replace([np.inf, -np.inf], 0, inplace=True)
        user_activity.fillna(0, inplace=True)
        
        return user_activity
    except Exception as e:
        print(f"Error calculating user activity: {e}")
        return pd.DataFrame()

def predict_churn_risk(user_activity):
    """Train churn prediction model with comprehensive validation"""
    try:
        if len(user_activity) < 50 or user_activity['potentially_churned'].nunique() < 2:
            print("Insufficient data for churn prediction")
            user_activity['churn_risk'] = 0
            return user_activity
            
        features = [
            'days_active', 'total_events', 'total_sessions',
            'feature_a_used', 'feature_b_used', 'has_upgraded',
            'events_per_day', 'feature_a_freq', 'feature_b_freq'
        ]
        
        X = user_activity[features]
        y = user_activity['potentially_churned'].astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        user_activity['churn_risk'] = model.predict_proba(X_scaled)[:, 1]
        
        # Evaluate model performance
        train_acc = accuracy_score(y, model.predict(X_scaled))
        print(f"Churn model trained (accuracy: {train_acc:.2f})")
        
        return user_activity
    except Exception as e:
        print(f"Error in churn prediction: {e}")
        user_activity['churn_risk'] = 0
        return user_activity

def segment_users(user_activity):
    """Perform user segmentation with robust validation"""
    try:
        if len(user_activity) < CLUSTERING_N_CLUSTERS * 10:
            print("Insufficient users for segmentation")
            user_activity['segment'] = 0
            return user_activity
            
        features = [
            'days_active', 'total_sessions', 'events_per_day',
            'feature_a_freq', 'feature_b_freq', 'churn_risk'
        ]
        
        X = user_activity[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(
            n_clusters=CLUSTERING_N_CLUSTERS,
            random_state=42,
            n_init='auto'
        )
        user_activity['segment'] = kmeans.fit_predict(X_scaled)
        
        print("User segmentation completed")
        return user_activity
    except Exception as e:
        print(f"Error in user segmentation: {e}")
        user_activity['segment'] = 0
        return user_activity

def calculate_cohort_retention(df, user_activity):
    """Calculate cohort retention with comprehensive validation"""
    try:
        if user_activity.empty or 'first_seen' not in user_activity.columns:
            return pd.DataFrame()
            
        user_activity['cohort_month'] = user_activity['first_seen'].dt.to_period('M')
        df['activity_month'] = df['timestamp'].dt.to_period('M')
        
        # Get unique user-month pairs
        monthly_active = df[['user_id', 'activity_month']].drop_duplicates()
        
        # Merge with cohort information
        cohort_data = pd.merge(
            monthly_active,
            user_activity[['user_id', 'cohort_month']],
            on='user_id',
            how='inner'
        )
        
        # Calculate months since cohort start
        cohort_data['months_since_cohort'] = (
            cohort_data['activity_month'] - cohort_data['cohort_month']
        ).apply(lambda x: x.n)
        
        # Calculate retention metrics
        cohort_counts = cohort_data.groupby(
            ['cohort_month', 'months_since_cohort']
        )['user_id'].nunique().reset_index()
        
        cohort_sizes = cohort_counts[
            cohort_counts['months_since_cohort'] == 0
        ][['cohort_month', 'user_id']].rename(columns={'user_id': 'cohort_size'})
        
        retention = pd.merge(cohort_counts, cohort_sizes, on='cohort_month')
        retention['retention_rate'] = retention['user_id'] / retention['cohort_size']
        
        # Pivot for visualization
        pivot_table = retention.pivot_table(
            index='cohort_month',
            columns='months_since_cohort',
            values='retention_rate'
        )
        
        return pivot_table
    except Exception as e:
        print(f"Error calculating cohort retention: {e}")
        return pd.DataFrame()

def calculate_funnel(df):
    """Calculate conversion funnel with comprehensive validation"""
    try:
        funnel_steps = [
            'signup', 'login', 'feature_A_used', 
            'feature_B_used', 'plan_upgraded'
        ]
        
        # Get unique users per step
        step_counts = []
        for step in funnel_steps:
            users = df[df['event_type'] == step]['user_id'].nunique()
            step_counts.append({'step': step, 'users': users})
        
        funnel = pd.DataFrame(step_counts)
        
        # Calculate conversion rates
        funnel['conversion_from_previous'] = funnel['users'] / funnel['users'].shift(1)
        funnel['overall_conversion'] = funnel['users'] / funnel['users'].iloc[0]
        
        return funnel
    except Exception as e:
        print(f"Error calculating funnel: {e}")
        return pd.DataFrame()

def generate_insights(daily_metrics, user_activity, funnel, cohort):
    """Generate automated insights with comprehensive validation"""
    insights = ["--- Automated Insights Report ---"]
    
    try:
        # Basic metrics
        insights.append(f"Period: {daily_metrics['date'].min().date()} to {daily_metrics['date'].max().date()}")
        insights.append(f"Total Users: {len(user_activity)}")
        insights.append(f"Total Sessions: {user_activity['total_sessions'].sum()}")
        
        # Signup trends
        recent_signups = daily_metrics['signups'].tail(7).mean()
        prev_signups = daily_metrics['signups'].iloc[-14:-7].mean()
        if prev_signups > 0:
            change = (recent_signups - prev_signups) / prev_signups * 100
            insights.append(f"Signups changed by {change:.1f}% week-over-week")
        
        # Churn insights
        churn_rate = user_activity['potentially_churned'].mean() * 100
        insights.append(f"Potential churn rate: {churn_rate:.1f}%")
        
        if 'churn_risk' in user_activity.columns:
            high_risk = (user_activity['churn_risk'] > 0.7).sum()
            insights.append(f"Users with high churn risk: {high_risk}")
        
        # Funnel insights
        if not funnel.empty:
            biggest_drop = funnel['conversion_from_previous'].idxmin()
            insights.append(
                f"Biggest drop: {funnel.loc[biggest_drop, 'step']} "
                f"({(1-funnel.loc[biggest_drop, 'conversion_from_previous'])*100:.1f}% drop)"
            )
        
        # Cohort insights
        if not cohort.empty and len(cohort) > 1:
            latest_cohort = cohort.index[-1]
            retention = cohort.loc[latest_cohort, 1] * 100
            insights.append(f"Latest cohort ({latest_cohort}): {retention:.1f}% retained after 1 month")
        
        return insights
    except Exception as e:
        print(f"Error generating insights: {e}")
        return insights + ["Error generating some insights"]

def save_outputs(paths, data):
    """Save all outputs with comprehensive error handling"""
    try:
        # Daily metrics
        if not data['daily_metrics'].empty:
            data['daily_metrics'].to_csv(paths['daily_summary'], index=False)
        
        # User activity
        if not data['user_activity'].empty:
            data['user_activity'].to_csv(paths['user_activity'], index=False)
        
        # Cohort retention
        if not data['cohort_retention'].empty:
            data['cohort_retention'].to_csv(paths['cohort_retention'])
        
        # Funnel
        if not data['funnel'].empty:
            data['funnel'].to_csv(paths['funnel_counts'], index=False)
        
        # Session metrics
        if not data['session_metrics'].empty:
            data['session_metrics'].to_csv(paths['session_metrics'], index=False)
        
        # Insights
        with open(paths['auto_insights'], 'w') as f:
            f.write("\n".join(data['insights']))
        
        print("All outputs saved successfully")
    except Exception as e:
        print(f"Error saving outputs: {e}")

def main():
    """Main execution flow"""
    print("Starting SaaS Metrics Pipeline")
    
    # Initialize paths
    paths = initialize_paths()
    
    # Load and validate raw data
    df = load_raw_data(paths['raw_logs'])
    
    # Calculate session metrics
    df, session_metrics = calculate_sessions(df)
    
    # Calculate daily metrics
    daily_metrics = calculate_daily_metrics(df)
    daily_metrics = detect_anomalies(daily_metrics)
    
    # Calculate user activity
    user_activity = calculate_user_activity(df)
    user_activity = predict_churn_risk(user_activity)
    user_activity = segment_users(user_activity)
    
    # Calculate cohort retention
    cohort_retention = calculate_cohort_retention(df, user_activity)
    
    # Calculate funnel
    funnel = calculate_funnel(df)
    
    # Generate insights
    insights = generate_insights(daily_metrics, user_activity, funnel, cohort_retention)
    
    # Prepare all outputs
    outputs = {
        'daily_metrics': daily_metrics,
        'user_activity': user_activity,
        'cohort_retention': cohort_retention,
        'funnel': funnel,
        'session_metrics': session_metrics,
        'insights': insights
    }
    
    # Save outputs
    save_outputs(paths, outputs)
    
    print("Pipeline completed successfully")

if __name__ == "__main__":
    main()