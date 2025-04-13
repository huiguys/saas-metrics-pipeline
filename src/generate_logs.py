# generate_logs.py

import pandas as pd
import random
from datetime import datetime, timedelta
import os # To ensure file is saved in the script's directory

# --- Configuration ---
NUM_USERS = 150 # Increased slightly for more data
NUM_EVENTS = 7500 # Increased slightly
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 4, 12) # Use recent end date closer to now
OUTPUT_FILENAME = 'raw_event_logs.csv'

USER_IDS = [f'user_{i}' for i in range(NUM_USERS)]
# More diverse events
EVENT_TYPES = ['signup', 'login', 'feature_A_used', 'feature_B_used', 'profile_update', 'search_performed', 'help_accessed', 'session_end', 'error_occurred', 'subscription_cancelled', 'plan_upgraded']
EVENT_WEIGHTS = [3, 30, 15, 10, 5, 10, 4, 15, 3, 2, 3] # Adjusted weights

# Ensure weights match event types length (excluding signup for weighted choice later)
if len(EVENT_TYPES) != len(EVENT_WEIGHTS):
    raise ValueError("Length of EVENT_TYPES and EVENT_WEIGHTS must be the same.")

events_data = []
script_dir = os.path.dirname(__file__) # Get directory where script is running
output_filepath = os.path.join(script_dir, OUTPUT_FILENAME)

# Simulate user signups first
signup_dates = {}
print("Simulating user signups...")
for user_id in USER_IDS:
    signup_time = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds() * 0.8))) # Signups happen earlier typically
    signup_dates[user_id] = signup_time
    events_data.append({
        'timestamp': signup_time.strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': user_id,
        'event_type': 'signup',
        'details': f'Signed up via {random.choice(["web", "mobile"])}'
    })

# Simulate other events
print(f"Simulating remaining {NUM_EVENTS - len(events_data)} events...")
generated_events_count = len(events_data)
event_types_excluding_signup = EVENT_TYPES[1:]
event_weights_excluding_signup = EVENT_WEIGHTS[1:]

while generated_events_count < NUM_EVENTS:
    user_id = random.choice(USER_IDS)
    user_signup_date = signup_dates[user_id]

    # Ensure event happens *after* signup but within the date range
    max_seconds_offset = (END_DATE - user_signup_date).total_seconds()
    if max_seconds_offset <= 1: continue # Skip if signup is too close to end date

    event_time = user_signup_date + timedelta(seconds=random.randint(1, int(max_seconds_offset)))

    # Choose event type based on weights
    event_type = random.choices(event_types_excluding_signup, weights=event_weights_excluding_signup, k=1)[0]

    details = ''
    # Add specific logic based on event type
    if event_type == 'feature_A_used':
        details = f'Used sub-feature {random.choice(["X", "Y", "Z"])}'
    elif event_type == 'error_occurred':
        details = f'Code {random.choice([400, 401, 404, 500, 503])}'
    elif event_type == 'search_performed':
        details = f'Searched for {random.choice(["billing", "integration", "tutorial"])}'
    elif event_type == 'subscription_cancelled':
         if (event_time - user_signup_date).days < 20: continue # Prevent immediate cancellation
         if random.random() > 0.1: continue # Make cancellation relatively rare
         details = f'Reason code {random.randint(1, 5)}'
    elif event_type == 'plan_upgraded':
         if (event_time - user_signup_date).days < 5: continue # Prevent immediate upgrade
         if random.random() > 0.15: continue # Make upgrade relatively rare
         details = 'Upgraded to Pro tier'

    events_data.append({
        'timestamp': event_time.strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': user_id,
        'event_type': event_type,
        'details': details
    })
    generated_events_count += 1
    if generated_events_count % 1000 == 0:
         print(f"...generated {generated_events_count} events...")


# Create DataFrame and save to CSV
df_events = pd.DataFrame(events_data)
df_events = df_events.sort_values(by='timestamp') # Sort logs chronologically

# Save the raw logs
df_events.to_csv(output_filepath, index=False)
print(f"\nGenerated {len(df_events)} events and saved to {output_filepath}")