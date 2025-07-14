import google.auth
import google.auth.transport.requests
import requests
import numpy as np
import pickle
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Pickle file for storing data
SPARK_APPS_DATA_PICKLE = 'spark_apps_data.pkl'


def save_data(data):
    """Saves data to a pickle file."""
    with open(SPARK_APPS_DATA_PICKLE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {SPARK_APPS_DATA_PICKLE}")


def load_data():
    """Loads data from a pickle file if it exists."""
    if os.path.exists(SPARK_APPS_DATA_PICKLE):
        with open(SPARK_APPS_DATA_PICKLE, 'rb') as f:
            print(f"Loading existing data from {SPARK_APPS_DATA_PICKLE}")
            return pickle.load(f)
    return {}


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def make_authenticated_request(url, headers):
    """Makes an authenticated GET request with retry logic."""
    print(f"Attempting to fetch data from {url}...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    print("Fetch successful.")
    return response.json()

def get_spark_apps_dataproc(history_server_url, app_name):
    """
    Queries a Dataproc Spark History Server to get a list of completed
    applications, including their duration and executor count.
    """
    try:
        # Authenticate using Application Default Credentials (ADC)
        credentials, _ = google.auth.default()
        authed_session = google.auth.transport.requests.Request()
        credentials.refresh(authed_session)

        if not history_server_url.endswith('/'):
            history_server_url += '/'

        api_url = f"{history_server_url}api/v1/applications"
        headers = {'Authorization': f'Bearer {credentials.token}'}


        response = requests.get(api_url+"?limit=500&status=completed", headers=headers)
        response.raise_for_status()
        apps = response.json()

        completed_apps = []
        for app in apps:
            # We only analyze completed jobs with a valid duration
            is_completed = app['attempts'][0].get('completed', False)
            if app['name'] == app_name and is_completed:
                app_id = app['id']
                start_time_ms = app['attempts'][0]['startTimeEpoch']
                end_time_ms = app['attempts'][0]['endTimeEpoch']
                duration_sec = (end_time_ms - start_time_ms) / 1000
                # app['attempts'][0]['duration'] / 1000

                # Skip jobs with invalid duration
                if duration_sec <= 0:
                    print(f"Skipping job with invalid duration: {app_id}")
                    continue

                # Get executor information for the application
                executors = make_authenticated_request(f"{api_url}/{app_id}/1/allexecutors", headers=headers)

                # Subtract 1 for the driver to get the executor count
                num_executors = len(executors) - 1 if len(executors) > 0 else 0

                completed_apps.append({
                    'executors': num_executors,
                    'duration_sec': duration_sec
                })
            else:
                print(f"Skipping job: {app['name']}")
        return completed_apps

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Spark History Server: {e}")
        return []
    except google.auth.exceptions.DefaultCredentialsError:
        print("Authentication failed. Please run 'gcloud auth application-default login'")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_all_spark_apps_dataproc(history_server_url):
    """
    Queries a Dataproc Spark History Server to get a list of all completed
    applications, including their duration and executor count.
    Resumes from previously saved data if available.
    """
    # Load existing data
    completed_apps_data = load_data()

    try:
        credentials, _ = google.auth.default()
        authed_session = google.auth.transport.requests.Request()
        credentials.refresh(authed_session)

        if not history_server_url.endswith('/'):
            history_server_url += '/'

        api_url = f"{history_server_url}api/v1/applications"
        headers = {'Authorization': f'Bearer {credentials.token}'}

        response = requests.get(api_url + "?status=completed", headers=headers)
        response.raise_for_status()
        all_apps = response.json()

        for app in all_apps:
            app_id = app['id']
            if app_id in completed_apps_data:
                print(f"Skipping already processed app: {app_id}")
                continue

            is_completed = app['attempts'][0].get('completed', False)
            if is_completed:
                start_time_ms = app['attempts'][0]['startTimeEpoch']
                end_time_ms = app['attempts'][0]['endTimeEpoch']
                duration_sec = (end_time_ms - start_time_ms) / 1000

                if duration_sec <= 0:
                    print(f"Skipping job with invalid duration: {app_id}")
                    continue

                try:
                    executors = make_authenticated_request(f"{api_url}/{app_id}/1/allexecutors", headers=headers)
                    num_executors = len(executors) - 1 if len(executors) > 0 else 0

                    completed_apps_data[app_id] = {
                        'app_name': app['name'],
                        'executors': num_executors,
                        'duration_sec': duration_sec
                    }
                except requests.exceptions.RequestException as e:
                    print(f"Failed to fetch executors for {app_id}: {e}")
                    print("Saving progress before exiting...")
                    save_data(completed_apps_data)
                    raise  # Re-raise the exception to stop the script

        # Save all data upon successful completion
        save_data(completed_apps_data)
        return list(completed_apps_data.values())

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Spark History Server: {e}")
        return list(completed_apps_data.values())
    except google.auth.exceptions.DefaultCredentialsError:
        print("Authentication failed. Please run 'gcloud auth application-default login'")
        return list(completed_apps_data.values())
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return list(completed_apps_data.values())


def analyze_by_duration_detailed(applications):
    """
    Groups applications by runtime duration and prints detailed statistical
    analysis of executor usage for each group.
    """
    if not applications:
        print(f"\n❌ No completed applications found to analyze.")
        return

    durations = [app['duration_sec'] for app in applications]

    if not durations:
        print(f"\n❌ No jobs with a valid duration found.")
        return

    # Define duration percentiles
    percentiles = np.arange(10, 101, 10)
    duration_thresholds = np.percentile(durations, percentiles)

    # Create buckets
    buckets = {}
    lower_bound = 0
    for i, p in enumerate(percentiles):
        upper_bound = duration_thresholds[i]
        buckets[f"{p-10}-{p}th percentile ({lower_bound:.0f}s - {upper_bound:.0f}s)"] = (lower_bound, upper_bound)
        lower_bound = upper_bound

    # Group applications into buckets
    grouped_apps = {name: [] for name in buckets}
    for app in applications:
        for name, (lower_bound, upper_bound) in buckets.items():
            if lower_bound < app['duration_sec'] <= upper_bound:
                grouped_apps[name].append(app)
                break

    # Analysis
    print(f"\n📊 Detailed Executor Usage Analysis by Job Duration")
    print("=" * 80)

    for name, apps_in_bucket in grouped_apps.items():
        print(f"\n--- {name} ---")
        if not apps_in_bucket:
            print("No jobs found in this duration range.")
            continue

        exec_list = [app['executors'] for app in apps_in_bucket]
        exec_np = np.array(exec_list)
        job_count = len(exec_np)

        print(f"  {'Total Jobs':<25}: {job_count}")

        if job_count > 0:
            # Executor percentiles
            exec_percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
            for p in exec_percentiles:
                percentile_value = np.percentile(exec_np, p)
                print(f"  {f'{p}th Percentile Executors':<25}: {percentile_value:.2f}")


def analyze_by_duration(applications):
    """
    Groups applications by runtime duration and prints statistical
    analysis of executor usage for each group.
    """
    if not applications:
        print(f"\n❌ No completed applications found to analyze.")
        return

    durations = [app['duration_sec'] for app in applications]

    if not durations:
        print(f"\n❌ No jobs with a valid duration found.")
        return

    # Step 1: Determine the best time ranges using quartiles
    p25, p50, p75 = np.percentile(durations, [25, 50, 75])
    max_duration = np.max(durations)

    # Define the duration buckets based on the calculated quartiles
    # (min -> 25%, 25% -> 50%, etc.)
    buckets = {
        f"Short (0 - {p25:.0f}s)": (0, p25),
        f"Medium ({p25:.0f}s - {p50:.0f}s)": (p25, p50),
        f"Long ({p50:.0f}s - {p75:.0f}s)": (p50, p75),
        f"Very Long ({p75:.0f}s+)": (p75, max_duration + 1)
    }

    # Step 2: Group the executor counts into these new buckets
    grouped_executors = {name: [] for name in buckets}
    for app in applications:
        for name, (lower_bound, upper_bound) in buckets.items():
            if lower_bound < app['duration_sec'] <= upper_bound:
                grouped_executors[name].append(app['executors'])
                break

    # Step 3: Print the analysis for each group
    print(f"\n📊 Executor Usage Analysis by Job Duration for")
    print("=" * 70)

    for name, exec_list in grouped_executors.items():
        print(f"\n--- {name} ---")
        if not exec_list:
            print("No jobs found in this duration range.")
            continue

        exec_np = np.array(exec_list)
        job_count = len(exec_np)
        mean_execs = np.mean(exec_np)
        median_execs = np.median(exec_np)
        p90_execs = np.percentile(exec_np, 90)
        p95_execs = np.percentile(exec_np, 95)

        print(f"  {'Total Jobs':<25}: {job_count}")
        print(f"  {'Mean Executors':<25}: {mean_execs:.2f}")
        print(f"  {'Median Executors':<25}: {median_execs:.2f}")
        print(f"  {'90th Percentile Executors':<25}: {p90_execs:.2f}")
        print(f"  {'95th Percentile Executors':<25}: {p95_execs:.2f}")


if __name__ == "__main__":
    history_server_url = "https://pxc633usabc3rjx6gfrazwspm4-dot-europe-west1.dataproc.googleusercontent.com/sparkhistory/"

    # Get the raw application data, resuming if possible
    all_completed_apps = get_all_spark_apps_dataproc(history_server_url)

    # Analyze and print the grouped statistics
    if all_completed_apps:
        analyze_by_duration_detailed(all_completed_apps)




