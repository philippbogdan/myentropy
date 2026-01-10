"""Google Calendar loader with desktop OAuth flow."""

import datetime
import os
import pickle

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
DEFAULT_CREDENTIALS_PATH = "credentials.json"
DEFAULT_TOKEN_PATH = "token.json"

EXCLUDED_CALENDARS = [
    "holiday@group.v.calendar.google.com",  # Skip holiday calendars by default
]


def get_credentials(credentials_path=None, token_path=None):
    """Get valid credentials, refreshing or prompting login as needed."""
    credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
    token_path = token_path or DEFAULT_TOKEN_PATH

    creds = None
    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Missing {credentials_path}. Download OAuth credentials from "
                    "Google Cloud Console and save as credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return creds


def get_calendar_service(credentials_path=None, token_path=None):
    """Build and return a Google Calendar API service."""
    creds = get_credentials(credentials_path, token_path)
    return build("calendar", "v3", credentials=creds)


def list_calendars(service, exclude_patterns=None):
    """List all calendars the user has access to.

    Args:
        service: Google Calendar API service
        exclude_patterns: List of substrings to exclude (e.g., 'holiday')

    Returns:
        List of dicts with 'id' and 'summary' for each calendar
    """
    exclude_patterns = exclude_patterns or EXCLUDED_CALENDARS
    calendars = service.calendarList().list().execute()

    result = []
    for cal in calendars.get("items", []):
        cal_id = cal.get("id", "")
        skip = any(pattern in cal_id for pattern in exclude_patterns)
        if not skip:
            result.append({
                "id": cal_id,
                "summary": cal.get("summary", "Unnamed"),
            })
    return result


def fetch_events(service, date=None, calendar_id="primary"):
    """Fetch all events for a given date.

    Args:
        service: Google Calendar API service
        date: datetime.date object (defaults to today)
        calendar_id: Calendar ID (defaults to primary)

    Returns:
        List of event dicts from the API
    """
    if date is None:
        date = datetime.date.today()

    start_of_day = datetime.datetime.combine(date, datetime.time.min)
    end_of_day = datetime.datetime.combine(date, datetime.time.max)

    time_min = start_of_day.isoformat() + "Z"
    time_max = end_of_day.isoformat() + "Z"

    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    return events_result.get("items", [])


def fetch_all_events(service, date=None, calendar_ids=None):
    """Fetch events from multiple calendars for a given date.

    Args:
        service: Google Calendar API service
        date: datetime.date object (defaults to today)
        calendar_ids: List of calendar IDs (defaults to primary calendar)

    Returns:
        List of event dicts from all calendars
    """
    if calendar_ids is None:
        calendar_ids = ["primary"]

    all_events = []
    for cal_id in calendar_ids:
        try:
            events = fetch_events(service, date, cal_id)
            all_events.extend(events)
        except Exception as e:
            print(f"Warning: Could not fetch from calendar {cal_id}: {e}")

    return all_events


def parse_event_time(event, key):
    """Parse start or end time from an event to minutes since midnight.

    Args:
        event: Event dict from API
        key: 'start' or 'end'

    Returns:
        Minutes since midnight (0-1440)
    """
    time_info = event.get(key, {})

    if "dateTime" in time_info:
        dt = datetime.datetime.fromisoformat(time_info["dateTime"].replace("Z", "+00:00"))
        dt_local = dt.astimezone()
        minutes = dt_local.hour * 60 + dt_local.minute
    elif "date" in time_info:
        if key == "start":
            minutes = 0
        else:
            minutes = 1440
    else:
        raise ValueError(f"Event missing {key} time: {event}")

    return max(0, min(1440, minutes))


def events_to_schedule_rows(events):
    """Convert Google Calendar events to schedule rows.

    Args:
        events: List of event dicts from API

    Returns:
        List of tuples: (start_minute, end_minute, activity_label)
    """
    rows = []
    for event in events:
        start = parse_event_time(event, "start")
        end = parse_event_time(event, "end")

        if start >= end:
            continue

        summary = event.get("summary", "untitled").strip()
        if not summary:
            summary = "untitled"

        rows.append((start, end, summary.lower()))

    rows.sort(key=lambda x: x[0])
    return rows


def merge_overlapping_rows(rows):
    """Merge overlapping events, keeping the first one for conflicts.

    Args:
        rows: Sorted list of (start, end, activity) tuples

    Returns:
        Non-overlapping list of schedule rows
    """
    if not rows:
        return []

    merged = []
    current_end = 0

    for start, end, activity in rows:
        if start >= current_end:
            merged.append((start, end, activity))
            current_end = end
        elif end > current_end:
            merged.append((current_end, end, activity))
            current_end = end

    return merged


def fill_gaps(rows, gap_label="unscheduled"):
    """Fill gaps in schedule with a default label.

    Args:
        rows: Sorted list of (start, end, activity) tuples
        gap_label: Label to use for unscheduled time

    Returns:
        Complete schedule covering 0-1440 minutes
    """
    rows = merge_overlapping_rows(rows)
    filled = []
    current = 0

    for start, end, activity in rows:
        if start > current:
            filled.append((current, start, gap_label))

        if start < current:
            start = current

        if start < end:
            filled.append((start, end, activity))
            current = end

    if current < 1440:
        filled.append((current, 1440, gap_label))

    return filled


def load_calendar_schedule(date=None, credentials_path=None, token_path=None,
                           calendar_ids=None, fill_gaps_label="unscheduled"):
    """Load a day's schedule from Google Calendar (all calendars aggregated).

    Args:
        date: datetime.date object (defaults to today)
        credentials_path: Path to OAuth credentials JSON
        token_path: Path to store/load token
        calendar_ids: List of calendar IDs (defaults to all non-excluded calendars)
        fill_gaps_label: Label for unscheduled time gaps

    Returns:
        List of tuples: (start_minute, end_minute, activity_label)
        covering the full 24-hour day (0-1440 minutes)
    """
    service = get_calendar_service(credentials_path, token_path)
    events = fetch_all_events(service, date, calendar_ids)
    rows = events_to_schedule_rows(events)
    return fill_gaps(rows, fill_gaps_label)


if __name__ == "__main__":
    import sys

    date = None
    if len(sys.argv) > 1:
        date = datetime.date.fromisoformat(sys.argv[1])

    print(f"Fetching calendar for {date or 'today'}...")

    try:
        schedule = load_calendar_schedule(date=date)
        print(f"\nFound {len(schedule)} time blocks:\n")

        for start, end, activity in schedule:
            start_h, start_m = divmod(start, 60)
            end_h, end_m = divmod(end, 60)
            print(f"  {start_h:02d}:{start_m:02d} - {end_h:02d}:{end_m:02d}  {activity}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
