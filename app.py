"""Flask web app for focus score display."""

import datetime
import json
import os
import pathlib
import secrets

from flask import Flask, redirect, request, session, url_for, render_template
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from calendar_loader import fetch_events, events_to_schedule_rows, fill_gaps
from focus import (
    build_activity_map,
    build_resolved_activity_map,
    load_schedule_categories,
    compute_focus_from_categories,
    build_classifier,
)

APP_DIR = pathlib.Path(__file__).parent.resolve()

app = Flask(__name__, template_folder=APP_DIR / "templates")
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CLIENT_SECRETS_FILE = APP_DIR / "credentials-web.json"

# Allow HTTP for local development only
if os.environ.get("FLASK_ENV") != "production":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


def get_client_config():
    """Get OAuth client config from env vars or file."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if client_id and client_secret:
        return {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

    with open(CLIENT_SECRETS_FILE) as f:
        return json.load(f)


def get_flow(redirect_uri):
    """Create OAuth flow for web app."""
    client_config = get_client_config()
    return Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )


def credentials_to_dict(credentials):
    """Convert credentials to dict for session storage."""
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }


def get_calendar_service():
    """Build calendar service from session credentials."""
    if "credentials" not in session:
        return None
    creds = Credentials(**session["credentials"])
    return build("calendar", "v3", credentials=creds)


def get_week_dates():
    """Get dates for the current week (Monday to Sunday)."""
    today = datetime.date.today()
    monday = today - datetime.timedelta(days=today.weekday())
    return [monday + datetime.timedelta(days=i) for i in range(7)]


def get_month_calendar(year, month):
    """Get calendar grid for a month (list of weeks, each week is list of dates or None)."""
    import calendar
    cal = calendar.Calendar(firstweekday=0)  # Monday first
    weeks = []
    for week in cal.monthdatescalendar(year, month):
        weeks.append(week)
    return weeks


def compute_focus_for_date(service, date, classifier=None, goals_context=None):
    """Compute focus score for a single date."""
    activity_map = build_activity_map()

    try:
        events = fetch_events(service, date, "primary")
    except Exception as e:
        print(f"Error fetching events for {date}: {e}")
        return None

    rows = events_to_schedule_rows(events)
    rows = fill_gaps(rows, "unscheduled")

    activity_map = build_resolved_activity_map(
        rows,
        activity_map,
        classifier=classifier,
        goals_context=goals_context,
    )
    categories = load_schedule_categories(rows, activity_map)
    result = compute_focus_from_categories(categories)
    return result["focus"]


@app.route("/")
def index():
    """Landing page with focus button or month calendar view."""
    if "credentials" not in session:
        return render_template("index.html", authenticated=False)

    service = get_calendar_service()
    if not service:
        return render_template("index.html", authenticated=False)

    classifier = build_classifier()
    classifier.call_count = 0  # Track LLM calls
    goals_context = session.get("goals_context", "Be productive and focused")

    today = datetime.date.today()

    # Get month/year from query params or use current
    year = request.args.get("year", today.year, type=int)
    month = request.args.get("month", today.month, type=int)

    # Handle month overflow
    if month < 1:
        month = 12
        year -= 1
    elif month > 12:
        month = 1
        year += 1

    weeks = get_month_calendar(year, month)

    # Load cached data from session
    activity_map = build_activity_map()
    cached_mappings = session.get("activity_mappings", {})
    cached_scores = session.get("cached_scores", {})  # "YYYY-MM-DD" -> score
    activity_map.update(cached_mappings)

    # Check which days need computing (not cached or is today)
    days_to_compute = []
    for week in weeks:
        for date in week:
            if date <= today and date.month == month:
                date_key = date.isoformat()
                # Always recompute today, use cache for past days
                if date == today or date_key not in cached_scores:
                    days_to_compute.append(date)

    # PASS 1: Fetch events only for days we need to compute
    all_activities = set()
    day_events = {}  # date -> rows

    for date in days_to_compute:
        try:
            events = fetch_events(service, date, "primary")
            rows = events_to_schedule_rows(events)
            rows = fill_gaps(rows, "unscheduled")
            day_events[date] = rows
            for _, _, activity in rows:
                if activity not in activity_map:
                    all_activities.add(activity)
        except Exception as e:
            print(f"Error fetching events for {date}: {e}")
            day_events[date] = []

    # PASS 2: Batch classify only NEW unknown activities
    if all_activities and classifier:
        classified = classifier.classify_many(list(all_activities), goals_context)
        activity_map.update(classified)
        cached_mappings.update(classified)
        session["activity_mappings"] = cached_mappings

    # PASS 3: Compute scores for days we fetched, use cache for rest
    for date, rows in day_events.items():
        if rows:
            categories = load_schedule_categories(rows, activity_map)
            result = compute_focus_from_categories(categories)
            cached_scores[date.isoformat()] = result["focus"]
    session["cached_scores"] = cached_scores

    # Build calendar weeks with scores
    calendar_weeks = []
    for week in weeks:
        week_data = []
        for date in week:
            day_info = {
                "date": date,
                "day": date.day,
                "is_current_month": date.month == month,
                "is_future": date > today,
                "is_today": date == today,
                "score": None,
            }
            if date <= today and date.month == month:
                date_key = date.isoformat()
                day_info["score"] = cached_scores.get(date_key)
            week_data.append(day_info)
        calendar_weeks.append(week_data)

    month_name = datetime.date(year, month, 1).strftime("%B %Y")
    prev_month = {"year": year if month > 1 else year - 1, "month": month - 1 if month > 1 else 12}
    next_month = {"year": year if month < 12 else year + 1, "month": month + 1 if month < 12 else 1}

    return render_template(
        "index.html",
        authenticated=True,
        weeks=calendar_weeks,
        month_name=month_name,
        prev_month=prev_month,
        next_month=next_month,
        today=today,
        llm_calls=classifier.call_count,
        days_computed=len(days_to_compute),
    )


@app.route("/auth")
def auth():
    """Start OAuth flow."""
    flow = get_flow(url_for("oauth_callback", _external=True))
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
    )
    session["state"] = state
    return redirect(authorization_url)


@app.route("/oauth/callback")
def oauth_callback():
    """Handle OAuth callback."""
    flow = get_flow(url_for("oauth_callback", _external=True))
    flow.fetch_token(authorization_response=request.url)
    session["credentials"] = credentials_to_dict(flow.credentials)
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    """Clear session and log out."""
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
