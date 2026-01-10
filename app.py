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
    """Landing page with focus button or week view."""
    if "credentials" not in session:
        return render_template("index.html", authenticated=False)

    service = get_calendar_service()
    if not service:
        return render_template("index.html", authenticated=False)

    classifier = build_classifier()
    goals_context = session.get("goals_context", "Be productive and focused")

    today = datetime.date.today()
    week_dates = get_week_dates()

    days = []
    for date in week_dates:
        day_info = {
            "date": date,
            "weekday": date.strftime("%a"),
            "is_future": date > today,
            "is_today": date == today,
            "score": None,
        }
        if date <= today:
            day_info["score"] = compute_focus_for_date(
                service, date, classifier, goals_context
            )
        days.append(day_info)

    return render_template("index.html", authenticated=True, days=days)


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
