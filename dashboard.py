import streamlit as st
import pandas as pd
import os
from datetime import datetime, time as dtime
from io import BytesIO

# ----------------------------
# CONFIG
# ----------------------------
ATTENDANCE_DIR = "attendance"
STUDENTS_FILE = "students.csv"
AUTO_REFRESH_SEC = 8  # how often to refresh the dashboard (seconds)

# ----------------------------
# YOUR WEEKLY TIMETABLE (same structure you’re using in the main app)
# ----------------------------
TIMETABLE = {
    "Class A": {
        "Monday": [
            {"slot": "Slot 1", "start": "09:00", "end": "09:50"},
            {"slot": "Slot 2", "start": "10:05", "end": "10:55"},
            {"slot": "Slot 3", "start": "11:10", "end": "12:00"},
            {"slot": "Slot 4", "start": "13:00", "end": "13:50"},
            {"slot": "Slot 5", "start": "14:05", "end": "14:55"},
        ],
        "Tuesday": [
            {"slot": "Slot 1", "start": "09:00", "end": "09:50"},
            {"slot": "Slot 2", "start": "10:05", "end": "10:55"},
            {"slot": "Slot 3", "start": "11:10", "end": "12:00"},
            {"slot": "Slot 4", "start": "13:00", "end": "13:50"},
        ],
        "Wednesday": [
            {"slot": "Slot 1", "start": "09:00", "end": "09:50"},
            {"slot": "Slot 2", "start": "10:05", "end": "10:55"},
            {"slot": "Slot 3", "start": "11:10", "end": "12:00"},
            {"slot": "Slot 4", "start": "13:00", "end": "13:50"},
            {"slot": "Slot 5", "start": "14:05", "end": "14:55"},
            {"slot": "Slot 6", "start": "15:10", "end": "16:00"},
        ],
        "Thursday": [
            {"slot": "Slot 1", "start": "09:00", "end": "09:50"},
            {"slot": "Slot 2", "start": "10:05", "end": "10:55"},
            {"slot": "Slot 3", "start": "11:10", "end": "12:00"},
            {"slot": "Slot 4", "start": "13:00", "end": "13:50"},
            {"slot": "Slot 5", "start": "14:05", "end": "14:55"},
        ],
        "Friday": [
            {"slot": "Slot 1", "start": "09:00", "end": "09:50"},
            {"slot": "Slot 2", "start": "10:05", "end": "10:55"},
            {"slot": "Slot 3", "start": "11:10", "end": "12:00"},
            {"slot": "Slot 4", "start": "13:00", "end": "13:50"},
        ],
    },
    # Copy the same structure for Class B/C/D/E as you used in your app.
    # For brevity here we’ll reuse Class A’s schedule:
    "Class B": {}, "Class C": {}, "Class D": {}, "Class E": {}
}
for c in ["Class B", "Class C", "Class D", "Class E"]:
    TIMETABLE[c] = TIMETABLE["Class A"]

# ----------------------------
# HELPERS
# ----------------------------
def today_file() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

def load_students() -> pd.DataFrame:
    if not os.path.exists(STUDENTS_FILE) or os.stat(STUDENTS_FILE).st_size == 0:
        return pd.DataFrame(columns=["Roll", "Name", "Class"])
    df = pd.read_csv(STUDENTS_FILE)
    # Backward compatibility: if Class column missing, add Unknown
    if "Class" not in df.columns:
        df["Class"] = "Unknown"
    return df

def load_today_attendance() -> pd.DataFrame:
    path = today_file()
    if not os.path.exists(path):
        # Empty shell with slots added dynamically by your app—fallback:
        return pd.DataFrame(columns=["Roll", "Name", "Class"])
    return pd.read_csv(path)

def parse_hhmm(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(hour=h, minute=m)

def current_slot_for_class(class_name: str) -> str | None:
    weekday = datetime.now().strftime("%A")  # e.g., "Monday"
    slots = TIMETABLE.get(class_name, {}).get(weekday, [])
    now_t = datetime.now().time()
    for block in slots:
        if parse_hhmm(block["start"]) <= now_t <= parse_hhmm(block["end"]):
            return block["slot"]
    return None

def absent_list(df_students: pd.DataFrame, df_att: pd.DataFrame, class_name: str, slot_col: str) -> pd.DataFrame:
    # merge so we have all students in class, even if not in attendance file yet
    class_roster = df_students[df_students["Class"] == class_name][["Roll", "Name", "Class"]].copy()
    df = class_roster.merge(df_att, on=["Roll", "Name", "Class"], how="left")
    if slot_col not in df.columns:
        # If slot column not yet created, treat all as Absent
        df[slot_col] = "Absent"
    absentees = df[df[slot_col].fillna("Absent") != "Present"][["Roll", "Name"]].sort_values("Roll")
    return absentees

def make_absent_excel(absentees: pd.DataFrame, class_name: str, slot_col: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        absentees.to_excel(writer, index=False, sheet_name=f"{class_name}-{slot_col}")
    return output.getvalue()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="FaceMark 360 • Live Dashboard", layout="wide")
st.title("📊 Live Attendance Dashboard — FaceMark 360")

# Auto-refresh
st.caption(f"Auto-refreshing every {AUTO_REFRESH_SEC} seconds")
st.experimental_set_query_params(refresh=str(datetime.now().timestamp()))
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=8000, key="dashboard_refresh")

st.title("📊 Live Attendance Dashboard — FaceMark 360")
st.caption("Auto-refreshing every 8 seconds")
st.button("🔄 Refresh now")

# Load data
students_df = load_students()
attendance_df = load_today_attendance()

# Sidebar: pick class
classes = sorted(students_df["Class"].dropna().unique().tolist())
if not classes:
    st.warning("No classes found in students.csv. Please register students with a Class.")
    st.stop()

cls = st.sidebar.selectbox("Select Class", classes, index=0)

# Resolve current slot for that class
slot = current_slot_for_class(cls)
col_slot = slot if slot else "(No active slot)"
st.subheader(f"Class: **{cls}**  •  Current Slot: **{col_slot}**")

# Metrics
if slot:
    # Filter roster & compute metrics
    roster = students_df[students_df["Class"] == cls]
    total = len(roster)
    if slot not in attendance_df.columns:
        present = 0
    else:
        present = attendance_df[
            (attendance_df["Class"] == cls) & (attendance_df[slot] == "Present")
        ].shape[0]
    absent = total - present

    m1, m2, m3 = st.columns(3)
    m1.metric("Present", present)
    m2.metric("Absent", absent)
    m3.metric("Total", total)

    # Absent list
    st.markdown("### 🚫 Absent Students")
    abs_df = absent_list(students_df, attendance_df, cls, slot)
    st.dataframe(abs_df, use_container_width=True, height=320)

    # Download button (Excel)
    xlsx_bytes = make_absent_excel(abs_df, cls, slot)
    st.download_button(
        "⬇️ Download Absent List (Excel)",
        data=xlsx_bytes,
        file_name=f"Absent_{cls}_{slot}_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("No active slot for this class right now (based on timetable).")

# Light auto-refresh using empty placeholder + timer
ph = st.empty()
ph.info("This dashboard will auto-refresh.")
st.stop()
