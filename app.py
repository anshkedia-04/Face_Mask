# import streamlit as st
# st.set_page_config(layout="centered")

# import os
# import pickle
# import numpy as np
# import pandas as pd
# import cv2
# import torch
# from datetime import datetime
# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from sklearn.metrics.pairwise import cosine_similarity
# import time

# # =========================
# # Timetable for each class
# # =========================
# timetable = {
#     "Class A": {
#         "Monday": [
#             {"slot": "Slot 1", "start": "09:00", "end": "10:00"},
#             {"slot": "Slot 2", "start": "10:15", "end": "11:15"},
#             {"slot": "Slot 3", "start": "11:30", "end": "12:30"},
#             {"slot": "Slot 4", "start": "13:30", "end": "14:30"},
#             {"slot": "Slot 5", "start": "14:45", "end": "15:45"}
#         ],
#         "Tuesday": [
#             {"slot": "Slot 1", "start": "09:00", "end": "10:00"},
#             {"slot": "Slot 2", "start": "10:15", "end": "11:15"},
#             {"slot": "Slot 3", "start": "11:30", "end": "12:30"},
#             {"slot": "Slot 4", "start": "13:30", "end": "14:30"}
#         ],
#         "Wednesday": [
#             {"slot": "Slot 1", "start": "09:00", "end": "10:00"},
#             {"slot": "Slot 2", "start": "10:15", "end": "11:15"},
#             {"slot": "Slot 3", "start": "11:30", "end": "12:30"},
#             {"slot": "Slot 4", "start": "13:30", "end": "14:30"},
#             {"slot": "Slot 5", "start": "14:45", "end": "15:45"},
#             {"slot": "Slot 6", "start": "16:00", "end": "17:00"}
#         ],
#         "Thursday": [
#             {"slot": "Slot 1", "start": "09:00", "end": "10:00"},
#             {"slot": "Slot 2", "start": "10:15", "end": "11:15"},
#             {"slot": "Slot 3", "start": "11:30", "end": "12:30"},
#             {"slot": "Slot 4", "start": "13:30", "end": "14:30"}
#         ],
#         "Saturday": [
#             {"slot": "Slot 1", "start": "09:00", "end": "10:00"},
#             {"slot": "Slot 2", "start": "10:15", "end": "11:15"},
#             {"slot": "Slot 3", "start": "11:30", "end": "12:30"},
#             {"slot": "Slot 4", "start": "13:30", "end": "14:30"},
#             {"slot": "Slot 5", "start": "14:45", "end": "15:45"}
#         ]
#     },
    
#     # Similarly for other classes
#     "Class B": {
#         "Monday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"}
#         ],
#         "Tuesday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"},
#             {"slot": "Slot 6", "start": "15:30", "end": "16:30"}
#         ],
#         "Wednesday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"}
#         ],
#         "Thursday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"}
#         ],
#         "Friday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"},
#             {"slot": "Slot 6", "start": "15:30", "end": "16:30"}
#         ]
#     },
#     "Class C": {
#         "Monday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"}
#         ],
#         "Tuesday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"},
#             {"slot": "Slot 6", "start": "15:30", "end": "16:30"}
#         ],
#         "Wednesday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"}
#         ],
#         "Thursday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"}
#         ],
#         "Friday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"},
#             {"slot": "Slot 6", "start": "15:30", "end": "16:30"}
#         ]
#     },
#     "Class D": {
#         "Monday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"}
#         ],
#         "Tuesday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"},
#             {"slot": "Slot 6", "start": "15:30", "end": "16:30"}
#         ],
#         "Wednesday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"}
#         ],
#         "Thursday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"}
#         ],
#         "Friday": [
#             {"slot": "Slot 1", "start": "08:30", "end": "09:30"},
#             {"slot": "Slot 2", "start": "09:45", "end": "10:45"},
#             {"slot": "Slot 3", "start": "11:00", "end": "12:00"},
#             {"slot": "Slot 4", "start": "13:00", "end": "14:00"},
#             {"slot": "Slot 5", "start": "14:15", "end": "15:15"},
#             {"slot": "Slot 6", "start": "15:30", "end": "16:30"}
#         ]
#     },

#     # Repeat for Class C, Class D, Class E with varied slot counts per day
# }


# # =========================
# # Constants
# # =========================
# DATASET_DIR = 'dataset'
# STUDENTS_FILE = 'student_class_mapping.csv'
# ATTENDANCE_DIR = 'attendance'
# PASSWORD = "teacher123"
# EMBEDDINGS_FILE = "student_embeddings.pkl"

# # =========================
# # Ensure directories
# # =========================
# os.makedirs(ATTENDANCE_DIR, exist_ok=True)
# os.makedirs(DATASET_DIR, exist_ok=True)

# if not os.path.exists(STUDENTS_FILE):
#     pd.DataFrame(columns=["Roll", "Name"]).to_csv(STUDENTS_FILE, index=False)

# # =========================
# # Device configuration
# # =========================
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # =========================
# # Load FaceNet Models (cached)
# # =========================
# @st.cache_resource
# def load_facenet_models():
#     mtcnn_model = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
#     resnet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#     return mtcnn_model, resnet_model

# mtcnn, resnet = load_facenet_models()

# # =========================
# # Load student embeddings
# # =========================
# @st.cache_resource
# def load_student_embeddings():
#     if not os.path.exists(EMBEDDINGS_FILE):
#         st.error("❌ Embeddings file not found. Please run the embedding generator script first.")
#         st.stop()
#     with open(EMBEDDINGS_FILE, "rb") as f:
#         data = pickle.load(f)
#     return data["student_embeddings"]

# student_embeddings = load_student_embeddings()

# # =========================
# # Attendance utilities
# # =========================
# def get_attendance_file():
#     today = datetime.now().strftime("%Y-%m-%d")
#     return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

# def load_attendance():
#     file = get_attendance_file()
#     if not os.path.exists(file):
#         df = pd.read_csv(STUDENTS_FILE)
#         all_slots = {slot["slot"] for cls in timetable.values() for day in cls.values() for slot in day}
#         for slot in sorted(all_slots):
#             df[slot] = "Absent"
#         df.to_csv(file, index=False)
#     return pd.read_csv(file)

# def save_attendance(df):
#     df.to_csv(get_attendance_file(), index=False)

# def get_current_slot(class_name):
#     today = datetime.now().strftime("%A")
#     now_time = datetime.now().strftime("%H:%M")
#     if class_name not in timetable:
#         return None
#     slots_today = timetable[class_name].get(today, [])
#     for slot in slots_today:
#         if slot["start"] <= now_time <= slot["end"]:
#             return slot["slot"]
#     return None

# # =========================
# # Recognition & Attendance marking
# # =========================
# # =========================
# # Recognition & Attendance marking
# # =========================
# # =========================
# # Recognition & Attendance marking
# # =========================
# # =========================
# # Recognition & Attendance marking
# # =========================
# def recognize_and_mark_attendance(class_name):
#     slot = get_current_slot(class_name)
#     if not slot:
#         st.error("⏰ No ongoing slot for this class at the current time.")
#         return

#     df = load_attendance()
#     marked_students = {}

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("❌ Error: Could not open camera.")
#         return

#     frame_placeholder = st.empty()
#     st.warning(f"Class: {class_name} | Slot: {slot}")

#     # Create the stop button ONCE before the loop starts
#     stop_button = st.button("⏹ Stop Camera")

#     while cap.isOpened() and not stop_button:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("❌ Error receiving frame from camera.")
#             break

#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         try:
#             face = mtcnn(img)
#             if face is not None:
#                 with torch.no_grad():
#                     emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
                
#                 scores = {name: cosine_similarity([emb], [vec])[0][0] for name, vec in student_embeddings.items()}
#                 identified = max(scores, key=scores.get)
#                 score = scores[identified]

#                 if score >= 0.8:
#                     roll = identified.split('_')[0]
#                     name = '_'.join(identified.split('_')[1:])
                    
#                     student_row = df[df["Roll"].astype(str) == str(roll)]
#                     if not student_row.empty:
#                         if df.at[student_row.index[0], slot] != 'Present':
#                             df.at[student_row.index[0], slot] = 'Present'
#                             marked_students[roll] = name
#                             st.success(f"✅ Marked Present: {roll} - {name}")
                        
#                         boxes, _ = mtcnn.detect(img)
#                         if boxes is not None:
#                             box = [int(b) for b in boxes[0]]
#                             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                             cv2.putText(frame, f"{name} ({roll})", (box[0], box[1] - 10),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                 else:
#                     boxes, _ = mtcnn.detect(img)
#                     if boxes is not None:
#                         box = [int(b) for b in boxes[0]]
#                         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
#                         cv2.putText(frame, "Unknown", (box[0], box[1] - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         except Exception as e:
#             st.error(f"An error occurred during face detection: {e}")

#         frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
#         # Check if the button was clicked
#         if stop_button:
#             break

#     cap.release()
#     save_attendance(df)
#     st.info("Attendance marking session ended.")
#     if marked_students:
#         st.subheader("Attendance Marked")
#         for roll, name in marked_students.items():
#             st.write(f"- {roll}: {name}")
#     else:
#         st.warning("No new students were marked present in this session.")

# # def recognize_and_mark_attendance(selected_class):
# #     slot = get_current_slot(selected_class)  # You should have a function for this
# #     recognized_names = run_face_recognition()  # Your existing recognition logic
    
# #     # Load existing attendance
# #     df = load_attendance()
    
# #     for name in recognized_names:
# #         new_record = {
# #             "Name": name,
# #             "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #             "Class": selected_class,
# #             "Slot": slot,
# #             "Status": "Present"
# #         }
# #         df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    
# #     save_attendance(df)


# # =========================
# # Teacher Panel
# # =========================
# def show_teacher_panel():
#     st.subheader("📋 Attendance Dashboard")

#     df = load_attendance()

#     # Check if the dataframe contains the necessary 'Class' and 'Student Name' columns
#     if 'Class' not in df.columns or 'Student Name' not in df.columns:
#         st.error("❌ The 'Class' or 'Student Name' column is missing from students.csv. Please ensure your student data is correctly formatted.")
#         return
        
#     # Get a list of all slot columns from the timetable
#     all_slots = {slot["slot"] for cls in timetable.values() for day in cls.values() for slot in day}
#     slot_columns = sorted(list(all_slots))
    
#     # Filter the DataFrame to only show students marked as "Present" in any slot
#     present_mask = (df[slot_columns] == "Present").any(axis=1)
#     df_present = df[present_mask].copy()

#     if df_present.empty:
#         st.info("No students have been marked present yet.")
#         return

#     # Create a cleaner display DataFrame for the user
#     # This keeps the original data but simplifies the view
#     df_display = df_present[["Roll", "Student Name", "Class"] + slot_columns].copy()

#     # Filtering options for the user
#     unique_classes = sorted(df_display["Class"].unique())
#     unique_slots = [col for col in slot_columns if (df_display[col] == "Present").any()]

#     class_filter = st.selectbox("Filter by Class", ["All"] + unique_classes)
#     slot_filter = st.selectbox("Filter by Slot", ["All"] + unique_slots)
    
#     # Apply filters to the display DataFrame
#     if class_filter != "All":
#         df_display = df_display[df_display["Class"] == class_filter]

#     if slot_filter != "All":
#         # Keep only the columns for the selected slot and student details
#         df_display = df_display[df_display[slot_filter] == "Present"]
#         st.dataframe(df_display[["Roll", "Student Name", "Class"]])
#     else:
#         st.dataframe(df_display)

#     st.caption("Showing students marked present. Auto-refreshing every 8 seconds...")
#     time.sleep(8)
#     st.rerun()


# # =========================
# # Streamlit UI
# # =========================
# st.title("🎓 SmartAttend - AI-Based Attendance System")

# tab = st.selectbox("Choose Mode", ["📌 Mark Attendance", "🔐 Teacher Panel"])

# if tab == "📌 Mark Attendance":
#     class_choice = st.selectbox("Select Your Class", list(timetable.keys()))
#     if st.button("📷 Start Camera and Mark Attendance"):
#         recognize_and_mark_attendance(class_choice)

# elif tab == "🔐 Teacher Panel":
#     pwd = st.text_input("Enter Teacher Password", type="password")
#     if pwd == PASSWORD:
#         show_teacher_panel()
#     elif pwd:
#         st.error("Incorrect password.")






import streamlit as st
st.set_page_config(layout="centered")

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import torch
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import time


# =========================
# Constants
# =========================
DATASET_DIR = 'dataset'
STUDENTS_FILE = 'students.csv'
ATTENDANCE_DIR = 'attendance'
PASSWORD = "teacher123"
EMBEDDINGS_FILE = "student_embeddings.pkl"
NUM_SLOTS = 6
SLOTS = [f"Slot {i+1}" for i in range(NUM_SLOTS)]

# =========================
# Ensure directories
# =========================
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

if not os.path.exists(STUDENTS_FILE):
    pd.DataFrame(columns=["Roll", "Name"]).to_csv(STUDENTS_FILE, index=False)

# =========================
# Device configuration
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Load FaceNet Models (cached)
# =========================
@st.cache_resource
def load_facenet_models():
    mtcnn_model = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn_model, resnet_model

mtcnn, resnet = load_facenet_models()

# =========================
# Load student embeddings from pickle (cached)
# =========================
@st.cache_resource
def load_student_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error("❌ Embeddings file not found. Please run the embedding generator script first.")
        st.stop()
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["student_embeddings"]

student_embeddings = load_student_embeddings()

# =========================
# Attendance utilities
# =========================
def get_attendance_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

def load_attendance():
    file = get_attendance_file()
    if not os.path.exists(file):
        df = pd.read_csv(STUDENTS_FILE)
        for slot in SLOTS:
            df[slot] = "Absent"
        df.to_csv(file, index=False)
    return pd.read_csv(file)

def save_attendance(df):
    df.to_csv(get_attendance_file(), index=False)

# =========================
# Recognition & Attendance marking
# =========================
# def recognize_and_mark_attendance(slot):
#     df = load_attendance()
#     marked = []

#     cap = cv2.VideoCapture(0)
#     st.warning("Press 'Q' in webcam window to stop recognition.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         face = mtcnn(img)
#         if face is not None:
#             with torch.no_grad():
#                 emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
#             scores = {name: cosine_similarity([emb], [vec])[0][0] for name, vec in student_embeddings.items()}
#             identified = max(scores, key=scores.get)
#             score = scores[identified]

#             x, y, w, h = 50, 50, 200, 200  # Dummy bounding box

#             if score >= 0.8:
#                 try:
#                     roll = identified.split('_')[0]
#                     name = '_'.join(identified.split('_')[1:])
#                     student_row = df[df["Roll"].astype(str) == str(roll)]
#                 except Exception as e:
#                     st.error(f"Parsing error: {e}")
#                     continue

#                 if not student_row.empty:
#                     already_present = df.at[student_row.index[0], slot] == 'Present'
#                     if not already_present:
#                         df.at[student_row.index[0], slot] = 'Present'
#                         marked_time = datetime.now().strftime("%H:%M:%S")
#                         marked.append((roll, name))
#                         st.success(f"✅ Marked: {roll} - {name} at {marked_time} in {slot}")
#                     else:
#                         st.info(f"⚠️ {name} already marked Present in {slot}.")

#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{name}, {slot}", (x, y-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             else:
#                 cv2.putText(frame, "Unknown", (x, y-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#         cv2.imshow("Mark Attendance", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     save_attendance(df)

#     if not marked:
#         st.error("❌ No known face detected.")

def recognize_and_mark_attendance(slot):
    df = load_attendance()
    marked = []

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    st.warning("Click 'Stop Camera' or close tab to stop recognition.")

    stop_button = st.button("⏹ Stop Camera")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]
            scores = {name: cosine_similarity([emb], [vec])[0][0] for name, vec in student_embeddings.items()}
            identified = max(scores, key=scores.get)
            score = scores[identified]

            x, y, w, h = 50, 50, 200, 200  # Dummy bounding box

            if score >= 0.8:
                roll = identified.split('_')[0]
                name = '_'.join(identified.split('_')[1:])
                student_row = df[df["Roll"].astype(str) == str(roll)]

                if not student_row.empty:
                    already_present = df.at[student_row.index[0], slot] == 'Present'
                    if not already_present:
                        df.at[student_row.index[0], slot] = 'Present'
                        marked_time = datetime.now().strftime("%H:%M:%S")
                        marked.append((roll, name))
                        st.success(f"✅ Marked: {roll} - {name} at {marked_time} in {slot}")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}, {slot}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Show frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

        # Stop loop if button clicked
        if stop_button:
            break

    cap.release()
    save_attendance(df)

    if not marked:
        st.error("❌ No known face detected.")


# =========================
# Teacher Panel
# =========================
def view_attendance():
    st.subheader("📋 Attendance Dashboard")
    df = load_attendance()
    st.dataframe(df)
    time.sleep(10)
    st.experimental_rerun()

# =========================
# Streamlit UI
# =========================
st.title("🎓 SmartAttend - AI-Based Attendance System")

tab = st.selectbox("Choose Mode", ["📌 Mark Attendance", "🔐 Teacher Panel"])

if tab == "📌 Mark Attendance":
    slot = st.selectbox("Select Slot", SLOTS)
    if st.button("📷 Start Camera and Mark Attendance"):
        recognize_and_mark_attendance(slot)

elif tab == "🔐 Teacher Panel":
    pwd = st.text_input("Enter Teacher Password", type="password")
    if pwd == PASSWORD:
        view_attendance()
    elif pwd:
        st.error("Incorrect password.")
