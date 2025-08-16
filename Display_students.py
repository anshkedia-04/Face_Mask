import streamlit as st
import pandas as pd
import os

# Define the path to the students CSV file
STUDENTS_FILE = 'students.csv'

def show_student_data():
    """
    Reads the students.csv file and displays a filtered table in Streamlit.
    """
    st.title("👨‍🎓 Student Roster Viewer")

    # Check if the students.csv file exists
    if not os.path.exists(STUDENTS_FILE):
        st.error(f"❌ Error: The '{STUDENTS_FILE}' file was not found.")
        st.warning("Please ensure the script is in the same directory as the CSV file.")
        return

    # Load the data from the CSV file
    try:
        df = pd.read_csv(STUDENTS_FILE)
    except Exception as e:
        st.error(f"❌ Error loading data from '{STUDENTS_FILE}': {e}")
        return

    # Check if the 'Class' column exists in the DataFrame
    if 'Class' not in df.columns:
        st.error("❌ The 'Class' column is missing from the CSV file.")
        st.warning("Please ensure your students.csv has a 'Class' column.")
        return

    # Get a list of unique classes for the filter, sorted for clarity
    unique_classes = sorted(df['Class'].unique())

    # Create a selectbox for the class filter. 'All' is a default option.
    selected_class = st.selectbox(
        "Select a Class to view:",
        options=["All"] + unique_classes
    )

    # Filter the DataFrame based on the user's selection
    if selected_class == "All":
        filtered_df = df
    else:
        filtered_df = df[df['Class'] == selected_class]

    # Display the number of students found
    st.info(f"Showing {len(filtered_df)} student records.")
    
    # Display the filtered DataFrame
    st.dataframe(filtered_df, use_container_width=True)


if __name__ == '__main__':
    show_student_data()
