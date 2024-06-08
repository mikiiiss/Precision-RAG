
import streamlit as st
from prompt_generation import main
from io import StringIO
import sys

# Set page configuration
st.set_page_config(page_title="Prompt Generator", layout="wide")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
        .stTextInput>div>div>input {
            height: 50px !important;
            font-size: 1.2em !important;
            padding: 0.5em !important;
        }
        .stTextArea textarea {
            height: 400px !important; /* Further decreased height */
            font-size: 1.2em !important;
            padding: 0.5em !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a title to the app
st.title('Prompt Generator')

# Create two columns for layout
left_column, right_column = st.columns([1, 2])

# Place the text input in the left column
with left_column:
    query = st.text_input("Enter your query here:", key="query")
    base_query = st.text_input("Enter your base query here:", key="base_query")

# Initialize the output variable
output = ""

# Display the prompt box in the right column
with right_column:
    output_text_area = st.empty()
    with output_text_area.container():
        st.text_area("Generated Prompts", value=output, height=600)
# Generate prompts when the user clicks the "Generate Prompts" button
if st.button("Generate Prompts"):
    # Redirect stdout to a StringIO object to capture the output
    buffer = StringIO()
    sys.stdout = buffer

    # Call the main function from prompt_generation.py
    main(query, base_query)

    # Reset stdout to its default value
    sys.stdout = sys.__stdout__

    # Get the captured output as a string
    output = buffer.getvalue()

    # Update the prompt box with the generated prompts
    with output_text_area.container():
        st.text_area("Generated Prompts", value=output, height=600)