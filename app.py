# app.py
import json
import logging
import os
import random
import string
from datetime import datetime
from pathlib import Path

import streamlit as st
from rag_bot import create_educational_bot

# Configure page and logging
st.set_page_config(
    page_title="Educational RAG Assistant", page_icon="📚", layout="wide"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure education_data directory exists
EDUCATION_DATA_PATH = "educational_data"
KNOWLEDGE_BASE_PATH = "knowledge_base.pkl"

if not os.path.exists(EDUCATION_DATA_PATH):
    os.makedirs(EDUCATION_DATA_PATH)
    logger.info(f"Created {EDUCATION_DATA_PATH} directory")


def init_knowledge_base():
    """Initialize the knowledge base with comprehensive sample content if needed."""
    sample_file = os.path.join(EDUCATION_DATA_PATH, "sample.txt")
    if not any(Path(EDUCATION_DATA_PATH).glob("*.txt")):
        logger.info("Creating sample educational content...")
        sample_content = """# Mathematics - Grade 4-5
## Number Operations
- Addition and subtraction with multi-digit numbers
- Multiplication and division with 2-digit numbers
- Fractions and decimals basics
- Understanding place value to millions
- Problem-solving with word problems

## Geometry
- Understanding angles (acute, right, obtuse)
- Properties of 2D shapes
- Area and perimeter calculations
- Basic coordinate geometry
- Symmetry and patterns

# English Language Arts - Grade 4-5

## Reading Comprehension
- Main idea and supporting details
- Making inferences
- Character analysis
- Story elements
- Author's purpose

## Writing Skills
- Paragraph structure
- Essay organization
- Descriptive writing
- Narrative writing
- Opinion writing

# Science - Grade 4-5

## Life Science
- Plant and animal adaptations
- Food chains and ecosystems
- Human body systems
- Life cycles
- Classification of living things

## Physical Science
- States of matter
- Forces and motion
- Simple machines
- Energy forms
- Sound and light"""

        try:
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write(sample_content)
            logger.info("Created sample educational content file")
            return True
        except Exception as e:
            logger.error(f"Error creating sample content: {e}")
            return False
    return True


@st.cache_resource
def init_rag_bot():
    """Initialize the RAG bot with progress reporting."""
    try:
        model_path = "Llama-3.2-3B-Instruct-Q5_K_S.gguf"

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        with st.spinner("Initializing RAG Bot..."):
            # Initialize knowledge base first
            st.text("Setting up educational knowledge base...")
            if not init_knowledge_base():
                st.error("Failed to initialize knowledge base")
                st.stop()

            st.text("Loading model and processing educational content...")
            bot = create_educational_bot(model_path)
            st.success("Bot initialized successfully!")
            return bot

    except Exception as e:
        st.error(f"Failed to initialize bot: {str(e)}")
        st.stop()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "assignments" not in st.session_state:
    st.session_state.assignments = {}
if "codes" not in st.session_state:
    st.session_state.codes = {}
if "rag_bot" not in st.session_state:
    st.session_state.rag_bot = init_rag_bot()


def generate_code():
    """Generate a random assignment code."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def save_data():
    """Save assignments and codes to JSON file."""
    try:
        data = {
            "assignments": st.session_state.assignments,
            "codes": st.session_state.codes,
        }
        with open("data.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")


def load_data():
    """Load assignments and codes from JSON file."""
    try:
        if os.path.exists("data.json"):
            with open("data.json", "r") as f:
                data = json.load(f)
                st.session_state.assignments = data.get("assignments", {})
                st.session_state.codes = data.get("codes", {})
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state.assignments = {}
        st.session_state.codes = {}


def teacher_interface():
    """Handle the teacher interface."""
    st.title("Teacher Interface")

    with st.form("assignment_form"):
        st.subheader("Create New Assignment")
        title = st.text_input("Assignment Title")
        description_1 = st.text_input("Description")
        subject = st.selectbox(
            "Subject",
            ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science"],
        )
        level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        submit = st.form_submit_button("Generate Assignment")

    if submit and title:
        try:
            with st.spinner("Generating assignment..."):
                description = st.session_state.rag_bot.generate_assignment(
                    title, subject, level, description_1
                )

            code = generate_code()
            st.session_state.assignments[code] = {
                "title": title,
                "subject": subject,
                "level": level,
                "description": description,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.codes[code] = []
            save_data()

            st.success(f"Assignment created! Code: {code}")
            st.write("Generated Assignment:")
            st.write(description)

        except Exception as e:
            st.error(f"Error generating assignment: {str(e)}")

    if st.session_state.assignments:
        st.subheader("Existing Assignments")
        for code, assignment in st.session_state.assignments.items():
            try:
                with st.expander(
                    f"Assignment: {assignment.get('title', 'Untitled')} (Code: {code})"
                ):
                    st.write(f"Subject: {assignment.get('subject', 'N/A')}")
                    st.write(f"Level: {assignment.get('level', 'N/A')}")
                    st.write("Description:")
                    st.write(assignment.get("description", "No description available"))
                    st.write(
                        f"Students joined: {len(st.session_state.codes.get(code, []))}"
                    )
            except Exception as e:
                st.error(f"Error displaying assignment {code}: {str(e)}")
    else:
        st.info("No assignments created yet.")


def student_interface():
    """Handle the student interface."""
    st.title("Student Interface")

    code = st.text_input("Enter assignment code").upper()
    if st.button("Join Assignment"):
        if code in st.session_state.assignments:
            try:
                if code not in st.session_state.codes:
                    st.session_state.codes[code] = []
                student_id = f"student_{len(st.session_state.codes[code])}"
                st.session_state.codes[code].append(student_id)
                save_data()
                st.success("Successfully joined the assignment!")
                st.session_state.current_assignment = code
                st.session_state.student_id = student_id
                if student_id not in st.session_state.messages:
                    st.session_state.messages[student_id] = []
                st.rerun()
            except Exception as e:
                st.error(f"Error joining assignment: {str(e)}")
        else:
            st.error("Invalid code!")

    if "current_assignment" in st.session_state:
        try:
            assignment = st.session_state.assignments[
                st.session_state.current_assignment
            ]
            st.subheader(f"Assignment: {assignment.get('title', 'Untitled')}")
            st.write(f"Subject: {assignment.get('subject', 'N/A')}")
            st.write(f"Level: {assignment.get('level', 'N/A')}")
            st.write("Description:")
            st.write(assignment.get("description", "No description available"))

            student_id = st.session_state.student_id

            for message in st.session_state.messages.get(student_id, []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask for help"):
                st.session_state.messages[student_id].append(
                    {"role": "user", "content": prompt}
                )

                try:
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_bot.get_help_response(
                            prompt, assignment.get("description", "")
                        )

                    st.session_state.messages[student_id].append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
        except Exception as e:
            st.error(f"Error displaying current assignment: {str(e)}")
            st.session_state.pop("current_assignment", None)


def main():
    """Main application entry point."""
    load_data()

    st.sidebar.title("Educational RAG Assistant")
    mode = st.sidebar.selectbox("Select Mode", ["Teacher", "Student"])

    if mode == "Teacher":
        teacher_interface()
    else:
        student_interface()


if __name__ == "__main__":
    main()
