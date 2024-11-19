# rag_bot.py
import logging
import os
import pickle
from pathlib import Path
from typing import List
import gc

import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, model_path: str, role: str):
        logger.info(f"Initializing {role} agent...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
            
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=32,
                n_threads=4,
                n_gpu_layers=0,
                use_mlock=False,
                use_mmap=True,
                verbose=False
            )
            logger.info(f"{role} agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {role} agent: {str(e)}")
            raise

class TeacherAgent(BaseAgent):
    def __init__(self, model_path: str):
        super().__init__(model_path, "Teacher")

    def generate_assignment(self, title: str, subject: str, level: str) -> str:
        try:
            prompt = f"""[INST]
                        As a grade school teacher, create an engaging and age-appropriate assignment:

                        ASSIGNMENT DETAILS:
                        Title: {title}
                        Subject: {subject}
                        Grade Level: {level}

                        Please structure the assignment as follows:

                        1. LEARNING OBJECTIVES:
                        - List 2-3 clear, measurable objectives starting with "Students will be able to..."
                        - Make objectives appropriate for {level} level students

                        2. ESSENTIAL QUESTION:
                        - Provide one engaging question that sparks curiosity
                        - Make it relatable to students' daily lives

                        3. TASKS (create 3-4 tasks):
                        For each task:
                        - Number them clearly (Task 1, Task 2, etc.)
                        - Provide step-by-step instructions
                        - Estimate time needed
                        - List materials required
                        - Include a fun or interactive element

                        4. DIFFERENTIATION:
                        - Provide a simpler version for struggling students
                        - Add challenge options for advanced students

                        5. ASSESSMENT CRITERIA:
                        - List 3-4 specific things you'll look for in student work
                        - Include both effort and accuracy components
                        - Use student-friendly language

                        6. HOMEWORK (Optional):
                        - A simple follow-up activity for home
                        - Should take no more than 15-20 minutes

                        Remember to:
                        - Use child-friendly language
                        - Include visual elements or diagrams where helpful
                        - Make tasks interactive and engaging
                        - Keep instructions clear and concise
                        - Add encouraging notes or fun facts

                        For {subject} at {level} level, focus on making it:
                        - Age-appropriate
                        - Engaging and fun
                        - Challenging but achievable
                        - Connected to real-world examples

                        Please format the response with clear headings and bullet points for easy reading.[/INST]"""

            response = self.llm(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                top_k=40,
                echo=False,
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Assignment generation error: {str(e)}")
            return "Error generating assignment. Please try again."

class StudentAgent(BaseAgent):
    def __init__(self, model_path: str):
        super().__init__(model_path, "Student")

    def get_help_response(self, question: str, assignment_context: str) -> str:
        try:
            prompt = f"""[INST]
    You are a friendly and encouraging teaching assistant for grade school students. Your name is Buddy, and you're here to help students learn while having fun!

    CURRENT CONTEXT:
    Assignment: {assignment_context[:200]}

    Student's Message: {question}

    Your role is to:
    1. Be warm and friendly - use encouraging words and positive reinforcement
    2. Match your language to grade school level
    3. Use emojis occasionally to make responses fun
    4. Break down complex ideas into simple steps
    5. Provide examples from daily life
    6. Celebrate their efforts and progress
    7. Encourage curiosity and questions

    If the student:
    - Says hello/hi: Respond warmly and ask how you can help
    - Seems frustrated: Offer encouragement and break the problem into smaller steps
    - Shows interest: Share fun facts or interesting connections
    - Needs help: Guide them to the answer instead of giving it directly
    - Makes progress: Celebrate their achievement
    - Seems distracted: Gently bring focus back to the task
    - Asks off-topic questions: Answer briefly, then guide back to learning

    Response Guidelines:
    - Keep explanations short and clear
    - Use "Let's" to make it collaborative
    - Offer specific praise
    - Include gentle reminders
    - Make learning fun
    - Use analogies from their daily life
    - Break down complex problems
    - Encourage them to try

    Examples of good responses:
    For greeting: "Hi there! 👋 I'm Buddy, your friendly learning helper! How can I make learning fun for you today?"

    For math help: "Let's solve this step by step! Think of it like sharing cookies with friends - if you have 12 cookies and 3 friends, how many cookies would each friend get?"

    For reading help: "Great question! Let's be reading detectives 🔍 and look for clues in the story that tell us about the character's feelings."

    For writing help: "You're doing great! 🌟 Let's make your writing even more exciting. Can you think of a word that's more colorful than 'good'? Maybe 'fantastic' or 'amazing'?"

    Remember to:
    - Be patient and kind
    - Make them feel safe to ask questions
    - Keep the tone light and fun
    - Celebrate small wins
    - Encourage problem-solving
    - Build confidence

    Now, please respond to the student's message: {question}[/INST]"""

            response = self.llm(
                prompt,
                max_tokens=256,  # Increased for more natural conversation
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                top_k=40,
                echo=False,
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Help response error: {str(e)}")
            return "I'm sorry, I'm having trouble right now. But don't worry! Try asking me again in a different way. 😊"

class EducationalRAGBot:
    def __init__(self, model_path: str):
        logger.info("Initializing Educational RAG Bot...")
        self.model_path = model_path
        
        logger.info("Creating Teacher Agent...")
        self.teacher_agent = TeacherAgent(model_path)
        
        logger.info("Creating Student Agent...")
        self.student_agent = StudentAgent(model_path)
        
        logger.info("Educational RAG Bot initialized successfully")

    def generate_assignment(self, title: str, subject: str, level: str) -> str:
        return self.teacher_agent.generate_assignment(title, subject, level)

    def get_help_response(self, question: str, assignment_context: str) -> str:
        return self.student_agent.get_help_response(question, assignment_context)

def create_educational_bot(model_path: str) -> EducationalRAGBot:
    """Create and initialize the educational bot with progress reporting."""
    logger.info(f"Creating educational bot with model: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    try:
        bot = EducationalRAGBot(model_path)
        logger.info("Educational bot created successfully")
        return bot
    except Exception as e:
        logger.error(f"Failed to create educational bot: {str(e)}")
        raise