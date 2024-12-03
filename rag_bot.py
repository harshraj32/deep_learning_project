# rag_bot.py
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict
import gc

import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks: List[str] = []
        self.embeddings = None
        self.knn = None

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """Retrieve the k most relevant chunks for a query."""
        if not self.chunks or self.embeddings is None or self.knn is None:
            logger.warning("No documents have been processed yet")
            return []

        try:
            query_embedding = self.embedding_model.encode([query])

            # Adjust k to be at most the number of available chunks
            k = min(k, len(self.chunks))
            if k == 0:
                return []

            distances, indices = self.knn.kneighbors(query_embedding)
            return [self.chunks[idx] for idx in indices[0]]
        except Exception as e:
            logger.error(f"Error getting relevant chunks: {e}")
            return []

    def add_documents(
        self, file_paths: List[str], chunk_size: int = 500, overlap: int = 50
    ):
        """Process and store documents with overlap."""
        if not file_paths:
            logger.warning("No files provided to process")
            return

        self.chunks = []  # Reset chunks

        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                # Create overlapping chunks
                paragraphs = text.split("\n\n")
                current_chunk = ""

                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) < chunk_size:
                        current_chunk += paragraph + "\n\n"
                    else:
                        if current_chunk:
                            self.chunks.append(current_chunk.strip())
                        # Keep the overlap from the previous chunk
                        words = current_chunk.split()[-overlap:]
                        current_chunk = " ".join(words) + "\n\n" + paragraph + "\n\n"

                if current_chunk:
                    self.chunks.append(current_chunk.strip())

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

        if not self.chunks:
            logger.warning("No chunks were created from the documents")
            return

        try:
            # Create embeddings for all chunks
            self.embeddings = self.embedding_model.encode(self.chunks)

            # Initialize nearest neighbors with adjusted k
            k = min(3, len(self.chunks))
            self.knn = NearestNeighbors(n_neighbors=k, metric="cosine")
            self.knn.fit(self.embeddings)

            logger.info(f"Successfully processed {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            self.chunks = []
            self.embeddings = None
            self.knn = None

    def save(self, path: str):
        """Save the document store to disk."""
        try:
            with open(path, "wb") as f:
                pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
            logger.info(f"Successfully saved knowledge base to {path}")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            raise

    def load(self, path: str):
        """Load the document store from disk."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.embeddings = data["embeddings"]

                # Reinitialize KNN if we have data
                if self.chunks and len(self.chunks) > 0:
                    k = min(3, len(self.chunks))
                    self.knn = NearestNeighbors(n_neighbors=k, metric="cosine")
                    self.knn.fit(self.embeddings)

            logger.info(f"Successfully loaded knowledge base from {path}")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise


class BaseAgent:
    def __init__(self, model_path: str, role: str, doc_store: DocumentStore = None):
        logger.info(f"Initializing {role} agent...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.doc_store = doc_store
        try:
            n_gpu_layers = 35 if torch.cuda.is_available() else 0
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_batch=512,
                n_threads=8,
                n_gpu_layers=n_gpu_layers,
                use_mlock=True,
                use_mmap=True,
                verbose=False,
            )
            logger.info(f"{role} agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {role} agent: {str(e)}")
            raise


class TeacherAgent(BaseAgent):
    def __init__(self, model_path: str, doc_store: DocumentStore = None):
        super().__init__(model_path, "Teacher", doc_store)

    def generate_assignment(
        self, title: str, subject: str, level: str, description: str
    ) -> str:
        try:
            context = ""
            if self.doc_store:
                relevant_chunks = self.doc_store.get_relevant_chunks(
                    f"{subject} {level} {title} {description}"
                )
                context = "\n\n".join(relevant_chunks)

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
    def __init__(self, model_path: str, doc_store: DocumentStore = None):
        super().__init__(model_path, "Student", doc_store)

    def get_help_response(self, question: str, assignment_context: str) -> str:
        try:
            context = ""
            if self.doc_store:
                relevant_chunks = self.doc_store.get_relevant_chunks(
                    f"{question} {assignment_context}"
                )
                context = "\n\n".join(relevant_chunks)

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

        logger.info("Initializing Document Store...")
        self.doc_store = DocumentStore()

        # Initialize knowledge base
        self._init_knowledge_base()

        logger.info("Creating Teacher Agent...")
        self.teacher_agent = TeacherAgent(model_path, self.doc_store)

        logger.info("Creating Student Agent...")
        self.student_agent = StudentAgent(model_path, self.doc_store)

        logger.info("Educational RAG Bot initialized successfully")

    def _init_knowledge_base(self):
        """Initialize or load the knowledge base."""
        knowledge_base_path = "knowledge_base.pkl"
        education_data_path = "educational_data"

        if os.path.exists(knowledge_base_path):
            logger.info("Loading existing knowledge base...")
            self.doc_store.load(knowledge_base_path)
        else:
            logger.info("Creating new knowledge base...")
            if not os.path.exists(education_data_path):
                os.makedirs(education_data_path)

            text_files = list(Path(education_data_path).glob("*.txt"))
            if text_files:
                self.doc_store.add_documents([str(p) for p in text_files])
                self.doc_store.save(knowledge_base_path)
            else:
                logger.warning("No educational content files found.")

    def generate_assignment(
        self, title: str, subject: str, level: str, description: str
    ) -> str:
        return self.teacher_agent.generate_assignment(
            title, subject, level, description
        )

    def get_help_response(self, question: str, assignment_context: str) -> str:
        return self.student_agent.get_help_response(question, assignment_context)


def create_educational_bot(model_path: str) -> EducationalRAGBot:
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
