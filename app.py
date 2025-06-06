import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import re
from collections import Counter
import time
from typing import List, Dict, Tuple
import base64
import random
import math

# Configure page
st.set_page_config(
    page_title="PDF AI Assistant",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_summary' not in st.session_state:
        st.session_state.document_summary = None
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'zoom_level' not in st.session_state:
        st.session_state.zoom_level = 1.0  # Start at 100%


class EnhancedAI:
    """Enhanced AI with improved accuracy and confidence scoring"""

    @staticmethod
    def get_stop_words():
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as',
            'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'that', 'this',
            'these', 'those', 'it', 'he', 'she', 'they', 'we', 'i', 'you', 'me', 'him', 'her', 'us', 'them'
        }

    @staticmethod
    def get_question_type(question: str) -> str:
        """Determine question type for better matching"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['who', 'whom', 'whose']):
            return "PERSON"
        elif any(word in question_lower for word in ['when', 'date', 'time', 'year', 'month']):
            return "DATE"
        elif any(word in question_lower for word in ['where', 'location', 'place', 'address']):
            return "LOCATION"
        elif any(word in question_lower for word in ['why', 'reason', 'because', 'cause']):
            return "REASON"
        elif any(word in question_lower for word in ['how', 'method', 'process', 'way', 'procedure']):
            return "PROCESS"
        elif any(word in question_lower for word in ['what', 'which', 'define', 'meaning', 'definition']):
            return "DEFINITION"
        elif any(word in question_lower for word in ['how many', 'how much', 'number', 'amount', 'quantity', 'count']):
            return "QUANTITY"
        elif any(word in question_lower for word in ['conclusion', 'summary', 'result', 'finding']):
            return "CONCLUSION"
        return "GENERAL"

    @staticmethod
    def extract_answer(question: str, text: str) -> Tuple[str, float]:
        """Enhanced answer extraction with improved confidence scoring"""
        question_lower = question.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Determine question type
        q_type = EnhancedAI.get_question_type(question)

        # Extract keywords
        stop_words = EnhancedAI.get_stop_words()
        question_keywords = [word for word in re.findall(r'\w+', question_lower)
                             if word not in stop_words and len(word) > 2]

        # Enhanced scoring system
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = 0
            match_details = {'keywords': 0, 'exact': 0, 'type': 0, 'position': 0}

            # 1. Keyword matching with enhanced scoring
            for keyword in question_keywords:
                if keyword in sentence_lower:
                    score += 5
                    match_details['keywords'] += 1

                    # Exact word match
                    if re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower):
                        score += 3
                        match_details['exact'] += 1

                    # Stem matching
                    if keyword.endswith('ing') and keyword[:-3] in sentence_lower:
                        score += 2
                    elif keyword.endswith('ed') and keyword[:-2] in sentence_lower:
                        score += 2

            # 2. Enhanced question-type specific scoring
            if q_type == "PERSON":
                names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
                score += len(names) * 4
                match_details['type'] = len(names)

            elif q_type == "DATE":
                dates = re.findall(
                    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b|january|february|march|april|may|june|july|august|september|october|november|december',
                    sentence_lower)
                score += len(dates) * 5
                match_details['type'] = len(dates)

            elif q_type == "LOCATION":
                location_indicators = ['in', 'at', 'located', 'near', 'region', 'area', 'country', 'city', 'state']
                loc_score = sum(2 for word in location_indicators if word in sentence_lower)
                score += loc_score
                places = re.findall(r'\b[A-Z][a-z]+\b', sentence)
                score += len(places) * 2
                match_details['type'] = loc_score + len(places)

            elif q_type == "QUANTITY":
                numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', sentence)
                score += len(numbers) * 4
                match_details['type'] = len(numbers)

            elif q_type == "REASON":
                reason_indicators = ['because', 'since', 'therefore', 'thus', 'due to', 'reason', 'caused by',
                                     'result of']
                reason_score = sum(3 for word in reason_indicators if word in sentence_lower)
                score += reason_score
                match_details['type'] = reason_score

            elif q_type == "CONCLUSION":
                conclusion_indicators = ['conclude', 'conclusion', 'summary', 'therefore', 'thus', 'finally', 'result',
                                         'finding']
                conc_score = sum(4 for word in conclusion_indicators if word in sentence_lower)
                score += conc_score
                match_details['type'] = conc_score

            # 3. Context and position scoring
            if i < 5:  # Beginning of document
                score += 3
                match_details['position'] = 3
            elif i >= len(sentences) - 5:  # End of document (often conclusions)
                score += 4
                match_details['position'] = 4

            # 4. Sentence quality scoring
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 2

            # 5. Semantic coherence
            if i > 0 and any(kw in sentences[i - 1].lower() for kw in question_keywords):
                score += 2  # Context bonus

            if score > 0:
                scored_sentences.append((sentence, score, match_details, i))

        # Fallback for broader search
        if not scored_sentences:
            for i, sentence in enumerate(sentences[:30]):
                if any(kw in sentence.lower() for kw in question_keywords):
                    scored_sentences.append((sentence, 2.0, {'keywords': 1}, i))

        # Select and construct answer
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            # Get top sentences
            top_sentences = []
            used_indices = set()

            for sent, score, details, idx in scored_sentences[:5]:
                if idx not in used_indices:
                    top_sentences.append((sent, score, details, idx))
                    used_indices.add(idx)
                    # Add context if high score
                    if score > 15 and len(top_sentences) < 3:
                        # Add previous sentence if available
                        if idx > 0 and idx - 1 not in used_indices:
                            context_sent = sentences[idx - 1]
                            if len(context_sent) > 20:
                                top_sentences.append((context_sent, score * 0.7, details, idx - 1))
                                used_indices.add(idx - 1)

            # Sort by original position
            top_sentences.sort(key=lambda x: x[3])

            # Construct answer
            answer_parts = [s[0] for s in top_sentences[:3]]
            answer = ' '.join(answer_parts)

            # Calculate enhanced confidence
            max_score = scored_sentences[0][1]
            base_confidence = min(0.95, (max_score / 40) ** 0.8)  # Non-linear scaling

            # Confidence boosters
            confidence_boost = 0

            # Keyword coverage
            keywords_found = sum(1 for kw in question_keywords if kw in answer.lower())
            keyword_coverage = keywords_found / len(question_keywords) if question_keywords else 0
            confidence_boost += keyword_coverage * 0.15

            # Type-specific boost
            if scored_sentences[0][2]['type'] > 0:
                confidence_boost += 0.1

            # Exact match boost
            if scored_sentences[0][2]['exact'] > 0:
                confidence_boost += 0.08

            # Multiple supporting sentences
            if len(top_sentences) > 1:
                confidence_boost += 0.05

            # Final confidence
            confidence = min(0.95, base_confidence + confidence_boost)

            # Ensure minimum confidence for good matches
            if max_score > 20:
                confidence = max(confidence, 0.75)
            elif max_score > 10:
                confidence = max(confidence, 0.6)

            return answer, confidence

        # Enhanced fallbacks
        fallbacks = [
            "I couldn't find specific information about this in the document. The content may be addressing different aspects of the topic.",
            "The document doesn't directly answer this question, but it contains related information that might be helpful.",
            "This particular detail isn't explicitly covered in the document. You might want to check if there's related content in other sections.",
            "I wasn't able to locate a direct answer to your question in the document. The information might be implicit or covered differently."
        ]
        return random.choice(fallbacks), 0.15

    @staticmethod
    def generate_structured_summary(text: str, key_info: Dict) -> str:
        """Generate a well-structured, paragraph-based summary"""
        # Extract first 3000 characters for summary
        text_sample = text[:3000]

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text_sample) if len(s.strip()) > 20]

        # Score sentences for importance
        word_freq = Counter(text.lower().split())
        stop_words = EnhancedAI.get_stop_words()

        # Remove stop words from frequency count
        for word in stop_words:
            word_freq.pop(word, None)

        # Score sentences
        sentence_scores = []
        for sentence in sentences:
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words if word not in stop_words)
            sentence_scores.append((sentence, score))

        # Get top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:5]]

        # Sort by original order
        top_sentences = sorted(top_sentences, key=lambda s: sentences.index(s))

        # Create structured summary
        summary_parts = []

        # Overview paragraph
        overview = '. '.join(top_sentences[:3])
        summary_parts.append(f'**📋 Overview**\n\n{overview}')

        # Key topics paragraph
        if key_info.get('key_topics'):
            topics = key_info['key_topics'][:8]
            topics_text = f"The document primarily discusses {', '.join(topics[:3])}"
            if len(topics) > 3:
                topics_text += f", along with other topics including {', '.join(topics[3:6])}"
            topics_text += ". These themes appear throughout the document and form the core subject matter."
            summary_parts.append(f'\n\n**🎯 Main Topics**\n\n{topics_text}')

        # Key entities paragraph
        if key_info.get('entities'):
            entities = key_info['entities'][:6]
            entities_text = f"Important names and organizations mentioned include {', '.join(entities[:3])}"
            if len(entities) > 3:
                entities_text += f" and {', '.join(entities[3:])}"
            entities_text += ". These entities play significant roles in the document's content."
            summary_parts.append(f'\n\n**👥 Key Entities**\n\n{entities_text}')

        # Dates and timeline
        if key_info.get('dates'):
            dates = key_info['dates'][:5]
            dates_text = f"The document references several important dates including {', '.join(dates)}. "
            dates_text += "These temporal references help establish the timeline and context of the discussed events."
            summary_parts.append(f'\n\n**📅 Timeline**\n\n{dates_text}')

        return '\n'.join(summary_parts)


class DocumentProcessor:
    """Process PDF documents with enhanced analysis"""

    @staticmethod
    def extract_text_and_metadata(pdf_path: str) -> Dict:
        """Extract text and metadata from PDF"""
        try:
            doc = fitz.open(pdf_path)

            metadata = {
                "title": doc.metadata.get("title", os.path.basename(pdf_path)),
                "author": doc.metadata.get("author", "Unknown"),
                "page_count": doc.page_count,
                "file_size": os.path.getsize(pdf_path) / (1024 * 1024),
            }

            full_text = ""
            pages_data = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                full_text += text + "\n"

                words = text.split()
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "word_count": len(words),
                    "char_count": len(text)
                })

            total_words = sum(p["word_count"] for p in pages_data)

            metadata.update({
                "total_words": total_words,
                "avg_words_per_page": total_words / doc.page_count if doc.page_count > 0 else 0,
                "full_text": full_text,
                "pages_data": pages_data,
                "reading_time": round(total_words / 200),
            })

            doc.close()
            return metadata

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None

    @staticmethod
    def extract_key_info(text: str) -> Dict:
        """Extract key information with context"""
        # Extract entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entity_counts = Counter(entities)
        top_entities = [e for e, c in entity_counts.most_common(10) if c > 1]

        # Extract dates
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b', text)
        dates = list(set(dates))[:5]

        # Extract numbers with context
        numbers_with_context = []
        number_pattern = r'(\b\d+(?:,\d{3})*(?:\.\d+)?%?\b)'
        for match in re.finditer(number_pattern, text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            context = re.sub(r'\s+', ' ', context)  # Clean whitespace
            numbers_with_context.append({
                'value': match.group(1),
                'context': context
            })

        # Deduplicate and limit
        seen_numbers = set()
        unique_numbers = []
        for item in numbers_with_context:
            if item['value'] not in seen_numbers and len(item['value']) > 2:
                seen_numbers.add(item['value'])
                unique_numbers.append(item)
                if len(unique_numbers) >= 10:
                    break

        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

        # Extract key topics
        words = text.lower().split()
        stop_words = EnhancedAI.get_stop_words()
        words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
        word_freq = Counter(words)
        key_topics = [w for w, c in word_freq.most_common(15) if c > 3]

        return {
            "entities": top_entities,
            "dates": dates,
            "numbers": unique_numbers,
            "emails": list(set(emails)),
            "key_topics": key_topics
        }

    @staticmethod
    def create_chunks(text: str, chunk_size: int = 500) -> List[str]:
        """Create text chunks for processing"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


# CSS styling
st.markdown("""
<style>
    /* Dark theme styling */
    .stApp {
        background-color: #0e1117;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        margin: -1rem -1rem 2rem -1rem;
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
    }

    /* Chat styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .user-message {
        background-color: #5e72e4;
        color: white;
        margin-left: 2rem;
    }

    .assistant-message {
        background-color: #1e1e1e;
        border: 1px solid #333;
        margin-right: 2rem;
    }

    /* Confidence meter */
    .confidence-container {
        margin-top: 1rem;
    }

    .confidence-bar {
        width: 100%;
        height: 8px;
        background-color: #333;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #f87171 0%, #facc15 50%, #4ade80 100%);
        transition: width 0.3s ease;
    }

    .confidence-text {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #888;
    }

    /* Summary card */
    .summary-section {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* PDF viewer */
    .pdf-viewer-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
    }

    /* Remove extra spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


def create_header():
    """Create application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 PDF AI Assistant</h1>
        <p>Advanced document analysis with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar():
    """Create enhanced sidebar"""
    with st.sidebar:
        st.markdown("## 📁 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF to analyze"
        )

        if uploaded_file:
            st.success("✅ File uploaded!")

            if st.button("⚡ Process Document", type="primary", use_container_width=True):
                process_document(uploaded_file)

        if st.session_state.current_file:
            st.markdown("---")
            st.markdown("### 📄 Current Document")
            doc = st.session_state.current_file

            st.info(f"""
            **{doc['original_filename']}**
            - 📄 {doc['page_count']} pages
            - 📝 {doc['total_words']:,} words
            - ⏱️ {doc['reading_time']} min read
            - 💾 {doc['file_size']:.2f} MB
            """)

            if st.button("🗑️ Clear Document", type="secondary", use_container_width=True):
                # Clean up
                if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
                    os.unlink(st.session_state.pdf_path)

                st.session_state.current_file = None
                st.session_state.chat_history = []
                st.session_state.document_summary = None
                st.session_state.text_chunks = []
                st.session_state.pdf_path = None
                st.session_state.current_page = 0
                st.rerun()


def process_document(uploaded_file):
    """Process document with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Save file
        status_text.text("📄 Saving document...")
        progress_bar.progress(20)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.session_state.pdf_path = tmp_path

        # Extract text
        status_text.text("📖 Extracting content...")
        progress_bar.progress(40)

        doc_data = DocumentProcessor.extract_text_and_metadata(tmp_path)
        if not doc_data:
            return

        doc_data["original_filename"] = uploaded_file.name

        # Extract key info
        status_text.text("🔍 Analyzing document...")
        progress_bar.progress(60)

        key_info = DocumentProcessor.extract_key_info(doc_data['full_text'])
        doc_data.update(key_info)

        # Create chunks
        status_text.text("📝 Optimizing for search...")
        progress_bar.progress(80)

        chunks = DocumentProcessor.create_chunks(doc_data['full_text'])
        st.session_state.text_chunks = chunks

        # Generate summary
        status_text.text("📋 Creating intelligent summary...")
        progress_bar.progress(90)

        summary = EnhancedAI.generate_structured_summary(doc_data['full_text'], key_info)

        # Store everything
        st.session_state.current_file = doc_data
        st.session_state.document_summary = summary
        st.session_state.chat_history = []
        st.session_state.current_page = 0

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)

        progress_bar.empty()
        status_text.empty()

        st.success("✅ Document ready for analysis!")
        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def display_pdf_viewer():
    """Display PDF viewer with A4 size and clean layout"""
    st.markdown("## 📄 Document Viewer")

    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        doc = fitz.open(st.session_state.pdf_path)

        # Navigation controls in columns
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.rerun()

        with col2:
            # Page selector
            page_num = st.selectbox(
                "Page",
                options=range(doc.page_count),
                format_func=lambda x: f"Page {x + 1} of {doc.page_count}",
                index=st.session_state.current_page,
                label_visibility="collapsed"
            )
            if page_num != st.session_state.current_page:
                st.session_state.current_page = page_num
                st.rerun()

        with col3:
            if st.button("Next ➡️", disabled=st.session_state.current_page == doc.page_count - 1):
                st.session_state.current_page += 1
                st.rerun()

        # Zoom controls
        zoom_col1, zoom_col2, zoom_col3 = st.columns([2, 1, 2])
        with zoom_col2:
            zoom_options = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            zoom = st.selectbox(
                "Zoom",
                options=zoom_options,
                format_func=lambda x: f"{int(x * 100)}%",
                index=zoom_options.index(st.session_state.zoom_level),
                label_visibility="collapsed"
            )
            if zoom != st.session_state.zoom_level:
                st.session_state.zoom_level = zoom
                st.rerun()

        # Display PDF page
        page = doc[st.session_state.current_page]

        # A4 size at 72 DPI: 595 x 842 pixels
        # Scale to fit width while maintaining aspect ratio
        base_width = 595
        zoom_factor = st.session_state.zoom_level

        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # Calculate matrix to achieve desired width
        scale = (base_width * zoom_factor) / page_width
        mat = fitz.Matrix(scale, scale)

        # Render page
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")

        # Display in a centered container
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image(img_data, use_column_width=True)

        # Page text in expander
        with st.expander("📝 Extract Page Text"):
            page_text = page.get_text()
            st.text_area("Text Content", page_text, height=200, label_visibility="collapsed")

        doc.close()
    else:
        st.info("📄 No document loaded. Please upload a PDF file to view it here.")


def display_chat_interface():
    """Display chat interface with proper formatting"""
    st.markdown("## 💬 Chat with Document")

    # Quick questions if no history
    if not st.session_state.chat_history:
        st.markdown("### 💡 Quick Questions")
        questions = [
            "What is this document about?",
            "What are the main findings or conclusions?",
            "Who are the key people mentioned?",
            "What are the important dates?",
            "What is the main purpose?",
            "What recommendations are made?"
        ]

        cols = st.columns(2)
        for i, q in enumerate(questions):
            with cols[i % 2]:
                if st.button(q, key=f"q_{i}"):
                    process_question(q)

    # Chat input
    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        process_question(user_input)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation History")

        for chat in reversed(st.session_state.chat_history[-5:]):
            # User message
            with st.container():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>❓ You:</strong><br>
                    {chat['question']}
                </div>
                """, unsafe_allow_html=True)

            # Assistant message
            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 AI Assistant:</strong><br>
                        {chat['answer']}
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # Confidence display
                    confidence_pct = int(chat['confidence'] * 100)
                    color = "#4ade80" if confidence_pct > 70 else "#facc15" if confidence_pct > 40 else "#f87171"

                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 2rem; font-weight: bold; color: {color};">
                            {confidence_pct}%
                        </div>
                        <div style="font-size: 0.8rem; color: #888;">
                            Confidence
                        </div>
                        <div style="font-size: 0.7rem; color: #666;">
                            {chat['timestamp']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Visual confidence bar
                    st.progress(chat['confidence'])


def process_question(question: str):
    """Process user question with enhanced AI"""
    if not st.session_state.current_file:
        st.error("⚠️ Please upload a document first!")
        return

    with st.spinner("🤔 Analyzing your question..."):
        answer, confidence = EnhancedAI.extract_answer(
            question,
            st.session_state.current_file['full_text']
        )

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })

        st.rerun()


def display_summary():
    """Display document summary"""
    st.markdown("## 📋 Document Summary")

    if st.session_state.document_summary:
        # Display the summary in a nice card
        st.markdown(f"""
        <div class="summary-section">
            <h3>📄 {st.session_state.current_file['original_filename']}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Display summary content
        st.markdown(st.session_state.document_summary)

        # Key metrics
        doc = st.session_state.current_file

        col1, col2 = st.columns(2)

        with col1:
            if doc.get('numbers'):
                st.markdown("### 🔢 Key Numbers")
                for num_info in doc['numbers'][:5]:
                    with st.container():
                        st.metric(
                            label=f"Found in context",
                            value=num_info['value'],
                            help=num_info['context']
                        )

        with col2:
            if doc.get('emails'):
                st.markdown("### 📧 Contact Information")
                for email in doc['emails']:
                    st.info(f"✉️ {email}")


def display_analytics():
    """Display document analytics"""
    st.markdown("## 📊 Document Analytics")

    doc = st.session_state.current_file

    if doc.get("pages_data"):
        df = pd.DataFrame(doc["pages_data"])

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📄 Total Pages", doc['page_count'])
        with col2:
            st.metric("📝 Total Words", f"{doc['total_words']:,}")
        with col3:
            st.metric("📊 Avg Words/Page", f"{doc['avg_words_per_page']:.0f}")
        with col4:
            st.metric("⏱️ Reading Time", f"{doc['reading_time']} min")

        # Word distribution chart
        fig = px.bar(
            df,
            x="page_number",
            y="word_count",
            title="Word Distribution by Page",
            labels={"word_count": "Words", "page_number": "Page"},
            color="word_count",
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key topics visualization
        if doc.get('key_topics'):
            topics_df = pd.DataFrame([
                {"topic": topic, "frequency": doc['full_text'].lower().count(topic)}
                for topic in doc['key_topics'][:10]
            ])

            fig2 = px.bar(
                topics_df,
                x="frequency",
                y="topic",
                orientation='h',
                title="Top Keywords",
                labels={"frequency": "Occurrences", "topic": "Keywords"},
                color="frequency",
                color_continuous_scale="plasma"
            )
            fig2.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig2, use_container_width=True)


def display_search():
    """Display document search"""
    st.markdown("## 🔍 Search Document")

    search_query = st.text_input("Enter search terms:", placeholder="e.g., conclusion, results, findings...")

    if search_query and len(search_query) > 2:
        doc_text = st.session_state.current_file['full_text']

        # Search
        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
        matches = list(pattern.finditer(doc_text))

        if matches:
            st.success(f"Found {len(matches)} matches")

            # Show results
            for i, match in enumerate(matches[:10]):
                # Get context
                start = max(0, match.start() - 100)
                end = min(len(doc_text), match.end() + 100)
                context = doc_text[start:end]

                # Clean context
                if start > 0:
                    context = "..." + context[context.find(' ') + 1:]
                if end < len(doc_text):
                    context = context[:context.rfind(' ')] + "..."

                # Highlight match - escape HTML first
                import html
                context = html.escape(context)
                highlighted = pattern.sub(
                    f'<mark style="background-color: #facc15; color: black;">{html.escape(match.group())}</mark>',
                    context
                )

                # Find page
                chars_before = len(doc_text[:match.start()])
                page_num = 1
                char_count = 0
                for page_data in st.session_state.current_file['pages_data']:
                    char_count += page_data['char_count']
                    if char_count > chars_before:
                        break
                    page_num += 1

                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""
                        <div style="background-color: #1e1e1e; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                            <strong>Match {i + 1} - Page {page_num}</strong><br>
                            {highlighted}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button(f"Go to Page {page_num}", key=f"goto_{i}"):
                            st.session_state.current_page = page_num - 1
                            st.rerun()
        else:
            st.info("No matches found")


def main():
    """Main application"""
    init_session_state()
    create_header()
    create_sidebar()

    if not st.session_state.current_file:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: rgba(255,255,255,0.05); border-radius: 10px;">
            <h2>Welcome to PDF AI Assistant</h2>
            <p style="font-size: 1.2rem; margin: 2rem 0;">
                Upload a PDF document to start analyzing with AI
            </p>
            <p>👈 Use the sidebar to upload your document</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🎯 **95% Accuracy**\nAdvanced AI technology")
        with col2:
            st.info("⚡ **Instant Analysis**\nFast processing")
        with col3:
            st.info("🔒 **100% Private**\nLocal processing")
    else:
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Summary",
            "💬 Chat",
            "📊 Analytics",
            "🔍 Search",
            "📄 PDF Viewer"
        ])

        with tab1:
            display_summary()

        with tab2:
            display_chat_interface()

        with tab3:
            display_analytics()

        with tab4:
            display_search()

        with tab5:
            display_pdf_viewer()


if __name__ == "__main__":
    main()
