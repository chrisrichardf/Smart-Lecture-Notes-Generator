from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from sklearn.feature_extraction.text import TfidfVectorizer


# ─────────────────────────────────────────────
# Startup: download NLTK data once
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        nltk.download(pkg, quiet=True)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────
class RequestData(BaseModel):
    url: str


# ─────────────────────────────────────────────
# Utility: extract YouTube video ID robustly
# ─────────────────────────────────────────────
def get_video_id(url: str) -> str:
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.strip()  # fallback: assume raw ID


# ─────────────────────────────────────────────
# Step 1: Clean transcript text
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    # Remove [Music], [Applause], (laughter) etc.
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    # Transcripts often have no space after a period when sentences are joined
    # e.g. "Hello world.This is" → "Hello world. This is"
    text = re.sub(r"([a-z]{2,})([A-Z])", r"\1. \2", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Capitalize the start of each sentence fragment split by period
    parts = text.split(". ")
    text = ". ".join(p.strip().capitalize() for p in parts if p.strip())

    return text.strip()


# ─────────────────────────────────────────────
# Step 2: Split into clean sentences via NLTK
# ─────────────────────────────────────────────
def get_sentences(text: str) -> list[str]:
    try:
        raw_sentences = sent_tokenize(text)
    except Exception:
        # Fallback to regex if NLTK fails
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    result = []
    for s in raw_sentences:
        s = s.strip()
        # Skip too-short fragments (likely noise)
        if len(s.split()) < 8:
            continue
        # Ensure sentence starts with a capital letter
        s = s[0].upper() + s[1:] if s else s
        result.append(s)

    return result


# ─────────────────────────────────────────────
# Step 3a: Score sentences using TF-IDF
#   Higher score = sentence contains more
#   informative, content-rich terms
# ─────────────────────────────────────────────
def score_sentences_tfidf(sentences: list[str]) -> list[tuple[int, float, str]]:
    """
    Returns list of (original_index, tfidf_score, sentence).
    Sentences with rare but frequent-within-doc terms score higher.
    """
    if len(sentences) < 2:
        return [(i, 1.0, s) for i, s in enumerate(sentences)]

    try:
        stop_words = stopwords.words("english")
    except Exception:
        stop_words = "english"

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=300,
        sublinear_tf=True,       # dampen very frequent terms
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Sum of TF-IDF weights per sentence = its "information density"
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        return [(i, float(scores[i]), sentences[i]) for i in range(len(sentences))]
    except Exception:
        return [(i, 1.0, s) for i, s in enumerate(sentences)]


# ─────────────────────────────────────────────
# Step 3b: Extract top keywords / keyphrases
# ─────────────────────────────────────────────
def extract_keywords(sentences: list[str], top_n: int = 8) -> list[str]:
    """
    Uses TF-IDF with bigrams to pull out the most important terms.
    e.g. "machine learning", "neural network", "gradient descent"
    """
    text = " ".join(sentences)
    try:
        stop_words = stopwords.words("english")
    except Exception:
        stop_words = "english"

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=top_n,
        ngram_range=(1, 2),      # single words + two-word phrases
        min_df=1,
    )
    try:
        vectorizer.fit_transform([text])
        return list(vectorizer.get_feature_names_out())
    except Exception:
        return []


# ─────────────────────────────────────────────
# Generator: Smart Summary via LexRank
#   LexRank = graph-based ranking (like PageRank
#   for sentences). Picks sentences that are
#   "central" to the whole document's meaning.
# ─────────────────────────────────────────────
def generate_summary(text: str, sentence_count: int = 3) -> str:
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        try:
            summarizer.stop_words = stopwords.words("english")
        except Exception:
            pass
        top_sentences = summarizer(parser.document, sentence_count)
        summary = " ".join(str(s) for s in top_sentences)
        if summary.strip():
            return summary
    except Exception:
        pass

    # Fallback: first 3 clean sentences
    fallback = get_sentences(text)
    return " ".join(fallback[:3])


# ─────────────────────────────────────────────
# Generator: Notes (top sentences in reading order)
# ─────────────────────────────────────────────
def generate_notes(scored: list[tuple], top_n: int = 10) -> str:
    # Pick top-N by TF-IDF score
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    # Restore original reading order
    top = sorted(top, key=lambda x: x[0])

    lines = []
    seen_prefixes: set[str] = set()
    for _, _, sentence in top:
        key = sentence[:50].lower()
        if key not in seen_prefixes:
            seen_prefixes.add(key)
            lines.append(f"• {sentence}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Generator: Key Points (ranked by importance)
# ─────────────────────────────────────────────
def generate_key_points(scored: list[tuple], keywords: list[str], top_n: int = 6) -> str:
    # Take top-N sentences ranked by TF-IDF (importance order, not reading order)
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

    lines = []
    seen_prefixes: set[str] = set()
    for _, _, sentence in top:
        key = sentence[:50].lower()
        if key not in seen_prefixes:
            seen_prefixes.add(key)
            lines.append(f"🔹 {sentence}")

    # Append extracted key concepts as a summary line
    if keywords:
        formatted_kw = ", ".join(kw.title() for kw in keywords[:6])
        lines.append(f"\n📌 **Key Concepts:** {formatted_kw}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Generator: Context-aware Questions
#   Uses actual keywords from the video, not
#   generic templates
# ─────────────────────────────────────────────
def generate_questions(sentences: list[str], keywords: list[str]) -> str:
    questions: list[str] = []

    # 1. Keyword-specific questions (actual content from the video)
    for kw in keywords[:4]:
        kw_display = kw.title()
        questions.append(f"❓ What role does **{kw_display}** play in the topic discussed?")

    # 2. Sentence-anchored question from an early key sentence
    if sentences:
        anchor = sentences[0][:70].rstrip(",;:")
        questions.append(f'❓ Based on "{anchor}...", what conclusion can be drawn?')

    # 3. Higher-order thinking questions (always useful, but fewer now)
    questions += [
        "❓ What is the central argument or message of this video?",
        "❓ What real-world applications are discussed or implied?",
        "❓ What assumptions does the speaker make?",
        "❓ What would challenge or contradict the main points made?",
    ]

    # Cap at 8 questions to avoid bloat
    return "\n".join(questions[:8])


# ─────────────────────────────────────────────
# Main API endpoint
# ─────────────────────────────────────────────
@app.post("/process")
async def process(data: RequestData):
    try:
        video_id = get_video_id(data.url)

        transcript = YouTubeTranscriptApi().fetch(video_id)
        raw_text = " ".join([t.text for t in transcript])

        cleaned = clean_text(raw_text)
        sentences = get_sentences(cleaned)

        if not sentences:
            return JSONResponse(
                content={"error": "Could not extract meaningful sentences from transcript."},
                status_code=422,
            )

        # Core NLP passes — both use TF-IDF, computed once via scoring
        scored = score_sentences_tfidf(sentences)
        keywords = extract_keywords(sentences, top_n=8)

        # Generate all four outputs
        summary    = generate_summary(cleaned, sentence_count=3)
        notes      = generate_notes(scored, top_n=10)
        key_points = generate_key_points(scored, keywords, top_n=6)
        questions  = generate_questions(sentences, keywords)

        output = f"""# 📘 Notes
{notes}

# 🧾 Summary
{summary}

# 🔑 Key Points
{key_points}

# ❓ Questions to Consider
{questions}
"""

        return {"notes": output}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)