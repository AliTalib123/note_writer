import streamlit as st
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import io
import numpy as np
import soundfile as sf
import re
import pypdf
import torch
import wordninja

@st.cache_resource
def load_models():
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base"
    )
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("./bart_cnn_checkpoint", local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return asr_pipe, tokenizer, model, device

asr_pipe, tokenizer, model, device = load_models()

def fix_glued_words(text):
 
    def fix_token(token):
        # Skip short words, hyphenated, numeric, or already normal
        if len(token) <= 5 or '-' in token or token.isnumeric():
            return token
        # If token is all alpha and long, try splitting it
        if re.fullmatch(r'[a-zA-Z]{6,}', token):
            parts = wordninja.split(token)
            # Only apply if split produced more than 1 part and all parts are real (len > 1)
            if len(parts) > 1 and all(len(p) > 1 for p in parts):
                return ' '.join(parts)
        return token

    words = text.split()
    return ' '.join(fix_token(w) for w in words)


def clean_text(text):
    # Fix OCR mid-word spaces: "self -driving" → "self-driving"
    text = re.sub(r'(\w{3,})\s{1,2}-\s{1,2}(\w+)', r'\1-\2', text)
    # camelCase split: "WorldImpact" → "World Impact"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    # Strip non-ASCII garbage (ڃ etc.)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Strip URLs
    text = re.sub(r"http\S+", "", text)
    # Strip filler words
    fillers = ["um", "uh", "you know", "basically"]
    pattern = r'\b(?:' + '|'.join(fillers) + r')\b'
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()


    text = fix_glued_words(text)
    text = re.sub(r"\s+", " ", text).strip()

   
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen, deduped = set(), []
    for s in sentences:
        key = re.sub(r'\s+', ' ', s.strip().lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(s.strip())
    return ' '.join(deduped)

def postprocess_summary(summary):
    summary = summary.strip()
    summary = re.sub(r"^(in this (letter|paper|class|video)[, ]*)", "", summary, flags=re.IGNORECASE)
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    return summary


def chunk_text(text, max_words=900):
   
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk.split()) + len(sent.split()) <= max_words:
            chunk += sent + " "
        else:
            if chunk:
                chunks.append(chunk.strip())
            chunk = sent + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def summarize_text(text):
    if len(text.split()) < 20:
        return text

    total_words = len(text.split())
    target_words = int(total_words * 0.40)   # 40% of original

    chunks = chunk_text(text)
    target_per_chunk = max(56, target_words // len(chunks))

    max_len = min(512, int(target_per_chunk * 1.3))
    min_len = max(40, int(target_per_chunk * 0.9))  # tight band around target

    partial_summaries = []
    for c in chunks:
        inputs = tokenizer(c, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                num_beams=5,
                do_sample=False,
                length_penalty=1.0,   # neutral — don't push longer or shorter
                no_repeat_ngram_size=3,
                early_stopping=False
            )
        partial_summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    combined = " ".join(partial_summaries)

    
    actual_words = len(combined.split())
    ratio = actual_words / total_words if total_words > 0 else 0
    if ratio < 0.25:
        combined += (
            f"\n\n⚠️ Note: Summary is {actual_words} words "
            f"({ratio:.0%} of original {total_words} words). "
            "The input may be too short or too dense for a longer summary."
        )

    return postprocess_summary(combined)


def split_audio(data, samplerate, chunk_sec=30):
    chunk_len = chunk_sec * samplerate
    return [data[i:i+chunk_len] for i in range(0, len(data), chunk_len)]



mode = st.sidebar.radio(
    "Choose Mode",
    ["Lecture Summarizer", "Document Summarizer"]
)


if mode == "Lecture Summarizer":
    st.title("🎙️ Tongxue")
    if "lecture_summary" not in st.session_state:
        st.session_state.lecture_summary = ""
    if "lecture_word_count" not in st.session_state:
        st.session_state.lecture_word_count = 0

    audio_value = st.audio_input("Record to summarize")
    if audio_value and st.button("Process Lecture"):
        with st.spinner("Transcribing audio..."):
            audio_bytes = audio_value.read()
            audio_buffer = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(audio_buffer)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            chunks = split_audio(data, samplerate, chunk_sec=30)
            transcription = ""
            for chunk in chunks:
                result = asr_pipe(chunk)
                transcription += result["text"] + " "
            transcription = clean_text(transcription.strip())
            st.session_state.lecture_word_count = len(transcription.split())

        with st.spinner("Summarizing..."):
            st.session_state.lecture_summary = summarize_text(transcription)

    if st.session_state.lecture_summary:
        st.subheader("Summary")
        st.write(st.session_state.lecture_summary)
        file_name = st.text_input("Enter file name (must end with .txt)")
        if file_name and file_name.strip().endswith(".txt"):
            txt_data = io.BytesIO(st.session_state.lecture_summary.encode("utf-8"))
            txt_data.seek(0)
            st.download_button("⬇️ Download Lecture Summary", txt_data, file_name.strip(), "text/plain")


elif mode == "Document Summarizer":
    st.title("📄 Wenjian")

    if "doc_summary" not in st.session_state:
        st.session_state.doc_summary = ""
    if "doc_word_count" not in st.session_state:
        st.session_state.doc_word_count = 0
    if "doc_last_words" not in st.session_state:
        st.session_state.doc_last_words = ""

    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"], accept_multiple_files=False)

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            if uploaded_file.type == "application/pdf":
                reader = pypdf.PdfReader(uploaded_file)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                raw_text = " ".join(pages_text)
            else:
                raw_text = uploaded_file.read().decode("utf-8")

            word_count = len(raw_text.split())
            st.session_state.doc_word_count = word_count
            st.session_state.doc_last_words = raw_text.strip()[-300:]

            text = clean_text(raw_text)

            if len(text.split()) < 20:
                st.session_state.doc_summary = "⚠️ Document is too short to summarize."
            else:
                st.session_state.doc_summary = summarize_text(text)

    if st.session_state.doc_summary:
        word_count = st.session_state.get("doc_word_count", 0)
        last_words = st.session_state.get("doc_last_words", "")

        # Warn if document appears cut off
        if last_words and not last_words.rstrip().endswith(('.', '!', '?', '"', "'")):
            st.warning(
                "⚠️ Your document appears to be **cut off mid-sentence**. "
                "Make sure your file is complete before uploading."
            )
            with st.expander("See where your document ends"):
                st.text("..." + last_words)

        st.subheader("Summary")
        st.write(st.session_state.doc_summary)
        file_name = st.text_input("Enter file name (must end with .txt)")
        if file_name and file_name.strip().endswith(".txt"):
            txt_data = io.BytesIO(st.session_state.doc_summary.encode("utf-8"))
            txt_data.seek(0)
            st.download_button("⬇️ Download Document Summary", txt_data, file_name.strip(), "text/plain")
            





