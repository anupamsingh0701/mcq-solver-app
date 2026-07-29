import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import re
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForMultipleChoice, utils

# Disable HuggingFace weight loading progress bar to prevent sys.stderr flush Errno 22 on Windows Streamlit
utils.logging.disable_progress_bar()



# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Multiple-Choice Answering System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS Styling
st.markdown("""
    <style>
    /* Global Container Styles */
    .main {
        padding-top: 1.5rem;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2b55 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    .header-card h1 {
        color: #6C5CE7;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-card p {
        color: #B2BEC3;
        font-size: 1.05rem;
    }

    /* Input Card */
    .input-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }

    /* Answer Result Badges */
    .top-answer-box {
        background: linear-gradient(135deg, #00b894 0%, #00bece 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 184, 148, 0.25);
    }
    .top-answer-box h2 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .metric-badge {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 6px;
    }

    /* Probability Bars */
    .stProgress > div > div > div > div {
        background-color: #6C5CE7;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HELPER FUNCTIONS & PREPROCESSING
# ==============================================================================
OPTIONS = ["A", "B", "C", "D", "E"]

WRAPPER_PREFIXES = re.compile(
    r"^(pick the best (possible )?answer\s*:|"
    r"determine the correct (option|answer)\s*:|"
    r"select the most (accurate|correct) (option|answer)\s*:|"
    r"identify the correct (statement|option|answer)\s*:|"
    r"choose the (best|correct) (option|answer)\s*:)\s*",
    re.IGNORECASE,
)
WRAPPER_SUFFIXES = re.compile(
    r"\s*(among the listed options|carefully|from the options given below)\.?\s*$",
    re.IGNORECASE,
)

def extract_core_question(prompt: str) -> str:
    """Strip boilerplate wrappers from prompt text."""
    p = str(prompt).strip()
    p = WRAPPER_PREFIXES.sub("", p)
    p = WRAPPER_SUFFIXES.sub("", p)
    return p.strip()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 4))
except Exception:
    pass


# ==============================================================================
# 3. MODEL LOADERS WITH CACHING
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_mcq_model(model_name_or_path: str, checkpoint_path: str = None, hf_token: str = None):
    """Load Transformer Tokenizer and MultipleChoice Model with auto-recovery and private repo auth."""
    import shutil
    import tqdm
    from huggingface_hub import login

    # Disable tqdm printer to prevent Windows sys.stderr flush Errno 22
    tqdm.tqdm.status_printer = staticmethod(lambda file: lambda *args, **kwargs: None)

    token = hf_token or os.environ.get("HF_TOKEN")

    if token:
        try:
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

    kw = {"token": token} if token else {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kw)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, **kw)
        except Exception:
            # Fallback to standard tokenizer if custom repo lacks tokenizer files
            if "deberta" in model_name_or_path.lower():
                tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
            else:
                tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    try:
        model = AutoModelForMultipleChoice.from_pretrained(model_name_or_path, **kw)
    except (ValueError, Exception) as err:
        # If custom repo lacks a valid config.json model_type (e.g., custom state_dict repo):
        # Fallback to standard base model and download custom weights from the repo!
        base_model_id = "microsoft/deberta-v3-large" if "deberta" in model_name_or_path.lower() else "LIAMF-USP/roberta-large-finetuned-race"
        model = AutoModelForMultipleChoice.from_pretrained(base_model_id)
        try:
            from huggingface_hub import hf_hub_download
            for weight_filename in ["model.safetensors", "pytorch_model.bin", "model.pt"]:
                try:
                    weight_path = hf_hub_download(repo_id=model_name_or_path, filename=weight_filename, **kw)
                    if weight_filename.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(weight_path)
                    else:
                        state_dict = torch.load(weight_path, map_location=DEVICE)
                    model.load_state_dict(state_dict, strict=False)
                    break
                except Exception:
                    continue
        except Exception:
            pass



    # Load custom trained checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model



def predict_mcq_single(prompt: str, choices: list, tokenizer, model, max_len: int = 256):
    """Predict logits and probabilities for a single MCQ question with dynamic padding."""
    core_question = extract_core_question(prompt)
    
    # Format inputs for HuggingFace MultipleChoice model
    texts_a = [core_question] * 5
    texts_b = [f"Answer: {c}" for c in choices]

    encoding = tokenizer(
        texts_a,
        texts_b,
        max_length=max_len,
        padding=True,  # Dynamic padding to actual length
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].unsqueeze(0).to(DEVICE)        # (1, 5, seq_len)
    attention_mask = encoding["attention_mask"].unsqueeze(0).to(DEVICE)

    kw = {"input_ids": input_ids, "attention_mask": attention_mask}

    if "token_type_ids" in encoding and "deberta" not in model.__class__.__name__.lower():
        kw["token_type_ids"] = encoding["token_type_ids"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(**kw)
        logits = outputs.logits.squeeze(0).cpu().numpy()  # shape: (5,)
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

    return logits





# ==============================================================================
# 4. SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=70)
    st.title("Model Settings")

    model_option = st.selectbox(
        "Select Model Architecture",
        [
            "Private Model (anupam211/roberta-large-race-mcq)",
            "RoBERTa-Large (RACE)",
            "DeBERTa-v3-Large",
            "Ensemble (DeBERTa + RoBERTa)",
            "HuggingFace Online Model / Private Repo"
        ],
        index=0
    )

    custom_repo = ""
    hf_token_input = ""

    if "Private" in model_option or "Online" in model_option:
        custom_repo = st.text_input("Hugging Face Model Repo ID", value="anupam211/roberta-large-race-mcq", help="e.g. your-username/my-private-model")
        hf_token_input = st.text_input("Hugging Face Access Token (HF_TOKEN)", type="password", help="Required for Private Repos (starts with hf_...)")


    st.markdown("---")
    st.subheader("Inference Settings")
    max_len = st.slider("Max Token Sequence Length", 64, 512, 256, step=32)
    temperature = st.slider("Softmax Temperature", 0.1, 2.0, 1.0, step=0.1)

    if "Ensemble" in model_option:
        deberta_weight = st.slider("DeBERTa Weight", 0.0, 1.0, 0.6, step=0.05)
        roberta_weight = 1.0 - deberta_weight
        st.caption(f"Weights -> DeBERTa: `{deberta_weight:.2f}` | RoBERTa: `{roberta_weight:.2f}`")


    st.markdown("---")
    st.caption(f"**Hardware Device:** `{DEVICE.type.upper()}`")
    if torch.cuda.is_available():
        st.caption(f"GPU: `{torch.cuda.get_device_name(0)}`")


# Determine Model Paths
DEBERTA_MODEL = "microsoft/deberta-v3-large"
RACE_MODEL = "LIAMF-USP/roberta-large-finetuned-race"
DEBERTA_CKPT = "deberta_large_best.pt"
RACE_CKPT = "race_roberta_best.pt"


# ==============================================================================
# 5. MAIN APPLICATION UI
# ==============================================================================
st.markdown("""
    <div class="header-card">
        <h1>🧠 MCQ AI Question Answering & Reasoning Engine</h1>
        <p>Fine-Tuned DeBERTa-v3 & RoBERTa Deep Learning Multiple-Choice Model Deployment</p>
    </div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📝 Single Question Answering", "📁 Batch CSV Predictions", "ℹ️ Model Info & Metrics"])

# ------------------------------------------------------------------------------
# TAB 1: SINGLE QUESTION INTERACTIVE INFERENCE
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Input Multiple-Choice Question & Options")

    # Sample Preset Loader
    sample_presets = {
        "Custom Input": None,
        "Physics Preset": {
            "prompt": "Which of the following principles best explains why an ice skater spins faster when pulling their arms inward?",
            "A": "Conservation of linear momentum",
            "B": "Conservation of angular momentum",
            "C": "Conservation of mechanical energy",
            "D": "Newton's second law of rotational motion",
            "E": "Reduction of gravitational potential energy"
        },
        "Computer Science Preset": {
            "prompt": "What is the worst-case time complexity of QuickSort when a poor pivot is consistently chosen?",
            "A": "O(n log n)",
            "B": "O(n)",
            "C": "O(n^2)",
            "D": "O(log n)",
            "E": "O(2^n)"
        },
        "Biology Preset": {
            "prompt": "Which organelle is primarily responsible for ATP synthesis during cellular respiration?",
            "A": "Endoplasmic Reticulum",
            "B": "Golgi Apparatus",
            "C": "Mitochondria",
            "D": "Lysosome",
            "E": "Nucleolus"
        }
    }

    preset_choice = st.selectbox("⚡ Load Sample Preset Question", list(sample_presets.keys()))
    preset_data = sample_presets[preset_choice]

    col_q, col_opts = st.columns([1.2, 1])

    with col_q:
        default_prompt = preset_data["prompt"] if preset_data else "What key process converts light energy into chemical energy in plants?"
        prompt_input = st.text_area("Question Prompt", value=default_prompt, height=140)

    with col_opts:
        st.write("**Candidate Choices (A-E):**")
        opt_a = st.text_input("Option A", value=preset_data["A"] if preset_data else "Glycolysis")
        opt_b = st.text_input("Option B", value=preset_data["B"] if preset_data else "Photosynthesis")
        opt_c = st.text_input("Option C", value=preset_data["C"] if preset_data else "Fermentation")
        opt_d = st.text_input("Option D", value=preset_data["D"] if preset_data else "Transpiration")
        opt_e = st.text_input("Option E", value=preset_data["E"] if preset_data else "Photophosphorylation")

    choices_list = [opt_a, opt_b, opt_c, opt_d, opt_e]

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("🚀 Predict Best Answer", type="primary", use_container_width=True)

    if btn_predict:
        if not prompt_input.strip() or any(not c.strip() for c in choices_list):
            st.error("⚠️ Please fill in the prompt and all 5 options before predicting.")
        else:
            with st.spinner("Analyzing question context & computing transformer logits..."):
                try:
                    if "Private" in model_option or "Online" in model_option:
                        target_model = custom_repo.strip() or DEBERTA_MODEL
                        tok, mod = load_mcq_model(target_model, hf_token=hf_token_input.strip() or None)
                        logits = predict_mcq_single(prompt_input, choices_list, tok, mod, max_len)
                    elif "DeBERTa" in model_option:
                        tok, mod = load_mcq_model(DEBERTA_MODEL, DEBERTA_CKPT if os.path.exists(DEBERTA_CKPT) else None, hf_token=hf_token_input.strip() or None)
                        logits = predict_mcq_single(prompt_input, choices_list, tok, mod, max_len)
                    elif "RoBERTa" in model_option:
                        tok, mod = load_mcq_model(RACE_MODEL, RACE_CKPT if os.path.exists(RACE_CKPT) else None, hf_token=hf_token_input.strip() or None)
                        logits = predict_mcq_single(prompt_input, choices_list, tok, mod, max_len)
                    elif "Ensemble" in model_option:
                        tok1, mod1 = load_mcq_model(DEBERTA_MODEL, DEBERTA_CKPT if os.path.exists(DEBERTA_CKPT) else None, hf_token=hf_token_input.strip() or None)
                        logits1 = predict_mcq_single(prompt_input, choices_list, tok1, mod1, max_len)
                        
                        tok2, mod2 = load_mcq_model(RACE_MODEL, RACE_CKPT if os.path.exists(RACE_CKPT) else None, hf_token=hf_token_input.strip() or None)
                        logits2 = predict_mcq_single(prompt_input, choices_list, tok2, mod2, max_len)

                        logits = (deberta_weight * logits1) + (roberta_weight * logits2)


                    # Numerically Stable Softmax Probabilities
                    scaled_logits = np.nan_to_num(logits / temperature, nan=0.0)
                    shifted_logits = scaled_logits - np.max(scaled_logits)
                    exp_logits = np.exp(shifted_logits)
                    probs = exp_logits / np.sum(exp_logits)
                    probs = np.nan_to_num(probs, nan=0.2)

                    # Top predictions
                    top_idx = int(np.argmax(probs))
                    top_option = OPTIONS[top_idx]
                    top_text = choices_list[top_idx]
                    top_prob = probs[top_idx] * 100.0

                    # Ranked indices for MAP@3 format
                    ranked_indices = np.argsort(probs)[::-1]
                    top3_str = " ".join([OPTIONS[i] for i in ranked_indices[:3]])

                    st.markdown("---")
                    st.markdown("### 🏆 Prediction Results")

                    res_col1, res_col2 = st.columns([1, 1.2])

                    with res_col1:
                        st.markdown(f"""
                            <div class="top-answer-box">
                                <p style="font-size: 1rem; opacity: 0.9; margin: 0;">RECOMMENDED CHOICE</p>
                                <h2>Option {top_option}</h2>
                                <p style="font-size: 1.1rem; margin-top: 6px; font-weight: 500;">"{top_text}"</p>
                                <div class="metric-badge">Model Confidence: {top_prob:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)

                        st.info(f"**MAP@3 Prediction Ranking:** `{top3_str}`")

                    with res_col2:
                        st.markdown("#### Probability Distribution")
                        for idx, opt in enumerate(OPTIONS):
                            p_val = float(np.clip(probs[idx], 0.0, 1.0))
                            is_best = (idx == top_idx)
                            label_prefix = "⭐️ " if is_best else ""
                            st.write(f"**{label_prefix}Option {opt}:** {choices_list[idx]} (`{p_val * 100:.2f}%`)")
                            st.progress(p_val)


                except Exception as e:
                    st.error(f"❌ Error during model inference: {str(e)}")
                    st.code(traceback.format_exc())



# ------------------------------------------------------------------------------
# TAB 2: BATCH CSV FILE INFERENCE
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Upload Batch Questions CSV File")
    st.caption("Required CSV Columns: `prompt`, `A`, `B`, `C`, `D`, `E` (Optional: `id`)")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write("#### Preview Uploaded Dataset:", df_batch.head())

        required_cols = {"prompt", "A", "B", "C", "D", "E"}
        if not required_cols.issubset(df_batch.columns):
            st.error(f"Missing required columns! Required: `{required_cols}`")
        else:
            if st.button("▶️ Run Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    tok, mod = load_mcq_model(DEBERTA_MODEL, DEBERTA_CKPT if os.path.exists(DEBERTA_CKPT) else None)
                    
                    preds_list = []
                    top_option_list = []

                    progress_bar = st.progress(0)
                    total_rows = len(df_batch)

                    for i, row in df_batch.iterrows():
                        choices = [str(row[o]) for o in OPTIONS]
                        logits = predict_mcq_single(str(row["prompt"]), choices, tok, mod, max_len)
                        
                        ranked_indices = np.argsort(logits)[::-1]
                        top3_str = " ".join([OPTIONS[idx] for idx in ranked_indices[:3]])
                        top_opt = OPTIONS[ranked_indices[0]]

                        preds_list.append(top3_str)
                        top_option_list.append(top_opt)
                        
                        progress_bar.progress((i + 1) / total_rows)

                    df_batch["top_choice"] = top_option_list
                    df_batch["prediction"] = preds_list

                    st.success("✅ Batch Inference Complete!")
                    st.dataframe(df_batch.head(10))

                    # Download CSV
                    csv_data = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Batch Predictions CSV",
                        data=csv_data,
                        file_name="mcq_predictions_output.csv",
                        mime="text/csv"
                    )


# ------------------------------------------------------------------------------
# TAB 3: MODEL INFO & ARCHITECTURE
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### 📊 Model Benchmark & Architecture Summary")
    st.markdown("""
    This web application deploys fine-tuned transformer architectures for multiple-choice question answering.
    
    #### Architecture Highlights:
    1. **DeBERTa-v3-Large (`microsoft/deberta-v3-large`)**:
       - Utilizes Disentangled Attention and Enhanced Masked Language Modeling.
       - Inputs are structured as pairwise prompt-choice concatenations: `[CLS] Prompt [SEP] Answer: Choice [SEP]`.
    2. **RoBERTa-Large RACE (`LIAMF-USP/roberta-large-finetuned-race`)**:
       - Pre-trained on reading comprehension exams and fine-tuned for MCQ classification.
    3. **Ensemble Blending**:
       - Combines logit outputs via weighted linear fusion for superior generalization and higher MAP@3 scores.
    """)
