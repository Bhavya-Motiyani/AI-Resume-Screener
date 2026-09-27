import os
import gc
import re
import json
import time
import atexit
import asyncio
import subprocess
import traceback
import urllib.request
import torch

from flask import Flask, request, jsonify, render_template

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from huggingface_hub import hf_hub_download
from pypdf import PdfReader

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"pdf"}


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MINICPM_MODEL = "openbmb/MiniCPM5-2B"

HR_MODEL = "dante557/HR-Recruiter-Llama-3.1-8B-v1"


# ------------------------------------------------------------
# GGUF models run through llama.cpp's llama-server.exe
# 4GB VRAM + 8GB RAM can't hold an 8B Llama through transformers, so
# llama.cpp puts some layers on the GPU and keeps the rest on the CPU.
# The small skill model fits entirely on the GPU.
# One server is started per model and stopped afterwards (frees VRAM).
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Official llama.cpp Windows CUDA build (see setup notes)
LLAMA_SERVER_EXE = os.path.join(BASE_DIR, "llama_bin", "llama-server.exe")
LLAMA_PORT = 8081
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"
LLAMA_LOG = os.path.join(BASE_DIR, "llama_server.log")

# Skill-extraction model: Qwen2.5-3B-Instruct (auto-downloaded once, ~2 GB).
# Small enough to sit completely on a 4 GB GPU, so this step is fast.
SKILL_GGUF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
SKILL_GGUF_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"

# 99 = put every layer on the GPU. The prompt only holds the job
# description, so a small context window is enough.
SKILL_GPU_LAYERS = 99
SKILL_CTX = 3072

# HR model: plain Llama-3.1-8B-Instruct GGUF by default (auto-downloaded).
# If you convert your own HR fine-tune to GGUF, put its path here.
HR_GGUF_LOCAL = None
HR_GGUF_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
HR_GGUF_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

SKILL_GGUF_PATH = None
HR_GGUF_PATH = None

# HR model (Llama 8B) only: layers kept on the GPU (it has ~33). Lower this
# if the server runs out of memory, raise it if nvidia-smi shows spare VRAM.
N_GPU_LAYERS = 20

# HR model context window (resume + job description + answer).
N_CTX = 5120

# Trim long inputs sent to the models so prompt + answer stay inside N_CTX.
# (Skill checking in Python always uses the full, untrimmed text.)
MAX_RESUME_CHARS = 7500
MAX_JD_CHARS = 3500


# ============================================================
# MCP CONFIGURATION
# ============================================================

EXA_MCP_URL = "https://mcp.exa.ai/mcp"


# ============================================================
# QUANTIZATION CONFIG
# ============================================================

def get_bnb_config():

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )


# ============================================================
# GPU MEMORY CLEANUP
# ============================================================

def unload_model(model=None, tokenizer=None):

    if model is not None:
        del model

    if tokenizer is not None:
        del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("\n[Memory] Model unloaded and GPU cache cleared.\n")


# ============================================================
# LLAMA.CPP SERVER HELPERS (skill model + HR model)
# ============================================================

_running_servers = []


def ensure_models():

    global SKILL_GGUF_PATH, HR_GGUF_PATH

    print("Checking GGUF model files (first run downloads ~7GB)...")

    SKILL_GGUF_PATH = hf_hub_download(
        SKILL_GGUF_REPO,
        SKILL_GGUF_FILE
    )

    if HR_GGUF_LOCAL and os.path.exists(HR_GGUF_LOCAL):
        HR_GGUF_PATH = HR_GGUF_LOCAL
    else:
        HR_GGUF_PATH = hf_hub_download(
            HR_GGUF_REPO,
            HR_GGUF_FILE
        )

    if not os.path.exists(LLAMA_SERVER_EXE):
        print(f"\n[WARNING] {LLAMA_SERVER_EXE} not found (see setup notes).\n")


def start_llama_server(model_path, lora_path=None, n_gpu_layers=None, n_ctx=None):

    if not os.path.exists(LLAMA_SERVER_EXE):
        raise FileNotFoundError(
            f"{LLAMA_SERVER_EXE} not found. "
            "Download the llama.cpp Windows CUDA build first."
        )

    cmd = [
        LLAMA_SERVER_EXE,
        "-m", model_path,
        "-ngl", str(n_gpu_layers if n_gpu_layers is not None else N_GPU_LAYERS),
        "-c", str(n_ctx if n_ctx is not None else N_CTX),
        "-b", "256",
        "-np", "1",
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT)
    ]

    if lora_path:
        cmd += ["--lora", lora_path]

    log_file = open(LLAMA_LOG, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )

    proc._log_file = log_file
    _running_servers.append(proc)

    # Wait until the model is loaded and the server answers /health
    deadline = time.time() + 900

    while time.time() < deadline:

        if proc.poll() is not None:
            stop_llama_server(proc)
            raise RuntimeError(
                "llama-server stopped while loading the model "
                f"(probably out of memory). See {LLAMA_LOG}. "
                "Try a lower N_GPU_LAYERS."
            )

        try:
            with urllib.request.urlopen(
                f"{LLAMA_URL}/health",
                timeout=2
            ) as response:

                if response.status == 200:
                    return proc

        except Exception:
            pass

        time.sleep(1)

    stop_llama_server(proc)

    raise TimeoutError("llama-server took too long to load the model.")


def stop_llama_server(proc):

    if proc is None:
        return

    if proc.poll() is None:

        proc.terminate()

        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if hasattr(proc, "_log_file"):
        proc._log_file.close()

    if proc in _running_servers:
        _running_servers.remove(proc)

    print("\n[Memory] llama-server stopped, VRAM freed.\n")


atexit.register(
    lambda: [stop_llama_server(p) for p in list(_running_servers)]
)


def chatml_prompt(text):

    # Prompt format used by Qwen models
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n"
        f"<|im_start|>user\n{text.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def llama3_prompt(text):

    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{text.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_json(prompt, max_tokens, schema=None):

    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "json_schema": schema or {"type": "object"}
    }

    request_obj = urllib.request.Request(
        f"{LLAMA_URL}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request_obj, timeout=1800) as response:
        result = json.loads(response.read())

    return result["content"].strip()


def parse_json_output(raw, label):

    # Printed so you can see exactly what the model produced
    print(f"\n{label} raw output:\n{raw}\n")

    text = raw.strip()

    # Drop ```json fences or text before the first {
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

    start = text.find("{")

    if start > 0:
        text = text[start:]

    # Try as-is first, then try to close JSON that was cut off
    # because the model hit the token limit
    for suffix in ("", "}", '"}', "]}", '"]}'):

        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError(
        f"{label} output is not valid JSON",
        text,
        0
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# PDF → TEXT
# ============================================================

def extract_resume_text(pdf_file):

    pdf = PdfReader(pdf_file)

    resume = []

    for i in range(len(pdf.pages)):

        text = pdf.pages[i].extract_text()

        if text:
            resume.append(text)

    return "\n".join(resume)


# ============================================================
# SKILL MATCHING
# SKILL EXTRACTION (small model)  ->  SKILL CHECK + ATS SCORE (code)
#
# The model only reads the job description and lists its skills.
# Python then checks each skill against the resume text and
# calculates the score, so the score never depends on the model
# guessing a number.
# ============================================================

# Different spellings that count as the same skill
SKILL_VARIANTS = [
    {"scikit learn", "sklearn", "scikit-learn"},
    {"google colab", "colab"},
    {"jupyter notebook", "jupyter"},
    {"git", "github"},
    {"sql", "mysql", "postgresql", "sqlite"},
    {"nlp", "natural language processing"},
    {"power bi", "powerbi"},
    {"javascript", "js"},
]

# Common skills, found by keyword in the job description as a safety net
# in case the model misses some. Add your own here any time.
COMMON_SKILLS = [
    # languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "SQL", "HTML",
    "CSS", "PHP", "Kotlin", "Scala", "MATLAB", "Bash",
    # data / ML concepts
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Science", "Data Analysis", "Data Visualization",
    "Data Preprocessing", "Feature Engineering", "Statistics", "Probability",
    "Neural Networks", "Time Series", "Generative AI", "LLM", "Transformers",
    "Reinforcement Learning", "A/B Testing",
    # libraries / frameworks
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn", "TensorFlow",
    "PyTorch", "Keras", "OpenCV", "XGBoost", "Hugging Face", "Spark",
    "Hadoop", "Airflow", "Kafka", "Flask", "Django", "FastAPI", "React",
    "Angular", "Node.js", "Spring Boot", "REST API",
    # tools / platforms
    "Jupyter Notebook", "Google Colab", "Git", "GitHub", "Docker",
    "Kubernetes", "Linux", "Jira", "Postman", "CI/CD",
    # databases
    "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis",
    # cloud
    "AWS", "Azure", "GCP", "Google Cloud",
    # BI / office
    "Excel", "Google Sheets", "Power BI", "Tableau", "Looker Studio",
    "MS Office", "Microsoft Office", "PowerPoint",
    # accounting / business
    "Tally", "GST", "TDS", "Income Tax", "Bookkeeping", "Accounting",
    "QuickBooks", "SAP", "ERP", "Salesforce", "CRM",
    # ways of working
    "Agile", "Scrum",
]

MIN_SKILLS_FOR_SCORE = 3
MAX_DISPLAY_MISSING_SKILLS = 5   # shown on the frontend / sent for course search
MAX_COURSE_LINKS_PER_SKILL = 2   # course links generated per skill
MAX_WORDS_PER_SKILL = 4
MAX_MODEL_SKILLS = 60

# Soft skills are not scored (they can't be checked by keyword)
SOFT_SKILLS = (
    "communication", "analytical", "problem solving", "teamwork",
    "collaborat", "leadership", "detail oriented", "eager",
    "willingness", "learning ability", "mentorship",
)

# Filler the model sometimes leaves in front of a skill
FILLER = re.compile(
    r"^(?:basic|working|strong|good|solid|prior|hands[- ]on|practical|"
    r"some|knowledge|understanding|familiarity|experience|proficiency|"
    r"expertise|exposure|skills?|of|with|in|on|and|to|a|an|the)"
    r"(?:\s+|\s*[,:]\s*)",
    re.IGNORECASE,
)

# The model must answer with {"skills": [...]} and nothing longer
SKILLS_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {"type": "string", "maxLength": 40},
            "maxItems": 40
        }
    },
    "required": ["skills"]
}

MAJOR_SKILLS_SCHEMA = {
    "type": "object",
    "properties": {
        "major_missing_skills": {
            "type": "array",
            "items": {"type": "string", "maxLength": 40},
            "maxItems": MAX_DISPLAY_MISSING_SKILLS
        }
    },
    "required": ["major_missing_skills"]
}


def _norm(text):
    # lower case, and treat hyphens / line breaks / spaces the same
    return re.sub(r"[\s\-_\u2010-\u2015]+", " ", str(text).lower()).strip()


def clean_skill(name):
    # "basic understanding of SQL" -> "SQL"
    name = re.sub(r"\(.*?\)", " ", str(name)).strip(" .,;:-")

    previous = None
    while previous != name:
        previous = name
        name = FILLER.sub("", name).strip(" .,;:-")

    return name


def _forms(term):
    for group in SKILL_VARIANTS:
        if term in {_norm(g) for g in group}:
            return {_norm(g) for g in group}
    return {term}


def _key(name):
    term = _norm(name)
    forms = _forms(term)
    return min(forms) if len(forms) > 1 else term


def _has(term, text):
    # term and text must already be normalised
    pattern = r"(?<![a-z0-9+#])" + re.escape(term) + r"(?![a-z0-9+#])"
    return re.search(pattern, text) is not None


def skill_in_text(skill, text):

    term = _norm(skill).strip(" .,;:")

    if len(term) < 2:
        return False

    text = _norm(text)

    return any(_has(form, text) for form in _forms(term))


# Names that contain "/" or "&" but are ONE skill
PROTECTED_SKILLS = {"ci/cd", "a/b testing", "ai/ml", "r&d", "tcp/ip", "pl/sql", "ui/ux"}


def split_skill(name):
    # "TensorFlow or PyTorch" -> ["TensorFlow", "PyTorch"]
    if _norm(name) in PROTECTED_SKILLS:
        return [name]

    parts = re.split(r"\s*(?:/|&|\bor\b|\band\b)\s*", name, flags=re.IGNORECASE)
    parts = [clean_skill(part) for part in parts]
    parts = [part for part in parts if part]

    # e.g. "A/B ..." would split into a single letter: keep it whole
    if not parts or any(len(part) < 2 for part in parts):
        return [name]

    return parts


COMMON_DISPLAY = {}


def display_name(name):
    # "sql" -> "SQL", "probability" -> "Probability"
    if not COMMON_DISPLAY:
        for term in COMMON_SKILLS:
            COMMON_DISPLAY[_norm(term)] = term

    shown = COMMON_DISPLAY.get(_norm(name), name)

    if shown.islower():
        shown = shown[:1].upper() + shown[1:]

    return shown


def find_skills(model_skills, job_description, resume):
    found, seen = [], set()

    def add(name):
        key = _key(name)
        if key not in seen:
            seen.add(key)
            found.append(display_name(name))

    # 1) Skills the model spotted. They must really be in the job description.
    for skill in list(model_skills or [])[:MAX_MODEL_SKILLS]:

        name = clean_skill(skill)

        # Empty, or a whole sentence instead of a skill name
        if not name or len(re.findall(r"[\w+#.]+", name)) > MAX_WORDS_PER_SKILL:
            continue

        # "Jupyter Notebook / Google Colab" -> two separate skills
        for part in split_skill(name):

            if any(word in _norm(part) for word in SOFT_SKILLS):
                continue

            if skill_in_text(part, job_description):
                add(part)

    # 2) Common skills found by keyword, in case the model missed some
    jd_norm = _norm(job_description)

    for term in COMMON_SKILLS:

        term_norm = _norm(term)

        if not _has(term_norm, jd_norm):
            continue

        # Already covered by a longer skill the model gave, e.g.
        # "Tally ERP / Accounting Software" already covers "Tally"
        if any(_has(term_norm, _norm(existing)) for existing in found):
            continue

        add(term)

    # 3) Check each one against the resume
    matching = [s for s in found if skill_in_text(s, resume)]
    missing = [s for s in found if not skill_in_text(s, resume)]

    return matching, missing


def score_from_skills(matching, missing):
    total = len(matching) + len(missing)
    if total < MIN_SKILLS_FOR_SCORE:
        return None
    return round(100 * len(matching) / total)


def pick_major_missing_skills(missing, job_description):

    if len(missing) <= MAX_DISPLAY_MISSING_SKILLS:
        return missing

    server = None

    prompt = f"""
Here is a job description and a list of skills the resume is missing.

JOB DESCRIPTION:
{job_description}

MISSING SKILLS:
{json.dumps(missing)}

Pick the {MAX_DISPLAY_MISSING_SKILLS} skills from the MISSING SKILLS list that matter most for this job.
Prefer skills the job description calls required, essential or must-have over skills
described as good-to-have, a plus, or preferred. If there are ties, prefer skills
mentioned earlier or more often in the job description.

Return only JSON in exactly this format, using skill names copied exactly as
they appear in MISSING SKILLS:
{{"major_missing_skills": ["skill 1", "skill 2", "skill 3", "skill 4", "skill 5"]}}
"""

    try:

        server = start_llama_server(
            SKILL_GGUF_PATH,
            n_gpu_layers=SKILL_GPU_LAYERS,
            n_ctx=SKILL_CTX
        )

        output = generate_json(
            chatml_prompt(prompt),
            max_tokens=200,
            schema=MAJOR_SKILLS_SCHEMA
        )

        result = parse_json_output(output, "Major missing skills")

        picked = [
            skill for skill in result.get("major_missing_skills", [])
            if skill in missing
        ]

        # Fill up to 5 if the model picked fewer, or fewer than 5 exist
        for skill in missing:
            if len(picked) >= MAX_DISPLAY_MISSING_SKILLS:
                break
            if skill not in picked:
                picked.append(skill)

        return picked[:MAX_DISPLAY_MISSING_SKILLS]

    except Exception:

        traceback.print_exc()

        # If picking fails for any reason, fall back to the first 5
        # rather than losing the whole analysis
        return missing[:MAX_DISPLAY_MISSING_SKILLS]

    finally:

        stop_llama_server(server)


def run_skill_matcher(resume, job_description, full_resume=None, full_job_description=None):

    print("\n========================================")
    print("Loading skill-extraction model (Qwen2.5-3B)")
    print("========================================")

    server = None

    try:

        server = start_llama_server(
            SKILL_GGUF_PATH,
            n_gpu_layers=SKILL_GPU_LAYERS,
            n_ctx=SKILL_CTX
        )

        # Only the job description goes to the model. The resume is
        # checked in Python, so the model cannot mix the two up.
        prompt = f"""
Extract the skills from the job description below.

JOB DESCRIPTION:
{job_description}

List every specific skill, tool, technology, programming language, framework, library, platform, software and qualification keyword that is written in the job description above. Copy each name as it is written there.

Rules:
- Only names that appear in the job description. Do not add anything else, even if it is common for this kind of job.
- Names only (for example "Python", "SQL", "TensorFlow"), not sentences.
- No duplicates.

Return only JSON in exactly this format:
{{"skills": ["skill 1", "skill 2", "skill 3"]}}
"""

        output = generate_json(
            chatml_prompt(prompt),
            max_tokens=400,
            schema=SKILLS_SCHEMA
        )

        result = parse_json_output(output, "Skill extraction")

        # Be tolerant of other key names the model might use
        model_skills = (
            list(result.get("skills", []))
            + list(result.get("matching_skills", []))
            + list(result.get("missing_skills", []))
        )

        # Checked against the FULL texts, not the trimmed prompt versions
        matching, missing = find_skills(
            model_skills,
            full_job_description or job_description,
            full_resume or resume
        )

        # The ATS score always uses every missing skill found above.
        # Only a short "major" list is surfaced to the frontend and used
        # for course search, chosen by the model out of that full list.
        major_missing = pick_major_missing_skills(missing, job_description)

        final_result = {
            "ats_score": score_from_skills(matching, missing),
            "missing_skills": major_missing,
            "all_missing_skills": missing,
            "explanation": (
                f"Matched {len(matching)} of {len(matching) + len(missing)} "
                "skills listed in the job description "
                f"({len(major_missing)} shown as the top missing skills)."
            )
        }

        print("\nMatching skills:", matching)
        print("Missing skills:", missing)

        print("\nSkill matching output:")
        print(json.dumps(final_result, indent=4))

        return final_result

    finally:

        stop_llama_server(server)


# ============================================================
# HR-RECRUITER LLAMA
# ATS FRIENDLINESS + INTERVIEW QUESTIONS
# ============================================================

def run_hr_recruiter(resume, job_description):

    print("\n========================================")
    print("Loading HR-Recruiter Llama")
    print("========================================")

    server = None

    try:

        server = start_llama_server(HR_GGUF_PATH)

        messages = [
            {
                "role": "user",
                "content": f"""
You are an expert at giving tips on how to increase ATS friendliness and giving out potential interview questions
based on the job description.

Analyze the following RESUME against the JOB DESCRIPTION.

Your tasks are:
1. Return ONLY valid JSON. Do not include markdown, explanations, or any text outside the JSON. Not even '''json.
2. Give description on how to make resume more ATS friendly and aligned to the Job Description.
3. Based on the resume and job description, generate the potential technical interview questions. (Top 5)

---
RESUME:
{resume}

JOB DESCRIPTION:
{job_description}

Return the result in exactly this JSON format:
"Increase_ATS_Friendliness":" ",
"technical_questions": ["question 1","question 2","question 3","question 4","question 5"]

---
Note:
1) Start with tips directly. No beginnings like 'To make this resume more friendly, do this' and similar kind of stuff
"""
            }
        ]

        output = generate_json(
            llama3_prompt(messages[0]["content"]),
            max_tokens=700
        )

        result = parse_json_output(output, "HR-Recruiter")

        final_result = {
            "Increase_ATS_Friendliness":
                result.get("Increase_ATS_Friendliness", ""),

            "technical_questions":
                result.get("technical_questions", [])
        }

        print("\nHR-Recruiter output:")
        print(json.dumps(final_result, indent=4))

        return final_result

    finally:

        stop_llama_server(server)


# ============================================================
# MINICPM + MCP + EXA
# ============================================================

def run_minicpm(missing_skills):

    if not missing_skills:

        print("\nNo missing skills. Skipping MiniCPM.")

        return {}

    print("\n========================================")
    print("Loading MiniCPM")
    print("========================================")

    tokenizer = None
    model = None

    try:

        bnb_config = get_bnb_config()

        tokenizer = AutoTokenizer.from_pretrained(
            MINICPM_MODEL
        )

        model = AutoModelForCausalLM.from_pretrained(
            MINICPM_MODEL,
            device_map="auto",
            quantization_config=bnb_config
        )

        exa_tool = {
            "type": "function",
            "function": {
                "name": "web_search_exa",
                "description": """Search the web for any topic and get clean, ready-to-use content.Best for finding current information, news, facts, people, companies,
                or answering questions about any topic.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        skill_list_text = "\n".join(
            f"{i + 1}. {skill}" for i, skill in enumerate(missing_skills)
        )

        messages = [
            {
                "role": "user",
                "content": f"""
Your task is to find relevant online courses or learning resources for EACH of the skills listed below.

SKILLS:
{skill_list_text}

You have access to a web search tool called `web_search_exa`.

For EVERY skill listed above, call `web_search_exa` exactly once with a clear search
query for that skill. Call the tool once per skill, in this same turn, in the same
order as the list above.

Each query MUST start with the exact skill name in square brackets, followed by a
short search phrase. For example, for the skill "Python" a query would be:
"[Python] best online course for beginners"

Requirements:
1. One tool call per skill. Do not skip any skill.
2. Every query must start with "[skill name]" using the skill exactly as written above.
3. Do not answer in text, only call the tool.
"""
            }
        ]

        # --------------------------------------------
        # ONE generation call for every skill, instead
        # of one call per skill (much faster)
        # --------------------------------------------

        inputs = tokenizer.apply_chat_template(
            messages,
            tools=[exa_tool],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        inputs.pop("token_type_ids", None)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200 * len(missing_skills),
            do_sample=False,
            use_cache=True
        )

        output = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:]
        )

        print("\nMiniCPM output:")
        print(output)

        # --------------------------------------------
        # Extract one Exa search query per skill
        # --------------------------------------------

        found_queries = re.findall(
            r'<function name="web_search_exa">'
            r'<param name="query">(.*?)</param>'
            r'</function>',
            output,
            re.DOTALL
        )

        queries_by_skill = {}

        for skill in missing_skills:

            tag = f"[{skill.strip().lower()}]"

            match = next(
                (q for q in found_queries if q.strip().lower().startswith(tag)),
                None
            )

            if match:
                queries_by_skill[skill] = match.strip()
            else:
                # MiniCPM did not tag a query for this skill (it can miss
                # some when asked for several at once) - fall back to a
                # plain query built in code, so every skill still gets
                # searched instead of coming back empty.
                print(f"No tagged query for '{skill}', using a default query.")
                queries_by_skill[skill] = f"{skill} course tutorial for beginners"

        # --------------------------------------------
        # Run all searches together instead of one at
        # a time
        # --------------------------------------------

        async def search_all():

            results = await asyncio.gather(
                *(search_exa_mcp(query) for query in queries_by_skill.values()),
                return_exceptions=True
            )

            return dict(zip(queries_by_skill.keys(), results))

        results_by_skill = asyncio.run(search_all())

        all_results = {}

        for skill, result in results_by_skill.items():

            if isinstance(result, Exception):

                print(f"Search failed for '{skill}': {result}")

                all_results[skill] = []
                continue

            # Extract URLs
            urls = re.findall(
                r'https?://[^\s\]\)]+',
                result
            )

            # Remove duplicates
            urls = list(dict.fromkeys(urls))

            # Maximum MAX_COURSE_LINKS_PER_SKILL URLs
            urls = urls[:MAX_COURSE_LINKS_PER_SKILL]

            all_results[skill] = urls

        print("\nMiniCPM final result:")
        print(json.dumps(all_results, indent=4))

        return all_results

    finally:

        unload_model(model, tokenizer)


async def search_exa_mcp(query):

    async with streamable_http_client(
        EXA_MCP_URL
    ) as (read_stream, write_stream):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                "web_search_exa",
                {
                    "query": query
                }
            )

            return result.content[0].text


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_resume(resume, job_description):

    print("\n\n========================================")
    print("STARTING RESUME ANALYSIS")
    print("========================================")

    # Keep prompts inside the context window / VRAM budget,
    # but remember the full texts for checking skills in Python
    full_resume = resume
    full_job_description = job_description

    resume = resume[:MAX_RESUME_CHARS]
    job_description = job_description[:MAX_JD_CHARS]

    # --------------------------------------------------------
    # STEP 1 — SKILLS + ATS SCORE
    # --------------------------------------------------------

    skill_result = run_skill_matcher(
        resume,
        job_description,
        full_resume,
        full_job_description
    )

    # --------------------------------------------------------
    # STEP 2 — HR RECRUITER
    # --------------------------------------------------------

    hr_result = run_hr_recruiter(
        resume,
        job_description
    )

    # --------------------------------------------------------
    # STEP 3 — MINICPM
    # --------------------------------------------------------

    # Only the top "major" missing skills get course links
    missing_skills = skill_result.get(
        "missing_skills",
        []
    )

    courses = run_minicpm(
        missing_skills
    )

    # --------------------------------------------------------
    # STEP 4 — FINAL JSON
    # --------------------------------------------------------

    final_result = {

        "ats_score":
            skill_result.get("ats_score"),

        # Top 5 skills shown on the frontend / used for course search
        "missing_skills":
            skill_result.get("missing_skills", []),

        # Every missing skill, used to calculate ats_score
        "all_missing_skills":
            skill_result.get("all_missing_skills", []),

        "Increase_ATS_Friendliness":
            hr_result.get(
                "Increase_ATS_Friendliness",
                ""
            ),

        "technical_questions":
            hr_result.get(
                "technical_questions",
                []
            ),

        "courses":
            courses
    }

    return final_result


# ============================================================
# FLASK API
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # ----------------------------------------------------
        # Check resume
        # ----------------------------------------------------

        if "resume" not in request.files:

            return jsonify({
                "success": False,
                "error": "Please upload a resume PDF."
            }), 400

        resume_file = request.files["resume"]

        if resume_file.filename == "":

            return jsonify({
                "success": False,
                "error": "No resume file was selected."
            }), 400

        if not allowed_file(resume_file.filename):

            return jsonify({
                "success": False,
                "error": "Only PDF resume files are supported."
            }), 400

        # ----------------------------------------------------
        # Check JD
        # ----------------------------------------------------

        job_description = request.form.get(
            "job_description",
            ""
        ).strip()

        if not job_description:

            return jsonify({
                "success": False,
                "error": "Please provide a job description."
            }), 400

        # ----------------------------------------------------
        # Extract resume text
        # ----------------------------------------------------

        print("\nExtracting resume text...")

        resume_text = extract_resume_text(
            resume_file
        )

        if not resume_text.strip():

            return jsonify({
                "success": False,
                "error": "Could not extract text from the uploaded PDF."
            }), 400

        # ----------------------------------------------------
        # Run complete pipeline
        # ----------------------------------------------------

        final_result = analyze_resume(
            resume_text,
            job_description
        )

        return jsonify({
            "success": True,
            "data": final_result
        })

    except json.JSONDecodeError:

        return jsonify({
            "success": False,
            "error": "A model returned invalid JSON. Please try again."
        }), 500

    except torch.cuda.OutOfMemoryError:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return jsonify({
            "success": False,
            "error": "GPU memory was insufficient while processing the resume."
        }), 500

    except Exception as e:

        print("\nERROR:")
        print(str(e))
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": "An error occurred while processing the resume.",
            "details": str(e)
        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    ensure_models()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
