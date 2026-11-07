"""
gemini_extractor.py - Structured Entity Extraction using Google GenAI SDK.

Uses the official `google-genai` SDK to extract structured resume entities:
PERSON, SKILL, DEGREE, COMPANY.
Integrates with `privacy.py` to ensure all PII (emails, phone numbers) is masked
prior to external API dispatch.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.privacy import mask_resume_text

# Configure module logger for safe debugging (never logs PII or API credentials)
logger = logging.getLogger("ResumeAI.GeminiExtractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [ResumeAI.GeminiExtractor] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Known placeholder substrings that indicate an unconfigured key
PLACEHOLDER_SUBSTRINGS = [
    "your_gemini_api_key_here",
    "your_api_key_here",
    "your_key_here",
    "your_actual_gemini_api_key_here",
    "placeholder",
    "changeme",
    "enter_your_api_key",
    "todo",
]

# Supported official models known to support generate_content on Google AI Studio
VALID_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp",
]

DEFAULT_MODEL = "gemini-3.5-flash"


def sanitize_key(key: Optional[str]) -> Optional[str]:
    """
    Strips accidental leading/trailing whitespace, newlines, and surrounding quotes.
    Never logs or exposes the key value.
    """
    if not key:
        return None
    cleaned = key.strip()
    # Strip quotes if wrapped
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned if cleaned else None


def is_placeholder_key(key: Optional[str]) -> bool:
    """Checks whether an API key is a dummy or template placeholder."""
    if not key:
        return False
    lower_key = key.lower()
    return any(p in lower_key for p in PLACEHOLDER_SUBSTRINGS)


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_credentials_diagnostics() -> Dict[str, Any]:
    """
    Returns diagnostic inspection data for environment variables and paths.
    Never exposes the raw API key.
    """
    project_root = get_project_root()

    candidate_paths = [
        project_root / ".env",
        project_root / "ResumeAI" / ".env",
        project_root / "APIKEY.env",
        project_root / "ResumeAI" / "APIKEY.env",
    ]

    loaded_from = None
    for env_path in candidate_paths:
        if env_path.exists() and env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)
            if loaded_from is None:
                loaded_from = str(env_path)

    gemini_key_raw = os.getenv("GEMINI_API_KEY")
    google_key_raw = os.getenv("GOOGLE_API_KEY")

    active_key_raw = gemini_key_raw or google_key_raw
    active_key = sanitize_key(active_key_raw)

    key_source = "GEMINI_API_KEY" if gemini_key_raw else ("GOOGLE_API_KEY" if google_key_raw else "NONE")
    google_key_override_detected = bool(google_key_raw and gemini_key_raw and google_key_raw != gemini_key_raw)

    model_name_raw = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    model_name = sanitize_key(model_name_raw) or DEFAULT_MODEL

    is_placeholder = is_placeholder_key(active_key)

    return {
        "loaded_from": loaded_from,
        "key_source": key_source,
        "has_key": bool(active_key),
        "key_length": len(active_key) if active_key else 0,
        "is_placeholder": is_placeholder,
        "google_key_override_detected": google_key_override_detected,
        "model_name": model_name,
        "is_valid_model_name": model_name in VALID_GEMINI_MODELS,
    }


def load_api_credentials(
    with_diagnostics: bool = False
) -> Union[Tuple[Optional[str], str], Tuple[Optional[str], str, Dict[str, Any]]]:
    """
    Securely loads GEMINI_API_KEY and GEMINI_MODEL using explicit project-root paths.
    
    Checks GEMINI_API_KEY with priority over GOOGLE_API_KEY to prevent unintended overrides.
    Strips accidental whitespace and quotes.
    
    Args:
        with_diagnostics: If True, returns a 3-tuple (key, model, diagnostics_dict).
                          If False (default), returns a 2-tuple (key, model).
    
    Returns:
        (sanitized_api_key, model_name) or (sanitized_api_key, model_name, diagnostics)
    """
    diagnostics = get_credentials_diagnostics()
    gemini_key_raw = os.getenv("GEMINI_API_KEY")
    google_key_raw = os.getenv("GOOGLE_API_KEY")
    active_key = sanitize_key(gemini_key_raw or google_key_raw)
    model_name = diagnostics["model_name"]

    if with_diagnostics:
        return active_key, model_name, diagnostics
    return active_key, model_name


class ResumeEntitiesSchema(BaseModel):
    """Pydantic schema enforcing structured entity return from Gemini."""
    PERSON: List[str] = Field(
        default_factory=list,
        description="Full names of the resume candidate, references, or key persons"
    )
    SKILL: List[str] = Field(
        default_factory=list,
        description="Technical skills, soft skills, frameworks, tools, programming languages, and competencies"
    )
    DEGREE: List[str] = Field(
        default_factory=list,
        description="Degrees, diplomas, academic majors, certifications, and educational credentials"
    )
    COMPANY: List[str] = Field(
        default_factory=list,
        description="Companies, organizations, institutions, clients, or employers worked for"
    )


class GeminiExtractionError(Exception):
    """Base exception for Gemini extraction issues with friendly user diagnostics."""
    pass


def normalize_extracted_entities(parsed_data: Any) -> Dict[str, List[str]]:
    """
    Robustly normalizes raw JSON or dictionary from Gemini into the standard structure:
    {"PERSON": [...], "SKILL": [...], "DEGREE": [...], "COMPANY": [...]}

    Handles:
    - Missing keys (e.g., if Gemini only returned 2 fields like PERSON and SKILL).
    - Variations in key naming or casing (e.g. 'person', 'skills', 'education', 'companies').
    - Non-list values (single strings, comma-separated strings, numbers).
    - List-of-dicts formats (e.g. [{"entity": "PERSON", "value": "..."}, ...]).
    - Deduplication while preserving appearance order.
    """
    canonical: Dict[str, List[str]] = {
        "PERSON": [],
        "SKILL": [],
        "DEGREE": [],
        "COMPANY": [],
    }

    if not parsed_data:
        logger.warning("Parsed response data is empty. Returning blank entity collections.")
        return canonical

    # Flatten raw values into list of strings
    def to_string_list(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, str):
            clean_val = val.strip()
            if not clean_val:
                return []
            # If the LLM returned a comma-separated list in a string
            if "," in clean_val and "\n" not in clean_val:
                return [s.strip() for s in clean_val.split(",") if s.strip()]
            return [clean_val]
        if isinstance(val, (int, float)):
            return [str(val)]
        if isinstance(val, list):
            accum = []
            for item in val:
                accum.extend(to_string_list(item))
            return accum
        if isinstance(val, dict):
            # If the item is a dict like {"name": "Python"} or {"entity": "...", "value": "..."}
            sub_vals = []
            for sub_k in ["value", "name", "title", "text", "entity", "item"]:
                if sub_k in val and isinstance(val[sub_k], (str, int, float)):
                    sub_vals.append(str(val[sub_k]).strip())
            if sub_vals:
                return sub_vals
            # Fallback: flatten all string values of the dict
            return [str(v).strip() for v in val.values() if isinstance(v, (str, int, float)) and str(v).strip()]
        return []

    # Case 1: parsed_data is a single dictionary
    if isinstance(parsed_data, dict):
        key_mappings = {
            "PERSON": ["person", "persons", "name", "names", "candidate", "candidates", "full_name"],
            "SKILL": ["skill", "skills", "technologies", "technology", "tools", "competencies"],
            "DEGREE": ["degree", "degrees", "education", "qualification", "qualifications", "academics"],
            "COMPANY": ["company", "companies", "employer", "employers", "organization", "organizations", "experience"],
        }

        # Normalize keys to lowercase map for robust lookup
        lower_data = {str(k).lower().strip(): v for k, v in parsed_data.items()}

        for cat, synonyms in key_mappings.items():
            # Check exact match first
            if cat in parsed_data:
                canonical[cat] = to_string_list(parsed_data[cat])
            else:
                # Check lowercase synonyms
                found = False
                for syn in [cat.lower()] + synonyms:
                    if syn in lower_data:
                        canonical[cat] = to_string_list(lower_data[syn])
                        found = True
                        break
                if not found:
                    canonical[cat] = []

    # Case 2: parsed_data is a list of items (e.g. list of entity dictionaries)
    elif isinstance(parsed_data, list):
        for entry in parsed_data:
            if isinstance(entry, dict):
                # Subcase: [{"PERSON": [...]}, {"SKILL": [...]}]
                for cat in ["PERSON", "SKILL", "DEGREE", "COMPANY"]:
                    if cat in entry:
                        canonical[cat].extend(to_string_list(entry[cat]))
                    elif cat.lower() in entry:
                        canonical[cat].extend(to_string_list(entry[cat.lower()]))

                # Subcase: [{"category": "SKILL", "value": "Python"}]
                cat_label = entry.get("category") or entry.get("type") or entry.get("entity")
                val_data = entry.get("value") or entry.get("name") or entry.get("text")
                if cat_label and val_data:
                    cat_clean = str(cat_label).upper().strip()
                    if cat_clean in canonical:
                        canonical[cat_clean].extend(to_string_list(val_data))

    # Deduplicate while preserving order and case normalization
    cleaned_entities: Dict[str, List[str]] = {}
    for cat in ["PERSON", "SKILL", "DEGREE", "COMPANY"]:
        raw_items = canonical.get(cat, [])
        seen = set()
        deduped = []
        for item in raw_items:
            item_str = str(item).strip()
            if item_str and item_str.lower() not in seen:
                seen.add(item_str.lower())
                deduped.append(item_str)
        cleaned_entities[cat] = deduped

    return cleaned_entities


class GeminiExtractor:
    """Client for extracting structured resume entities using Google GenAI."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        env_key, env_model = load_api_credentials()
        self.diagnostics = get_credentials_diagnostics()
        self.api_key = sanitize_key(api_key) or env_key
        self.model_name = sanitize_key(model_name) or env_model or DEFAULT_MODEL
        self._client: Optional[genai.Client] = None

    def validate_key(self) -> None:
        """Validates that the key is present and not a placeholder before making API calls."""
        if not self.api_key:
            raise GeminiExtractionError(
                "Missing GEMINI_API_KEY: No API key was found in your environment or .env file.\n"
                "Please add your Gemini API key to .env (e.g. GEMINI_API_KEY=AIzaSy...).\n"
                "You can generate a free key at https://aistudio.google.com/."
            )
        if is_placeholder_key(self.api_key):
            raise GeminiExtractionError(
                "Placeholder GEMINI_API_KEY detected: Your .env file contains the default placeholder "
                "('your_gemini_api_key_here').\n"
                "Please replace it with your actual Google AI Studio API key.\n"
                "Get your key at: https://aistudio.google.com/."
            )

    def get_client(self) -> genai.Client:
        """Initializes or returns cached GenAI client."""
        self.validate_key()
        if self._client is None:
            try:
                # Initialize official Google GenAI SDK (explicitly API key, not Vertex AI)
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise GeminiExtractionError(f"Failed to initialize Google GenAI client: {e}") from e
        return self._client

    def test_connection(self) -> Tuple[bool, str]:
        """
        Validates API key and model connectivity with a lightweight ping prompt.
        
        Returns:
            (is_successful, status_message)
        """
        try:
            self.validate_key()
        except GeminiExtractionError as gee:
            return False, str(gee)

        try:
            client = self.get_client()
            logger.info("Testing Gemini connection to model: %s", self.model_name)
            response = client.models.generate_content(
                model=self.model_name,
                contents="Hello, reply with single word: OK",
            )
            if response and response.text:
                logger.info("Connection test succeeded for model: %s", self.model_name)
                return True, f"Connected to {self.model_name} successfully."
            return False, "Received empty response from Gemini API."
        except Exception as e:
            return False, self._format_error(e)

    def _format_error(self, e: Exception) -> str:
        """Translates low-level API exceptions into actionable diagnostic messages."""
        err_str = str(e)

        # 1. Invalid API Key
        if "API_KEY_INVALID" in err_str or ("400" in err_str and "API key not valid" in err_str):
            return (
                "400 INVALID_ARGUMENT: The provided API key is invalid.\n"
                "Please verify that you copied the complete key from Google AI Studio (https://aistudio.google.com/) "
                "without leading/trailing spaces or extraneous quotes."
            )

        # 2. Permission Denied / API Restrictions
        if "PERMISSION_DENIED" in err_str or "403" in err_str:
            return (
                "403 PERMISSION_DENIED: The API key does not have permission to access the Gemini API.\n"
                "Please check if API key restrictions (e.g., HTTP referrers, IP address, API limitations) "
                "are enabled in the Google Cloud Console, or generate a fresh key in Google AI Studio."
            )

        # 3. Model Not Found / Unavailable
        if "NOT_FOUND" in err_str or "404" in err_str:
            return (
                f"404 NOT_FOUND: Model '{self.model_name}' was not found or is unavailable for this API key.\n"
                f"Recommended supported models: {', '.join(VALID_GEMINI_MODELS)}.\n"
                f"Update GEMINI_MODEL={DEFAULT_MODEL} in your .env file."
            )

        # 4. Quota Exceeded / Rate Limit
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            return (
                "429 RESOURCE_EXHAUSTED: Gemini API rate limit or daily quota exceeded.\n"
                "Please wait a moment before retrying, or review quota tiers in Google AI Studio."
            )

        # 5. Service Unavailable / High demand
        if "503" in err_str or "UNAVAILABLE" in err_str:
            return (
                f"503 UNAVAILABLE: Model '{self.model_name}' is currently experiencing a temporary traffic spike.\n"
                f"Please wait a moment and retry, or switch to '{DEFAULT_MODEL}'."
            )

        # 6. Network / Connection errors
        if any(term in err_str for term in ["ConnectError", "Timeout", "gaierror", "getaddrinfo"]):
            return (
                f"Network error: Unable to reach Google Gemini API endpoints.\n"
                f"Please check your internet connection, proxy settings, or firewall."
            )

        return f"Gemini API request failed: {err_str}"

    def extract_entities(
        self,
        resume_text: str,
        apply_privacy_masking: bool = True
    ) -> Dict[str, Any]:
        """
        Extracts structured entities from resume text.
        
        1. Masks emails and telephone numbers (PII protection).
        2. Dispatches sanitized text to Gemini with structured JSON output schema.
        3. Validates and returns structured entity dictionary.
        
        Args:
            resume_text: Raw or pre-extracted resume text.
            apply_privacy_masking: Whether to mask emails and phones before dispatch.
            
        Returns:
            Dictionary with:
                - entities: {"PERSON": [...], "SKILL": [...], "DEGREE": [...], "COMPANY": [...]}
                - privacy: Audit metadata of masked PII
                - masked_text: The privacy-sanitized text sent to Gemini
                - raw_response: Parsed JSON structure
                - model_used: Name of the Gemini model used
        """
        if not resume_text or not resume_text.strip():
            raise GeminiExtractionError("Resume text is empty. No text to extract entities from.")

        logger.info("Initiating entity extraction (input length: %d chars)", len(resume_text))

        # 1. Privacy Masking
        if apply_privacy_masking:
            masked_text, privacy_meta = mask_resume_text(resume_text)
            logger.info(
                "Privacy filter applied: %d email(s), %d phone(s) masked",
                privacy_meta["emails_count"],
                privacy_meta["phones_count"],
            )
        else:
            masked_text = resume_text
            privacy_meta = {"emails_count": 0, "phones_count": 0, "masked": False}

        # 2. Validate Key & Prepare GenAI Client
        client = self.get_client()

        system_instruction = (
            "You are an expert resume parsing and Named Entity Recognition (NER) AI system. "
            "Analyze the candidate's resume text and accurately extract entities into four categories:\n"
            "1. PERSON: The full name of the resume candidate and any referenced individuals.\n"
            "2. SKILL: Hard technical skills, soft skills, programming languages, libraries, tools, "
            "frameworks, methodologies, and technical competencies.\n"
            "3. DEGREE: Academic degrees, diplomas, certificates, majors, and educational credentials "
            "(e.g., B.S. Computer Science, Master of Science, High School Diploma, AWS Certified Architect).\n"
            "4. COMPANY: Names of companies, organizations, employers, clients, or academic institutions "
            "where the candidate worked or interned.\n\n"
            "Rules:\n"
            "- Extract distinct, high-quality entities without duplicates.\n"
            "- Normalize entity capitalization where appropriate (e.g., 'Python', 'React', 'Google').\n"
            "- Do not invent or hallucinate entities not present or implied in the text.\n"
            "- Strictly output valid JSON matching the requested schema."
        )

        prompt = (
            f"Please extract all relevant entities (PERSON, SKILL, DEGREE, COMPANY) from the following resume text:\n\n"
            f"```\n{masked_text}\n```"
        )

        logger.info("Dispatching request to Gemini model: %s", self.model_name)

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=ResumeEntitiesSchema,
                    temperature=0.1,
                ),
            )
        except Exception as e:
            formatted_msg = self._format_error(e)
            logger.error("Gemini API call failed: %s", formatted_msg)
            raise GeminiExtractionError(formatted_msg) from e

        if not response or not response.text:
            raise GeminiExtractionError("Received an empty response from Gemini API.")

        raw_text = response.text.strip()
        logger.info("Received Gemini response (raw length: %d chars)", len(raw_text))

        # 3. Robust JSON Parsing (Handling code blocks, leading prose, trailing tokens)
        parsed_data = None
        cleaned_json_str = raw_text

        # Strip markdown fences if present
        if cleaned_json_str.startswith("```"):
            cleaned_json_str = re.sub(r"^```(?:json)?\s*", "", cleaned_json_str)
            cleaned_json_str = re.sub(r"\s*```$", "", cleaned_json_str)

        try:
            parsed_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError:
            # Fallback: extract substring between first { and last }
            json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if json_match:
                try:
                    parsed_data = json.loads(json_match.group(1))
                except json.JSONDecodeError as sub_jde:
                    raise GeminiExtractionError(
                        f"Failed to parse extracted JSON block: {sub_jde}. Response preview: {raw_text[:200]}"
                    ) from sub_jde
            else:
                raise GeminiExtractionError(
                    f"No valid JSON found in Gemini response. Response preview: {raw_text[:200]}"
                )

        # 4. Standardize and validate all 4 entity collections
        entities = normalize_extracted_entities(parsed_data)

        logger.info(
            "Extraction completed successfully: %d PERSON, %d SKILL, %d DEGREE, %d COMPANY",
            len(entities["PERSON"]),
            len(entities["SKILL"]),
            len(entities["DEGREE"]),
            len(entities["COMPANY"]),
        )

        return {
            "entities": entities,
            "privacy": privacy_meta,
            "masked_text": masked_text,
            "model_used": self.model_name,
            "raw_response": parsed_data,
        }


def extract_resume_entities(
    resume_text: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience functional interface for resume entity extraction."""
    extractor = GeminiExtractor(api_key=api_key, model_name=model_name)
    return extractor.extract_entities(resume_text)
