"""
test_gemini.py - Diagnostic and Connection Verification for Google Gemini API.

Verifies:
1. Secure loading of GEMINI_API_KEY from explicit project-root .env.
2. Checks key presence and length (NEVER printing, logging, or exposing the key value).
3. Verifies that GOOGLE_API_KEY is not unintentionally overriding GEMINI_API_KEY.
4. Validates that the key is not an empty or default template placeholder.
5. Executes an official `google-genai` SDK request requesting the response "Gemini connected".
6. Provides clear diagnostic reporting for all error classes (400, 403, 404, 429, network).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gemini_extractor import (
    DEFAULT_MODEL,
    VALID_GEMINI_MODELS,
    GeminiExtractionError,
    GeminiExtractor,
    get_credentials_diagnostics,
    is_placeholder_key,
    load_api_credentials,
    sanitize_key,
)


def run_gemini_diagnostics() -> bool:
    print("=" * 65)
    print("RESUMEAI P30 - GEMINI API CONFIGURATION & CONNECTION DIAGNOSTICS")
    print("=" * 65)

    # 1. Inspect Environment Files and Paths
    diag = get_credentials_diagnostics()
    print("\n[STEP 1] Environment Inspection:")
    print(f"  • Project Root:          {PROJECT_ROOT}")
    print(f"  • .env File Loaded From: {diag.get('loaded_from') or 'No .env file found'}")
    print(f"  • Key Source Variable:   {diag.get('key_source')}")
    print(f"  • GOOGLE_API_KEY Clash:  {'YES (Conflict Detected)' if diag.get('google_key_override_detected') else 'None (Clean)'}")

    # 2. Key Security Verification (Never printing the value)
    print("\n[STEP 2] API Key Security & Validation:")
    has_key = diag.get("has_key", False)
    key_len = diag.get("key_length", 0)
    is_placeholder = diag.get("is_placeholder", False)

    print(f"  • API Key Present:       {'Yes' if has_key else 'No'}")
    print(f"  • API Key Length:        {key_len} characters")
    print(f"  • Is Placeholder String: {'YES (Action Required)' if is_placeholder else 'No'}")

    # 3. Model Configuration
    model_name = diag.get("model_name") or DEFAULT_MODEL
    is_valid_model = diag.get("is_valid_model_name", False)
    print("\n[STEP 3] Model Configuration:")
    print(f"  • Configured Model:      {model_name}")
    print(f"  • Valid Official Model:  {'Yes' if is_valid_model else f'Warning (Not in official list: {VALID_GEMINI_MODELS})'}")

    # 4. Preliminary Checks
    if not has_key:
        print("\n" + "=" * 65)
        print("❌ STATUS: MISSING API KEY")
        print("=" * 65)
        print("Reason: No GEMINI_API_KEY was found in your environment or .env file.")
        print("\nHow to fix:")
        print("  1. Generate a free API key at Google AI Studio: https://aistudio.google.com/")
        print("  2. Open the .env file in the project root.")
        print("  3. Set: GEMINI_API_KEY=AIzaSyYourActualKeyHere")
        print("  4. Save the file and re-run this script.")
        return False

    if is_placeholder:
        print("\n" + "=" * 65)
        print("❌ STATUS: PLACEHOLDER DETECTED (Error 400 Cause)")
        print("=" * 65)
        print("Reason: GEMINI_API_KEY in .env is currently set to a template placeholder ('your_gemini_api_key_here').")
        print("Google's API rejects placeholder keys with '400 INVALID_ARGUMENT - API key not valid'.")
        print("\nHow to fix:")
        print("  1. Visit Google AI Studio: https://aistudio.google.com/")
        print("  2. Click 'Get API key' and generate a new Gemini API key.")
        print("  3. Replace 'your_gemini_api_key_here' in your .env file with your real key.")
        print("  4. Save the file and re-run: python test_gemini.py")
        return False

    # 5. Live Connection Test
    print("\n[STEP 4] Initiating Live Gemini Connection Test...")
    print(f"  • Sending request: 'Reply with exactly: Gemini connected' using {model_name}...")

    try:
        from google import genai
        # Explicit 2-tuple unpacking (or call without unpacking)
        api_key, active_model = load_api_credentials()
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=active_model,
            contents="Reply with exactly: Gemini connected",
        )

        reply_text = response.text.strip() if response and response.text else ""
        print(f"  • Response received: '{reply_text}'")

        if "Gemini connected" in reply_text or len(reply_text) > 0:
            print("\n" + "=" * 65)
            print("✅ STATUS: SUCCESS - GEMINI API CONNECTED!")
            print("=" * 65)
            print(f"Verified connection to {active_model}.")
            print("ResumeAI is ready for real entity extraction.")
            return True
        else:
            print("\n" + "=" * 65)
            print("⚠️ STATUS: RECEIVED EMPTY OR UNEXPECTED RESPONSE")
            print("=" * 65)
            return False

    except Exception as e:
        print("\n" + "=" * 65)
        print("❌ STATUS: CONNECTION FAILED")
        print("=" * 65)
        extractor = GeminiExtractor(model_name=model_name)
        formatted_error = extractor._format_error(e)
        print(formatted_error)
        return False


if __name__ == "__main__":
    success = run_gemini_diagnostics()
    sys.exit(0 if success else 1)
