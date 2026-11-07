"""
privacy.py - PII Masking and Privacy Protection for Resume Processing.

Ensures sensitive contact details (emails, phone numbers) are masked before
sending text to external APIs, preserving user privacy.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# Regex pattern for email addresses
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
)

# Comprehensive regex patterns for phone numbers:
# Covers formats like:
# +1-555-123-4567, (555) 123-4567, +91 9876543210, 98765 43210,
# +44 20 7946 0958, 555.123.4567, +1 (555) 000-0000
PHONE_PATTERN = re.compile(
    r'(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}(?:[\s.-]?\d{1,5})?)'
)

# Strict phone pattern to avoid matching ordinary years (e.g., 2018-2022) or numerical stats
PHONE_STRICT_PATTERN = re.compile(
    r'(?<!\w)(?:'
    # International format with + prefix: +1-234-567-8901, +91 9876543210
    r'(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}'
    # Standalone standard phone numbers with punctuation or brackets: (123) 456-7890, 123-456-7890
    r'|(?:\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4})'
    r'|(?:\b\d{3}[-.]\d{3}[-.]\d{4}\b)'
    # 10 or 11 digit contiguous numbers prefixed with Tel/Phone/Mob/Cell or common start
    r'|(?:\b\d{5}[\s.-]?\d{5}\b)'
    r'|(?:\b(?:\+?91|0)?[6-9]\d{9}\b)'
    r')(?!\w)'
)


class PrivacyMasker:
    """Masks Personally Identifiable Information (PII) from resume text."""

    def __init__(
        self,
        email_placeholder: str = "[EMAIL_REDACTED]",
        phone_placeholder: str = "[PHONE_REDACTED]",
    ):
        self.email_placeholder = email_placeholder
        self.phone_placeholder = phone_placeholder

    def mask_emails(self, text: str) -> Tuple[str, List[str]]:
        """Finds and replaces all email addresses with the email placeholder."""
        found = EMAIL_PATTERN.findall(text)
        masked = EMAIL_PATTERN.sub(self.email_placeholder, text)
        return masked, found

    def mask_phone_numbers(self, text: str) -> Tuple[str, List[str]]:
        """Finds and replaces all telephone numbers with the phone placeholder."""
        # Find candidates using strict regex
        matches = list(PHONE_STRICT_PATTERN.finditer(text))
        found = []
        
        # Sort matches in reverse order so replacements do not distort indices
        matches.sort(key=lambda m: m.start(), reverse=True)
        masked_text = text
        for m in matches:
            val = m.group(0).strip()
            # Ignore standard 4-digit years like 2020 or year ranges like 2018-2022
            if re.fullmatch(r'\d{4}', val) or re.fullmatch(r'\d{4}[-\s]\d{4}', val):
                continue
            # Strip excessive length or short false positives
            digits_only = re.sub(r'\D', '', val)
            if len(digits_only) < 7 or len(digits_only) > 15:
                continue
            found.append(val)
            start, end = m.start(), m.end()
            masked_text = masked_text[:start] + self.phone_placeholder + masked_text[end:]
            
        return masked_text, found

    def mask_pii(self, text: str) -> Tuple[str, Dict[str, any]]:
        """
        Masks both emails and phone numbers from resume text before external API submission.
        
        Returns:
            masked_text (str): Cleaned text safe for LLM consumption.
            metadata (dict): Audit trail with counts and redacted item categories.
        """
        if not text:
            return "", {"emails_count": 0, "phones_count": 0, "masked": False}

        # Mask emails first to avoid any digits in emails being caught by phone regex
        text_masked_email, emails = self.mask_emails(text)
        text_masked_all, phones = self.mask_phone_numbers(text_masked_email)

        metadata = {
            "emails_count": len(emails),
            "phones_count": len(phones),
            "total_redactions": len(emails) + len(phones),
            "masked": (len(emails) + len(phones)) > 0,
            "detected_emails": emails,
            "detected_phones": phones,
        }

        return text_masked_all, metadata


def mask_resume_text(text: str) -> Tuple[str, Dict[str, any]]:
    """Convenience functional interface for PII masking."""
    masker = PrivacyMasker()
    return masker.mask_pii(text)
