"""
Lightweight NIM LLM client abstraction.

This module centralizes the HTTP/SDK call to the NIM LLM so the
integration code in `mcp_integration.py` remains focused on
business logic. Keep this file lean and well-documented to make
it easy to mock in tests.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional, Dict, Any

try:
    import requests
except Exception:  # pragma: no cover - network deps
    requests = None  # guarded where used


def _extract_json_from_text(text: str) -> Optional[str]:
    """Return the first balanced JSON object found in `text` or None."""
    if not text or '{' not in text:
        return None

    for start_idx in range(len(text)):
        if text[start_idx] != '{':
            continue
        depth = 0
        for i in range(start_idx, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
    return None


def call_nim_llm(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_retries: int = 3,
    timeout_base: int = 45,
    max_tokens: int = 1500,
) -> Optional[Dict[str, Any]]:
    """
    Call the NIM LLM and return parsed JSON or None.

    This function uses requests directly to avoid OpenAI SDK issues
    with proxy settings.
    """
    def _err(msg: str, raw: Optional[str] = None) -> Dict[str, Any]:
        d = {'_error': msg}
        if raw:
            d['_raw_http_body'] = raw
        return d

    # Skip SDK entirely due to proxy issues
    print("⚠️ Using requests directly to avoid OpenAI SDK proxy issues")

    # Prepare system message for agricultural pathologist role
    system_msg = (
        'You are an expert agricultural pathologist. Respond with a single valid JSON object only (no additional text or markdown). '
        'The JSON must include these keys: "disease_profile" (array of 3-6 short bullet strings), "recommendations" (array), "prevention_strategies" (array), "danger_level" (string), "economic_impact" (string), "treatment_timeline" (string), and "monitoring_advice" (string). '
        'If MCP confidence is below 60% include optional keys: "differential_diagnosis" (array) and "confidence_notes" (string). '
        'Provide specific product names and dosages where possible. Return only JSON.'
    )

    if not requests:
        print("❌ requests library not available; cannot call NIM LLM")
        return _err('requests_unavailable')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': (
                'You are an expert agricultural pathologist. Respond with a single valid JSON object only (no additional text or markdown). '
                'The JSON must include these keys: "disease_profile" (array of 3-6 short bullet strings), "recommendations" (array), "prevention_strategies" (array), "danger_level" (string), "economic_impact" (string), "treatment_timeline" (string), and "monitoring_advice" (string). '
                'If MCP confidence is below 60% include optional keys: "differential_diagnosis" (array) and "confidence_notes" (string). '
                'Provide specific product names and dosages where possible. Return only JSON. '
                'CRITICAL: Do NOT include chain-of-thought or internal reasoning. Wrap the JSON exactly between <JSON> and </JSON> tags and return nothing outside those tags.'
            )},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.3,
        'top_p': 0.8,
        'max_tokens': max_tokens,
        'stream': False
    }

    response = None
    for attempt in range(max_retries):
        try:
            print(f"🔄 NIM API attempt {attempt + 1}/{max_retries} (no timeout)")
            # Do not pass timeout -> requests will wait until response or underlying socket timeout
            response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            break
        except Exception as e:
            # We intentionally don't special-case Timeout since no timeout param is used
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return _err(f'attempt_failed:{e}')
            time.sleep(0.5)
            continue

    if response is None:
        print("❌ No HTTP response received from NIM API")
        return _err('no_http_response')

    try:
        print(f"🔁 Raw HTTP status: {getattr(response, 'status_code', 'N/A')}")
        try:
            hdrs = dict(response.headers)
        except Exception:
            hdrs = str(response.headers)
        print(f"🔁 Raw response headers: {hdrs}")
        raw_body = getattr(response, 'text', '')
        print(f"🔁 Raw response body (truncated): {raw_body[:2000]}")
    except Exception as e:
        print(f"⚠️ Could not read raw response: {e}")
        raw_body = None

    if response.status_code != 200:
        print(f"NIM API error: {response.status_code} - {getattr(response, 'text', '')}")
        return _err(f'http_{response.status_code}', raw=getattr(response, 'text', None))

    try:
        result = response.json()
    except Exception as e:
        print(f"❌ Response JSON parsing failed: {e}")
        return _err(f'response_json_parse_failed:{e}', raw=raw_body)

    # Locate assistant content
    try:
        choice = result.get('choices', [{}])[0]
        message = choice.get('message', {})
        # Preserve reasoning_content separately when present (gpt-oss thinking)
        reasoning_content = message.get('reasoning_content')
        # Prefer explicit assistant content; delta may be present in streaming-like envelopes
        content = message.get('content') or message.get('delta')
        if not content or len(str(content).strip()) < 10:
            # If content is empty, we may still have reasoning content
            if not reasoning_content or len(str(reasoning_content).strip()) < 10:
                print("❌ NIM API returned empty or too-short content")
                return _err('empty_content', raw=raw_body)
            # use reasoning_content as fallback for parsing
            text = str(reasoning_content).strip()
        else:
            text = str(content).strip()
        # Strip markdown fences if present
        if text.startswith('```json'):
            text = text.replace('```json', '').replace('```', '').strip()
        elif text.startswith('```'):
            text = text.replace('```', '').strip()

        try:
            enhanced_data = json.loads(text)
            print("✅ Successfully parsed JSON response from NIM")
        except json.JSONDecodeError:
            print("⚠️ Strict JSON parse failed, attempting to extract JSON from text")
            candidate = _extract_json_from_text(text)
            if not candidate:
                print("❌ Could not extract JSON from assistant text")
                return _err('json_extract_failed', raw=text or raw_body)
            try:
                enhanced_data = json.loads(candidate)
                print("✅ Extracted JSON object from assistant prose")
            except Exception as e:
                print(f"⚠️ Extracted candidate JSON failed to parse: {e}")
                return _err(f'extracted_json_parse_failed:{e}', raw=text or raw_body)

        # Basic validation
        if not enhanced_data.get('recommendations'):
            print("❌ NIM returned no recommendations — treating as no enhancement")
            return _err('no_recommendations', raw=text or raw_body)

        # Attach raw HTTP envelope and assistant text for inspection
        if raw_body:
            enhanced_data['_raw_http_body'] = raw_body
        # Include reasoning content separately so callers can inspect chain-of-thought
        if reasoning_content:
            enhanced_data['_raw_reasoning_content'] = reasoning_content
        try:
            enhanced_data['_raw_http_json'] = result
        except Exception:
            pass
        # also provide the assistant text we parsed
        enhanced_data['_raw_assistant_text'] = text

        return enhanced_data
    except Exception as e:
        print(f"Error processing NIM response: {e}")
        return _err(f'processing_error:{e}')
