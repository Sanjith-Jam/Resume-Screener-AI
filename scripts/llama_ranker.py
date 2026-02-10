"""
LLM-powered ranking and feedback engine using Llama 3.1 via Ollama.
Produces detailed, reasoned rankings with per-resume feedback.
"""

import json
import re
import ollama


SYSTEM_PROMPT = """You are an expert technical recruiter and hiring manager with 20+ years 
of experience evaluating candidates. You provide precise, evidence-based 
assessments grounded in the actual content of each resume relative to a 
specific job description.

RULES:
- Base every claim on concrete evidence from the resume text.
- Never fabricate skills or experience not present in the resume.
- When comparing resumes, cite specific differences (e.g. "Resume A lists 
  5 years of Python vs Resume B's 2 years").
- Be direct and constructive in feedback; avoid vague praise."""


def _build_ranking_prompt(
    jd_text: str,
    resumes: list[dict],
    similarity_results: list[dict],
) -> str:
    """
    Build the user prompt for Llama. Includes the JD, each resume's text
    (truncated to ~2000 words to fit context), and the BERT similarity scores
    as quantitative evidence.
    """
    prompt_parts = []

    prompt_parts.append("=" * 60)
    prompt_parts.append("JOB DESCRIPTION")
    prompt_parts.append("=" * 60)
    prompt_parts.append(jd_text[:6000])
    prompt_parts.append("")

    for i, resume in enumerate(resumes):
        sim = similarity_results[i] if i < len(similarity_results) else {}
        prompt_parts.append("-" * 60)
        prompt_parts.append(f"RESUME {i+1}: {resume['filename']}")
        prompt_parts.append(f"  BERT Cosine Score (weighted): {sim.get('weighted_score', 'N/A')}%")
        prompt_parts.append(f"  BERT Cosine Score (raw):      {sim.get('raw_cosine_score', 'N/A')}%")
        if sim.get("section_scores"):
            prompt_parts.append(f"  Section scores: {json.dumps(sim['section_scores'])}")
        prompt_parts.append("-" * 60)
        # Truncate resume to ~2000 words
        words = resume["cleaned_text"].split()
        text = " ".join(words[:2000])
        if len(words) > 2000:
            text += "\n[...truncated...]"
        prompt_parts.append(text)
        prompt_parts.append("")

    prompt_parts.append("=" * 60)
    prompt_parts.append("INSTRUCTIONS")
    prompt_parts.append("=" * 60)
    prompt_parts.append("""
Analyze these resumes against the job description above. You MUST return 
a valid JSON object with this exact structure and nothing else:

{
  "final_ranking": [
    {
      "rank": 1,
      "filename": "<filename>",
      "overall_match_pct": <number 0-100>,
      "recommendation": "STRONG MATCH" | "GOOD MATCH" | "PARTIAL MATCH" | "WEAK MATCH",
      "strengths": ["strength 1", "strength 2", ...],
      "weaknesses": ["weakness 1", "weakness 2", ...],
      "missing_requirements": ["requirement 1", ...],
      "feedback": "<2-3 sentence actionable feedback for this candidate>"
    }
  ],
  "pairwise_comparisons": [
    {
      "higher_ranked": "<filename>",
      "lower_ranked": "<filename>",
      "reasoning": "<specific reasons why higher is ranked above lower, citing evidence from both resumes>"
    }
  ],
  "summary": "<overall analysis paragraph covering the candidate pool quality and hiring recommendation>"
}

IMPORTANT:
- The overall_match_pct is YOUR assessment (not just the BERT score). Use the 
  BERT scores as one input but apply your own judgment about qualifications.
- For pairwise_comparisons, include a comparison for EVERY adjacent pair in your ranking.
- Be specific: cite actual skills, years of experience, project names, or 
  education details from the resumes.
- Return ONLY the JSON object. No markdown fences, no extra text.
""")

    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------

def rank_resumes(
    jd_text: str,
    resumes: list[dict],
    similarity_results: list[dict],
    model: str = "llama3.1",
) -> dict:
    """
    Send the JD + resumes + BERT scores to Llama for reasoned ranking.
    Returns parsed JSON or a fallback dict on error.
    """
    # Sort resumes to match the order in similarity_results
    sim_lookup = {s["filename"]: s for s in similarity_results}
    ordered_sims = [sim_lookup.get(r["filename"], {}) for r in resumes]

    user_prompt = _build_ranking_prompt(jd_text, resumes, ordered_sims)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.3,       # low temp for consistent, precise reasoning
                "num_predict": 4096,       # enough tokens for detailed output
                "top_p": 0.9,
            },
        )

        raw = response["message"]["content"].strip()
        return _parse_response(raw)

    except ollama.ResponseError as e:
        return {"error": f"Ollama error: {e}", "raw_response": ""}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "raw_response": ""}


def _parse_response(raw: str) -> dict:
    """Try to extract valid JSON from the LLM response."""
    # Remove markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "error": "Could not parse LLM response as JSON.",
        "raw_response": raw,
    }


def get_detailed_feedback(
    jd_text: str,
    resume_text: str,
    resume_filename: str,
    model: str = "llama3.1",
) -> str:
    """Get in-depth feedback for a single resume against the JD."""
    prompt = f"""Analyze this resume against the job description and provide detailed, 
actionable feedback.

JOB DESCRIPTION:
{jd_text[:4000]}

RESUME ({resume_filename}):
{resume_text[:4000]}

Provide:
1. An overall match assessment (percentage and category)
2. Top 5 strengths relevant to this role
3. Top 5 gaps or weaknesses
4. Specific suggestions to improve this resume for this role
5. Keywords from the JD that are missing from the resume

Be specific and cite evidence from both documents."""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 2048},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Error generating feedback: {e}"
