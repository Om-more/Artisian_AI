import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

def ask_qwen(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.80
        }
    }

    r = requests.post(OLLAMA_URL, json=payload)
    return r.json()["response"]

def build_prompt(user_query, rag_context, events, intent, product_context, history):
    
    if intent == "creative_naming":
        return f"""
You are a creative branding assistant for Indian artisans.

Conversation so far:
{history}

PRODUCT CONTEXT (must strictly follow):
{product_context}

NAMING RULES:
- You MAY invent names creatively.
- Names MUST clearly relate to the product type, material, form, or usage.
- Avoid unrelated mythological, religious, or ritual references.
- Names should be simple, premium, and suitable for online marketplaces.
- Do NOT introduce facts not implied by the product context.

USER QUERY:
{user_query}

TASK:
1. Suggest 5 suitable product names.
2. For each name, give a 1-line marketplace-ready description.
"""

    event_text = ""
    for e in events:
        event_text += f"""
Event Name: {e['event_name']}
Description: {e['description']}
Official Link: {e['official_link']}
"""

    return f"""
You are an assistant helping Indian artisans.

Conversation so far:
{history}

PRODUCT CONTEXT:
{product_context}

RULES (must follow strictly):
- Use ONLY the provided knowledge context and event information.
- Do NOT invent facts, dates, prices, materials, or claims.
- If information is missing, say so clearly.
- Use general reasoning ONLY to explain or connect the provided information.

USER QUESTION:
{user_query}

KNOWLEDGE CONTEXT:
{rag_context}

RELEVANT EVENTS:
{event_text if event_text else "No relevant events found."}

TASK:
Give a clear, practical, and grounded answer.
Mention events only if they are directly relevant, and redirect to official links for details.
"""
    return f"""

Conversation so far:
{history}

PRODUCT CONTEXT:
{product_context}

USER QUESTION:
{user_query}

    SUGGEST NETWORK AND BUSINESS EXPANSION IDEAS FOR LOCAL CRAFTSMEN IN INDIA IF SPECIFICALLY ASKED ABOUT EXPANSION BASED QUESTIONS
"""
    return f"""
Conversation so far:
{history}

PRODUCT CONTEXT:
{product_context}

USER QUESTION:
{user_query}

CRITICAL RULES:
- The user does NOT know their production cost.
- You MUST NOT invent prices, costs, currencies, or numerical examples.
- You MUST NOT give sample calculations or ranges.
- You MUST NOT mention specific market prices.

    ALLOWED:
- You MAY explain pricing and negotiation strategies in qualitative terms.
- You MAY refer to common practices among Indian craftsmen and craft markets,
  but only as generalized patterns (e.g., "many artisans", "often observed").
- You MAY provide step-by-step guidance that helps the artisan estimate cost
  without using numbers.

YOUR TASK:
1. Clearly explain why fixing a selling price without knowing cost is risky
   for artisans, especially in negotiation-heavy Indian markets.
2. List the minimum components an Indian artisan typically considers
   when estimating cost (materials, time, effort, transport, wastage).
3. Suggest practical ways Indian craftsmen often approximate cost
   when exact records are unavailable (e.g., daily work comparison,
   batch-based thinking, relative effort).
4. Offer qualitative pricing strategies (such as starting higher to allow
   negotiation or separating décor and utility positioning) WITHOUT numbers.

IMPORTANT STYLE GUIDELINES:
- Use calm, respectful, and supportive language.
- Avoid absolutes or guarantees.
- Do not sound like a textbook or a financial advisor.
- Speak in a way that feels familiar to Indian craftsmen and local markets.
"""

    return f"""
      If ANYTHING feels APART or NOT RELATED TO {rag_context} from this query then politely deny about information and say information not available 
"""
   
    
    return prompt
