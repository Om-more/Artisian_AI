from rag_file import retrieve_rag_context, match_events
from llm import ask_qwen
from llm import build_prompt
from blip import describe_image

def detect_intent(user_query):
    naming_keywords = ["name", "naming", "call it", "what should i name"]
    for k in naming_keywords:
        if k in user_query.lower():
            return "creative_naming"
    return "knowledge"

def infer_category(image_description):
    text = image_description.lower()

    # --- Pottery / Ceramics FIRST ---
    if any(x in text for x in [
        "vase", "pot", "pottery", "ceramic", "clay",
        "earthen", "amphora", "urn", "jar"
    ]):
        return "Pottery"

    # --- Paintings ---
    if any(x in text for x in [
        "painting", "canvas", "artwork", "painted"
    ]):
        return "Paintings"


    # --- Sculptures ---
    if any(x in text for x in [
        "sculpture", "statue", "figurine"
    ]):
        return "Sculptures"


    return "Paintings"  


s = ["bye","exit", "close", "goodbye"]
conversation = []
history = ""

user_query = input("Ask your question (or press Enter to skip): ").strip()
while (user_query != "" and user_query.lower() not in s):
    image_path = input("Enter image path (or press Enter to skip): ").strip()

    user_city = input("Enter your city: ").strip()
    user_state = input("Enter your state: ").strip()

    image_description = ""
    category = ""

    if image_path:
        print("\n Analyzing image...")
        image_description = describe_image(image_path)
        print("Image description:", image_description)

        category = infer_category(image_description)
        print("Inferred category:", category)

    # ---- FALLBACK CATEGORY ----
    if not category:
        category = input("Enter art category manually: ").strip()


    # ---- RAG ----
    rag_context = retrieve_rag_context(
        user_query + " " + image_description
    )

    events = match_events(user_city, user_state, category)
    intent = detect_intent(user_query)
    product_context = f"""
    Product type: {category}
    Material: Handcrafted, traditional materials
    Usage: Typical {category.lower()} use for display or décor
    """
    
        
    # ---- PROMPT ----
    final_prompt = build_prompt(
        user_query,
        rag_context,
        events,
        intent,
        product_context,
        history
    )


    # ---- LLM ----
    print("\n Generating response...\n")
    answer = ask_qwen(final_prompt)
    print("FINAL ANSWER:\n")
    print(answer)
     
    conversation.append({
        "user": user_query,
        "assistant": answer
    })
    
    for t in conversation:
        history += f"User: {t['user']}, Assistant: {t['assistant']} \n"
    user_query = input("Ask your question (or press Enter to skip): ").strip()
