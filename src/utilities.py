def clean_output_text(text:str):
    return "\n".join([item.strip() for item in text.split("\n")]).strip()