from deep_translator import GoogleTranslator

langs_list = GoogleTranslator().get_supported_languages()
langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)

def translate_text(src, dest_lang):
    obj = GoogleTranslator(source='auto', target=dest_lang)
    words = list(src.split(" "))
    result = []
    index = 0
    while index <= len(words):
        text_needed = " ".join(words[index:index+100])
        result.append(obj.translate(text_needed))
        index += 100
    return " ".join(result)

# print(langs_dict)
