import re

class GrammarCorrector:
    def __init__(self):
        self.tool = None
        self.initialized = False
        # Try initializing LanguageTool
        try:
            import language_tool_python
            # Using English by default, but it can download others as needed
            self.tool = language_tool_python.LanguageTool('en-US')
            self.initialized = True
            print("LanguageTool initialized successfully.")
        except Exception as e:
            print(f"LanguageTool failed to initialize (likely due to missing Java). Using Python Fallback Corrector. Error: {e}")

    def correct(self, text):
        """
        Corrects grammar of the input text.
        If LanguageTool is available, uses it. Otherwise, falls back to the Python corrector.
        """
        if not text or not text.strip():
            return "", []

        if self.initialized and self.tool:
            try:
                corrected_text = self.tool.correct(text)
                matches = self.tool.check(text)
                # Formulate a simplified response
                corrections = []
                for m in matches:
                    corrections.append({
                        'message': m.message,
                        'offset': m.offset,
                        'length': m.errorLength,
                        'replacements': m.replacements[:3]
                    })
                return corrected_text, corrections
            except Exception as e:
                print(f"LanguageTool runtime exception: {e}. Falling back...")
                
        # Pure-Python Fallback Corrector
        return self._local_correct(text)

    def _local_correct(self, text):
        """
        A local, rule-based text cleaner and grammar corrector for standard text.
        """
        corrected = text.strip()
        corrections = []

        # Rule 1: Capitalize first letter of every sentence
        def cap_sentence(match):
            return match.group(1) + match.group(2).upper()
        
        # Regex to find start of text or sentence endings followed by space and letter
        temp = re.sub(r'(^|[.!?]\s+)([a-z])', cap_sentence, corrected)
        if temp != corrected:
            corrections.append({
                'message': 'Capitalize sentence starts.',
                'offset': 0,
                'length': len(corrected),
                'replacements': [temp[:30] + "..."]
            })
            corrected = temp

        # Rule 2: Fix spacing around punctuation (e.g. word ,word -> word, word)
        # Spaces before punctuation
        temp = re.sub(r'\s+([.,!?;:])', r'\1', corrected)
        # Spaces after punctuation
        temp = re.sub(r'([.,!?;:])(?=[a-zA-Z])', r'\1 ', temp)
        if temp != corrected:
            corrections.append({
                'message': 'Correct punctuation spacing.',
                'offset': 0,
                'length': len(corrected),
                'replacements': [temp[:30] + "..."]
            })
            corrected = temp

        # Rule 3: Correct double spaces
        temp = re.sub(r' {2,}', ' ', corrected)
        if temp != corrected:
            corrections.append({
                'message': 'Remove duplicate spaces.',
                'offset': 0,
                'length': len(corrected),
                'replacements': [temp[:30] + "..."]
            })
            corrected = temp

        # Rule 4: Common phonetic / spelling mappings in Indian English
        mappings = {
            r'\bi\b': 'I',
            r'\bdont\b': "don't",
            r'\bcant\b': "can't",
            r'\bwont\b': "won't",
            r'\bwhats\b': "what's",
            r'\bthats\b': "that's",
            r'\bplz\b': "please",
            r'\bthx\b': "thank you",
            r'\bhospital\b': "hospital",
            r'\bveg\b': "vegetarian"
        }
        for pat, rep in mappings.items():
            temp, count = re.subn(pat, rep, corrected, flags=re.IGNORECASE)
            if count > 0:
                corrections.append({
                    'message': f"Standardize word '{rep}'.",
                    'offset': 0,
                    'length': len(corrected),
                    'replacements': [rep]
                })
                corrected = temp

        return corrected, corrections

# Singleton Instance
_corrector = None

def get_corrector():
    global _corrector
    if _corrector is None:
        _corrector = GrammarCorrector()
    return _corrector

def correct_grammar(text):
    """
    Module level convenience wrapper.
    """
    corrector = get_corrector()
    return corrector.correct(text)
