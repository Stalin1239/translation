import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_aws_client

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    aws_client = get_aws_client('translate')
    print("AWS Client instantiated:", aws_client is not None)
    if aws_client:
        try:
            response = aws_client.translate_text(
                Text="Keep right! Pass the obstacle on the right side.",
                SourceLanguageCode="en",
                TargetLanguageCode="kn"
            )
            print("Translation successful!")
            print("Translated Text:", response.get('TranslatedText'))
        except Exception as e:
            print("AWS Translate failed with error:", e)
