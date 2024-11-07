import whisper
import difflib
import numpy as np
import string
import ssl

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Proceed with your model download
model = whisper.load_model('base')

def clean_text(text):
    """
    Clean text by removing punctuation and normalizing whitespace.
    
    Parameters:
    - text (str): Input text to clean
    
    Returns:
    - str: Cleaned text
    """
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text.translate(translator)
    
    # Normalize whitespace and convert to lowercase
    return ' '.join(text_no_punctuation.lower().split())

def transcribe_and_compare(audio_path, expected_text):
    """
    Transcribe an audio file using Whisper and compare it with expected text.
    
    Parameters:
    - audio_path (str): Path to the audio file
    - expected_text (str): The expected transcription text
    
    Returns:
    - Dictionary containing transcription results and accuracy metrics
    """
    # Load the Whisper model
    model = whisper.load_model('base')
    
    # Transcribe the audio
    result = model.transcribe(audio_path)
    transcribed_text = result['text'].strip()
    
    # Clean and normalize texts
    normalized_transcribed = clean_text(transcribed_text)
    normalized_expected = clean_text(expected_text)
    
    # Calculate accuracies
    char_accuracy = calculate_character_accuracy(normalized_transcribed, normalized_expected)
    word_accuracy = calculate_word_accuracy(normalized_transcribed, normalized_expected)
    
    # Generate a diff to show differences
    diff = generate_text_diff(normalized_transcribed, normalized_expected)
    
    return {
        'original_transcription': transcribed_text,
        'expected_text': expected_text,
        'cleaned_transcription': normalized_transcribed,
        'cleaned_expected': normalized_expected,
        'character_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'diff': diff,
        'whisper_confidence': result.get('confidence', 'N/A')
    }

def calculate_character_accuracy(transcribed, expected):
    """
    Calculate character-level accuracy between transcribed and expected texts.
    
    Returns accuracy as a percentage.
    """
    matcher = difflib.SequenceMatcher(None, transcribed, expected)
    matching_chars = matcher.ratio() * 100
    return round(matching_chars, 2)

def calculate_word_accuracy(transcribed, expected):
    """
    Calculate word-level accuracy between transcribed and expected texts.
    
    Returns accuracy as a percentage.
    """
    transcribed_words = transcribed.split()
    expected_words = expected.split()
    
    # Use SequenceMatcher to find the longest matching sequence of words
    matcher = difflib.SequenceMatcher(None, transcribed_words, expected_words)
    matching_blocks = matcher.get_matching_blocks()
    
    # Sum the number of matched words in each block
    matching_words = sum(block.size for block in matching_blocks)
    
    # Use the length of the longer text to calculate the accuracy as a percentage
    total_words = max(len(transcribed_words), len(expected_words))
    
    # Prevent division by zero
    if total_words == 0:
        return 0
    
    accuracy = (matching_words / total_words) * 100
    return round(accuracy, 2)


def generate_text_diff(transcribed, expected):
    """
    Generate a human-readable diff between transcribed and expected texts.
    """
    diff = list(difflib.ndiff(transcribed, expected))
    return ''.join(diff)

# Example usage
if __name__ == "__main__":
    audio_file_path = "/Users/elisoaptsiauri/Downloads/video1.mp3"
    expected_transcription = "Hello there, and welcome to your digital portfolio review. Over the next couple of minutes, we'll look at the portfolio you have with us, its change in value over the last 12 months, the fees you have been charged, and our principles for managing your money. First, let's look at the total current value of the portfolio we manage for you. Your money is invested in a medium risk portfolio, which is designed for investors who plan to hold it for at least five years and balances your investment over a number of different asset classes. Equities, which enable you to share in the profits and growth of companies listed on the stock exchange. This is why they are sometimes referred to simply as shares. Bonds, which are in effect loans to companies and governments, who pay interest on the bonds. These are also known as 'fixed interest' investments as the interest is usually fixed at the time the bond is issued. Commercial property, for example investing in offices, warehouses and shopping centres, where your investment benefits from the rent paid and any increase in the property's value.The portfolio may also include varying levels of cash from time to time. This will depend on how we anticipate markets to perform in the short term and the rates of interest available."
    
    # Run the comparison
    result = transcribe_and_compare(audio_file_path, expected_transcription)
    
    # Print results
    print("\nTranscription Results:")
    #print(f"\nOriginal Transcription: {result['original_transcription']}")
    print(f"\nCleaned Transcription: {result['cleaned_transcription']}")
    print("\n")
    print(f"Expected Text: {result['cleaned_expected']}")
    print(f"\nCharacter Accuracy: {result['character_accuracy']}%")
    print(f"\nWord Accuracy: {result['word_accuracy']}%")
    # print("\nDiff Visualization:")
    # print(result['diff'])