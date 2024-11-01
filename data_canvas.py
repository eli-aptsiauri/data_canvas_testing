import whisper
import difflib
import numpy as np

import ssl
import urllib.request

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Proceed with your model download
model = whisper.load_model('base')


def transcribe_and_compare(audio_path, expected_text):
    """
    Transcribe an audio file using Whisper and compare it with expected text.
    
    Parameters:
    - audio_path (str): Path to the audio file
    - expected_text (str): The expected transcription text
    
    Returns:
    - Dictionary containing transcription results and accuracy metrics
    """
    # Load the Whisper model (you can change model size: 'tiny', 'base', 'small', 'medium', 'large')
    model = whisper.load_model('base')
    
    # Transcribe the audio
    result = model.transcribe(audio_path)
    transcribed_text = result['text'].strip()
    
    # Normalize texts (lowercase, remove extra whitespace)
    normalized_transcribed = ' '.join(transcribed_text.lower().split())
    normalized_expected = ' '.join(expected_text.lower().split())
    
    # Calculate character-level and word-level accuracy
    char_accuracy = calculate_character_accuracy(normalized_transcribed, normalized_expected)
    word_accuracy = calculate_word_accuracy(normalized_transcribed, normalized_expected)
    
    # Generate a diff to show differences
    diff = generate_text_diff(normalized_transcribed, normalized_expected)
    
    return {
        'original_transcription': transcribed_text,
        'expected_text': expected_text,
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
    # Use Levenshtein distance-based approach
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
    
    # Count matching words
    matching_words = sum(1 for w1, w2 in zip(transcribed_words, expected_words) if w1 == w2)
    
    # Use the length of the shorter text for calculation
    total_words = min(len(transcribed_words), len(expected_words))
    
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
    # Example paths and texts - replace with your actual audio file and expected text
    audio_file_path = "/path/to/your/source_audio.wav"
    expected_transcription = "Your expected transcription here"
    
    # Run the comparison
    result = transcribe_and_compare(audio_file_path, expected_transcription)
    
    # Print results
    print("Transcription Results:")
    print(f"Original Transcription: {result['original_transcription']}")
    print(f"Expected Text: {result['expected_text']}")
    print(f"Character Accuracy: {result['character_accuracy']}%")
    print(f"Word Accuracy: {result['word_accuracy']}%")
    print("\nDiff Visualization:")
    print(result['diff'])