import whisper
import difflib
import numpy as np
import string
import ssl
import urllib.request
from moviepy.editor import VideoFileClip
import os
import re

# Setup Whisper model
model = whisper.load_model('base')

def clean_text(text):
    # Remove punctuation (including special characters like smart quotes)
    text = re.sub(r'[^\w\s]', '', text)  # This removes all characters except words and whitespace
    
    # Normalize whitespace and convert to lowercase
    return ' '.join(text.lower().split())

def download_video(url):
    local_path = "/tmp/temp_video.mp4"
    try:
        with urllib.request.urlopen(url) as response:
            with open(local_path, 'wb') as out_file:
                out_file.write(response.read())
    except ssl.SSLError as e:
        print(f"SSL error occurred: {e}")
        # Optionally handle or re-raise the error
        raise

    return local_path

def extract_audio(video_path):
    with VideoFileClip(video_path) as video:
        audio_path = "/tmp/temp_audio.mp3"
        video.audio.write_audiofile(audio_path)
    return audio_path

def load_expected_text(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def transcribe_and_compare(url, expected_text_file):
    video_path = download_video(url)
    audio_path = extract_audio(video_path)

    # Load expected text from file
    expected_text = load_expected_text(expected_text_file)

    # Transcribe the audio
    result = model.transcribe(audio_path)
    transcribed_text = result['text'].strip()

    # Clean texts
    normalized_transcribed = clean_text(transcribed_text)
    normalized_expected = clean_text(expected_text)

    # Calculate accuracy
    char_accuracy = calculate_segmented_character_accuracy(normalized_transcribed, normalized_expected)
    word_accuracy = calculate_word_accuracy(normalized_transcribed, normalized_expected)
    diff = generate_text_diff(normalized_transcribed, normalized_expected)

    # Cleanup
    os.remove(video_path)
    os.remove(audio_path)

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

def calculate_segmented_character_accuracy(transcribed, expected, segment_size=200):
    """
    Calculate character-level accuracy between transcribed and expected texts
    by dividing them into segments.
    
    Parameters:
    - transcribed (str): The transcribed text.
    - expected (str): The expected (reference) text.
    - segment_size (int): The size of each segment in characters.
    
    Returns:
    - float: Average character accuracy as a percentage.
    """
    # Split transcribed and expected texts into segments
    transcribed_segments = [transcribed[i:i + segment_size] for i in range(0, len(transcribed), segment_size)]
    expected_segments = [expected[i:i + segment_size] for i in range(0, len(expected), segment_size)]

    # Ensure both lists are the same length by padding the shorter one with empty strings
    max_segments = max(len(transcribed_segments), len(expected_segments))
    transcribed_segments += [''] * (max_segments - len(transcribed_segments))
    expected_segments += [''] * (max_segments - len(expected_segments))

    # Calculate character accuracy for each segment
    segment_accuracies = []
    for transcribed_segment, expected_segment in zip(transcribed_segments, expected_segments):
        matcher = difflib.SequenceMatcher(None, transcribed_segment, expected_segment)
        segment_accuracy = matcher.ratio() * 100
        segment_accuracies.append(segment_accuracy)

    # Calculate average character accuracy across all segments
    average_accuracy = sum(segment_accuracies) / len(segment_accuracies)
    return round(average_accuracy, 2)


def calculate_word_accuracy(transcribed, expected):
    transcribed_words = transcribed.split()
    expected_words = expected.split()
    matcher = difflib.SequenceMatcher(None, transcribed_words, expected_words)
    matching_words = sum(block.size for block in matcher.get_matching_blocks())
    total_words = max(len(transcribed_words), len(expected_words))
    return round((matching_words / total_words) * 100, 2) if total_words else 0

def generate_text_diff(transcribed, expected):
    return ''.join(difflib.ndiff(transcribed, expected))

# Usage example
if __name__ == "__main__":
    url = "https://staging-cdn.videocanvas.co.uk/vc/clefipwu20001qfpikh928ax3/assets/clefipwuf0003qfpi6a4fzwsn-1.mp4"
    expected_text_file = "expected_text.txt"  # Path to the file containing expected transcription
    result = transcribe_and_compare(url, expected_text_file)
    #print(result)
    print("\nTranscription Results:")
    print(f"\nCleaned Transcription: {result['cleaned_transcription']}")
    print("\n")
    print(f"Expected Text: {result['cleaned_expected']}")
    print(f"\nCharacter Accuracy: {result['character_accuracy']}%")
    print(f"\nWord Accuracy: {result['word_accuracy']}%")
