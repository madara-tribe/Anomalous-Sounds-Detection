from IPython.display import Audio
import numpy as np
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import os

def mix_audio(original_audio, augmented_audio, weight_original=0.5, weight_augmented=0.5):
    return original_audio.overlay(augmented_audio, gain_during_overlay=(weight_augmented - weight_original))


def add_gaussian_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return np.clip(audio + noise, -1.0, 1.0)
    
def add_white_noise(audio, noise_level=0.005):
    noise = np.random.uniform(-noise_level, noise_level, audio.shape)
    return np.clip(audio + noise, -1.0, 1.0)

def pitch_shift(audio_segment, semitones=2):
    # Change the frame rate to achieve pitch shift
    shifted_audio = audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * (2.0 ** (semitones / 12.0)))
    })
    # Set back to the original frame rate
    return shifted_audio.set_frame_rate(audio_segment.frame_rate)
    
def add_brownian_noise(audio_segment, noise_level=0.005):
    samples = np.array(audio_segment.get_array_of_samples())
    brownian_noise = np.cumsum(np.random.normal(0, noise_level, len(samples)))
    noisy_samples = np.clip(samples + brownian_noise, -32768, 32767).astype(np.int16)
    noisy_audio = audio_segment._spawn(noisy_samples.tobytes())
    return noisy_audio
    
import random
from gtts import gTTS
def create_voice_sound(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("voice/voice_sound.mp3")
    voice_audio = AudioSegment.from_mp3("voice/voice_sound.mp3")
    return voice_audio
    
def change_speed(audio_segment, speed=1.0):
    altered_audio = audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * speed)
    })
    return altered_audio.set_frame_rate(audio_segment.frame_rate)
    
def mix_mp3_with_wav(mp3_file, wav_audio_segment, weight_original=0.5, weight_mp3=0.5):
    mp3_audio = AudioSegment.from_mp3(mp3_file)
    mixed_audio = mix_audio(wav_audio_segment, mp3_audio, weight_original=weight_original, weight_augmented=weight_mp3)
    return mixed_audio
    
def add_environmental_noise(audio_segment, noise_file, weight_original=0.7, weight_noise=0.3, mp3=True):
    if mp3:
        noise_audio = AudioSegment.from_mp3(noise_file)
    else:
        noise_audio = AudioSegment.from_file(noise_file)
    noise_audio = noise_audio - 20  # Lower the volume of the noise to make it background noise
    mixed_audio = mix_audio(audio_segment, noise_audio, weight_original=weight_original, weight_augmented=weight_noise)
    return mixed_audio
    
    
# Main Execution
if __name__ == "__main__":
    INPUT_FILE = "fake_sounds/normal_10_seconds.wav"
    OUTPUT_DIR = "augmented_audio"
    NOISE_LEVELS = 2

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the input audio file
    audio_data, sample_rate = load_audio(INPUT_FILE)

        
    # Apply Gaussian noise
    gaussian_noisy_audio = add_gaussian_noise(audio_data, noise_level=0.01 * NOISE_LEVELS)
    gaussian_output_file = os.path.join(OUTPUT_DIR, f"gaussian_noise_level_{noise_level}.wav")
    save_audio(gaussian_output_file, gaussian_noisy_audio, sample_rate)

    # Apply white noise
    white_noisy_audio = add_white_noise(audio_data, noise_level=0.01 * NOISE_LEVELS)
    white_output_file = os.path.join(OUTPUT_DIR, f"white_noise_level_{noise_level}.wav")
    save_audio(white_output_file, white_noisy_audio, sample_rate)

    """ pitch shift """
    audio_segment = AudioSegment.from_wav(INPUT_FILE)
    shifted_audio = pitch_shift(audio_segment, semitones=NOISE_LEVELS)
    output_file = os.path.join(OUTPUT_DIR, f"pitchshift_tonelevel{tone_level}.wav")
    shifted_audio.export(output_file, format="wav")


    """ brown noise """
    # Add Brownian noise to the original audio
    noisy_audio = add_brownian_noise(audio_segment, noise_level=NOISE_LEVELS)
    output_file = os.path.join(OUTPUT_DIR, f"brownian_noise_noiselevel{noise_level}.wav")
    noisy_audio.export(output_file, format="wav")

    """ mix pitch_shift"""
    mixed_audio = mix_audio(audio_segment, shifted_audio, weight_original=0.6, weight_augmented=0.4)
    # Export the mixed audio to a new file
    mixed_output_file = os.path.join(OUTPUT_DIR, "mix_pitch_shift.wav")
    mixed_audio.export(mixed_output_file, format="wav")

    """ mix brown noise"""

    mixed_audio = mix_audio(audio_segment, noisy_audio, weight_original=0.6, weight_augmented=0.4)
    # Export the mixed audio to a new file
    mixed_output_file = os.path.join(OUTPUT_DIR, "mix_brownian_noise.wav")
    mixed_audio.export(mixed_output_file, format="wav")


    """voice mix """
    audio_segment = AudioSegment.from_wav(INPUT_FILE)
    voice_text = "how dare you admire the person in the midst of the battle"
    tts = gTTS(text=voice_text, lang='en', slow=False)
    tts.save("/tmp/voice_sound.mp3")
    voice_audio = AudioSegment.from_mp3("/tmp/voice_sound.mp3")
    mixed_voice_audio = mix_audio(audio_segment, voice_audio, weight_original=0.5, weight_augmented=0.5)
    mixed_voice_output_file  = os.path.join(OUTPUT_DIR,"mixed_voice_audio1.wav")
    mixed_voice_audio.export(mixed_voice_output_file, format="wav")


    """change spped of normal sound """
    speed = 0.5
    speed_changed_audio = change_speed(audio_segment, speed=speed)
    output_file = os.path.join(OUTPUT_DIR, f"speed_changed_speed{speed}.wav")
    speed_changed_audio.export(output_file, format="wav")

    """ Mix other noise """
    audio_segment = AudioSegment.from_wav(INPUT_FILE)
    file = "ohter_elemenatl_sound"
    # Replace with your environmental noise file path
    mp3_file = os.path.join("environmental", file + ".mp3")

    mixed_mp3_wav_audio = mix_mp3_with_wav(mp3_file, audio_segment, weight_original=0.7, weight_mp3=0.3)
    mixed_mp3_wav_output_file = os.path.join(OUTPUT_DIR, "mixed_mp3_wav_audio" + file + ".wav")
    mixed_mp3_wav_audio.export(mixed_mp3_wav_output_file, format="wav")



