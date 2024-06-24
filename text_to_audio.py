from gtts import gTTS
import os

def text_to_speech(text, filename="output.mp3", lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    print(f"Text-to-speech conversion completed. Audio saved as {filename}")

if __name__ == "__main__":
    text_input = "hi my creator, am metaphor. Thanks to let me in this virtual world "
    output_file = input("Enter the output filename (default: output.mp3): ") or "output.mp3"

    text_to_speech(text_input, filename=output_file)
