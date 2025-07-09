from telegram.ext import Updater, MessageHandler, Filters
import librosa
import numpy as np
import tensorflow as tf
import os
import subprocess

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Label map
label_map = {
    0: "Belly Pain",
    1: "Burping",
    2: "Discomfort",
    3: "Hungry",
    4: "Tired"
}

# Fungsi prediksi
def predict_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = model.predict(mfcc_mean)
    return label_map[np.argmax(prediction)]

# Handler audio
def handle_audio(update, context):
    file = update.message.audio or update.message.voice or update.message.document
    if not file:
        update.message.reply_text("Tolong kirim file audio (.mp3 atau .wav).")
        return

    downloaded_file = file.get_file().download(custom_path='input_audio')

    # Konversi ke .wav jika bukan
    if not downloaded_file.endswith('.wav'):
        wav_file = 'converted.wav'
        subprocess.call(['ffmpeg', '-y', '-i', downloaded_file, wav_file])
    else:
        wav_file = downloaded_file

    try:
        result = predict_audio(wav_file)
        update.message.reply_text(f"Prediksi tangisan bayi: *{result}*", parse_mode='Markdown')
    except Exception as e:
        update.message.reply_text("Gagal memproses audio.")
        print("ERROR:", e)

    os.remove(downloaded_file)
    if os.path.exists('converted.wav'):
        os.remove('converted.wav')

# Bot start
def main():
    TOKEN = os.getenv("7731244099:AAGqBiY0YFd_lYsXMOQqbOnCd64JiTo0JxQ")  # Ambil token dari variabel environment
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.audio | Filters.voice | Filters.document.audio, handle_audio))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
