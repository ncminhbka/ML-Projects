from elevenlabs.client import ElevenLabs
from elevenlabs.play import play # Để phát âm thanh ngay lập tức
from elevenlabs import save # Để lưu file

# Thay thế "YOUR_API_KEY" bằng API Key của bạn
# Hoặc thiết lập biến môi trường ELEVENLABS_API_KEY
client = ElevenLabs(api_key="YOUR_API_KEY")

# Gọi API chuyển văn bản thành giọng nói
audio = client.text_to_speech.convert(
    text="Xin chào! Đây là một bài kiểm tra API ElevenLabs cơ bản.",
    voice_id="JBFqnCBsd6RMkjVDRZzb", # Một voice ID mẫu, bạn có thể thay đổi
    model_id="eleven_multilingual_v2", # Model hỗ trợ đa ngôn ngữ
    output_format="mp3_44100_128",
)

# Phát âm thanh (cần cài đặt MPV hoặc FFmpeg)
# play(audio) 

# Lưu file âm thanh
save(audio, "output_elevenlabs.mp3")

print("Đã tạo và lưu file âm thanh thành công: output_elevenlabs.mp3")