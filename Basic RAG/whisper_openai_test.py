import whisper

model = whisper.load_model("small")   # hoặc "base", "medium", "large"
result = model.transcribe("videoplayback.m4a", language="vi")
print(result["text"])

with open ('text2.txt', 'w', encoding = "utf-8") as file:  
    file.write(result['text'])