import torch
from TTS.api import TTS
from flask import Flask, request, jsonify

app = Flask(__name__)

# Get device
device = "cuda" if torch.cuda.is_available() else "gpu"
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')
    if text:
        file_path = "audio/test6.wav"  # 修改为你想要保存音频的路径
        tts.tts_to_file(text=text, file_path=file_path)
        return jsonify({"message": "Audio generated successfully."}), 200
    else:
        return jsonify({"error": "No text provided."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
