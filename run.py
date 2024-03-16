import logging
import os
import torch
from TTS.api import TTS
from flask import Flask, request, jsonify

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.DEBUG)  # 设置日志级别为DEBUG，记录所有级别的日志消息

# 获取设备
device = "cuda" if torch.cuda.is_available() else "gpu"
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST")

# 设置音频文件保存路径
audio_directory = "audio"

@app.route('/synthesize', methods=['GET'])
def synthesize():
    logging.info("Audio generation request received.")
    text = request.args.get('text', '')
    if text:
        logging.info("Audio generation started.")
        # 生成音频文件的文件名
        file_name = "test10.wav"
        file_path = os.path.join(audio_directory, file_name)  # 构建完整的文件路径
        tts.tts_to_file(text=text, file_path=file_path)
        # 记录日志消息
        logging.info("Audio generated successfully.")
        # 返回带有CORS头信息的响应，包括音频文件的URL
        audio_url = request.host_url + file_path.replace("\\", "/")  # 构建音频文件的URL，并替换反斜杠为正斜杠
        response = jsonify({"message": "Audio generated successfully.", "audioUrl": audio_url})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
        return response, 200
    else:
        # 记录日志消息
        logging.error("No text provided.")
        response = jsonify({"error": "No text provided."})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
        return response, 400

if __name__ == '__main__':
    # 确保音频文件保存目录存在
    if not os.path.exists(audio_directory):
        os.makedirs(audio_directory)
    app.run(host='0.0.0.0', port=8081)
