from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, AudioMessage
)
import os
import uuid
from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader

load_dotenv('.env')

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

youtube = Youtube(step=4)
website = Website()
memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'), memory_message_count=2)

# IMPORTANT:
# The OpenAI API key is server-side only. Never accept it from LINE users
# and never store it in MongoDB / db.json.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError('OPENAI_API_KEY is not configured on the server.')

openai_model = OpenAIModel(api_key=OPENAI_API_KEY)


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    logger.info(f'{user_id}: {text}')

    try:
        # /註冊 is intentionally removed. The API key is configured on the server.
        if text.startswith('/註冊'):
            msg = TextSendMessage(
                text='此 Bot 不需要註冊 OpenAI API Token。API Key 已由伺服器安全設定。'
            )

        elif text.startswith('/指令說明'):
            msg = TextSendMessage(
                text="指令：\n"
                     "/系統訊息 + Prompt\n"
                     "👉 設定希望 ChatGPT 扮演的角色，例如：請你扮演擅長做總結的人\n\n"
                     "/清除\n"
                     "👉 清除當前對話歷史\n\n"
                     "/圖像 + Prompt\n"
                     "👉 使用圖像模型生成圖像\n\n"
                     "語音輸入\n"
                     "👉 將語音轉成文字，再由 ChatGPT 回覆\n\n"
                     "其他文字輸入\n"
                     "👉 直接與 AI 助理對話"
            )

        elif text.startswith('/系統訊息'):
            memory.change_system_message(user_id, text[5:].strip())
            msg = TextSendMessage(text='輸入成功')

        elif text.startswith('/清除'):
            memory.remove(user_id)
            msg = TextSendMessage(text='歷史訊息清除成功')

        elif text.startswith('/圖像'):
            prompt = text[3:].strip()
            if not prompt:
                msg = TextSendMessage(text='請在 /圖像 後面輸入描述，例如：/圖像 一隻坐在窗邊的橘貓')
            else:
                memory.append(user_id, 'user', prompt)
                is_successful, response, error_message = openai_model.image_generations(prompt)
                if not is_successful:
                    raise Exception(error_message)

                url = response['data'][0]['url']
                msg = ImageSendMessage(
                    original_content_url=url,
                    preview_image_url=url
                )
                memory.append(user_id, 'assistant', url)

        else:
            memory.append(user_id, 'user', text)
            url = website.get_url_from_text(text)

            if url:
                if youtube.retrieve_video_id(text):
                    is_successful, chunks, error_message = youtube.get_transcript_chunks(
                        youtube.retrieve_video_id(text)
                    )
                    if not is_successful:
                        raise Exception(error_message)

                    youtube_transcript_reader = YoutubeTranscriptReader(
                        openai_model, os.getenv('OPENAI_MODEL_ENGINE')
                    )
                    is_successful, response, error_message = youtube_transcript_reader.summarize(chunks)
                    if not is_successful:
                        raise Exception(error_message)

                    role, response = get_role_and_content(response)
                    msg = TextSendMessage(text=response)

                else:
                    chunks = website.get_content_from_url(url)
                    if len(chunks) == 0:
                        raise Exception('無法撈取此網站文字')

                    website_reader = WebsiteReader(
                        openai_model, os.getenv('OPENAI_MODEL_ENGINE')
                    )
                    is_successful, response, error_message = website_reader.summarize(chunks)
                    if not is_successful:
                        raise Exception(error_message)

                    role, response = get_role_and_content(response)
                    msg = TextSendMessage(text=response)

            else:
                is_successful, response, error_message = openai_model.chat_completions(
                    memory.get(user_id),
                    os.getenv('OPENAI_MODEL_ENGINE')
                )
                if not is_successful:
                    raise Exception(error_message)

                role, response = get_role_and_content(response)
                msg = TextSendMessage(text=response)

            memory.append(user_id, role, response)

    except Exception as e:
        memory.remove(user_id)

        error_text = str(e)
        if error_text.startswith('Incorrect API key provided'):
            msg = TextSendMessage(text='伺服器上的 OpenAI API Key 無效，請由管理員檢查伺服器設定。')
        elif error_text.startswith('That model is currently overloaded with other requests.'):
            msg = TextSendMessage(text='AI 服務目前負載較高，請稍後再試。')
        else:
            msg = TextSendMessage(text=error_text)

    line_bot_api.reply_message(event.reply_token, msg)


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    user_id = event.source.user_id
    audio_content = line_bot_api.get_message_content(event.message.id)
    input_audio_path = f'{str(uuid.uuid4())}.m4a'

    with open(input_audio_path, 'wb') as fd:
        for chunk in audio_content.iter_content():
            fd.write(chunk)

    try:
        is_successful, response, error_message = openai_model.audio_transcriptions(
            input_audio_path,
            'whisper-1'
        )
        if not is_successful:
            raise Exception(error_message)

        memory.append(user_id, 'user', response['text'])

        is_successful, response, error_message = openai_model.chat_completions(
            memory.get(user_id),
            os.getenv('OPENAI_MODEL_ENGINE')
        )
        if not is_successful:
            raise Exception(error_message)

        role, response = get_role_and_content(response)
        memory.append(user_id, role, response)
        msg = TextSendMessage(text=response)

    except Exception as e:
        memory.remove(user_id)

        error_text = str(e)
        if error_text.startswith('Incorrect API key provided'):
            msg = TextSendMessage(text='伺服器上的 OpenAI API Key 無效，請由管理員檢查伺服器設定。')
        else:
            msg = TextSendMessage(text=error_text)

    finally:
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)

    line_bot_api.reply_message(event.reply_token, msg)


@app.route("/", methods=['GET'])
def home():
    return 'Hello World'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
