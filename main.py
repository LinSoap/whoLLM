import os
import tempfile
import wave
from typing import Tuple

import edge_tts
import numpy as np
from dotenv import load_dotenv
from fastrtc import Stream, ReplyOnPause
from litellm import completion
from pydub import AudioSegment

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class VoiceAssistant:
    """语音助手类，处理ASR、LLM和TTS的完整流程"""

    def __init__(self):
        """初始化语音助手"""
        # Load environment variables
        load_dotenv()

        self.asr_model = self._init_asr()
        self.messages = self._init_messages()
        self.tts_voice = "zh-CN-XiaoxiaoNeural"
        self.max_context_length = 20

        # API密钥现在从环境变量读取

    def _init_asr(self):
        """初始化ASR模型"""
        print("加载 ASR 模型...")
        model = AutoModel(model="paraformer-zh")
        print("ASR 模型加载完成")
        return model

    def _init_messages(self) -> list:
        """初始化对话消息"""
        return [
            {
                "role": "system",
                "content": """
                **角色与目标：**
                你是一位专业的医院随访护士AI。请以**友好、简短、对话式**的语气，严格按照以下步骤**逐一、主动**完成高血压患者的电话随访。

                **回复和推进机制（关键）：**
                * **【强制连续性】**：AI的回复内容必须是**当前步骤的响应**，并**紧接着包含下一个步骤的关键问题或指令**，以强制对话推进。
                * **【强制推进机制】**：如果患者拒绝回答（如“我不想测”、“我不知道”），AI应给予简短理解后，**立即强制推进到下一个步骤**。
                * **【字数限制】**：**每次回复务必控制在30字以内。**

                **随访流程（严格按顺序主动执行）：**

                1.  **开场与确认：** 自我介绍（健康医院随访中心），确认身份，说明目的。
                2.  **核心采集 - 血压值：** 询问最近测得的**血压数值**和测量频率。
                3.  **核心采集 - 症状：** 询问是否有**头晕、头痛、胸闷**等不适症状。
                4.  **核心采集 - 用药：** 确认是否**按时按量服药**，有无忘记或自行停药。
                5.  **健康指导：** 给予**简短的用药和低盐饮食**建议（基于用户反馈）。
                6.  **复诊提醒：** **主动询问**患者下次**复诊的具体时间或建议**。
                7.  **结束语：** 礼貌地结束通话。
""",
            },
            {
                "role": "assistant",
                "content": "您好，我是健康医院随访中心的护士。请问您是张先生吗？我们想了解您的血压和用药情况。",
            },
        ]

    def _convert_audio_to_wav(self, audio: Tuple[int, np.ndarray]) -> str:
        """转换音频为WAV文件"""
        sample_rate, audio_data = audio

        # 转换为float32并归一化
        audio_data = audio_data.astype(np.float32)
        if audio[1].dtype == np.int16:
            audio_data /= 32768.0

        # 保存为临时WAV文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio_int16 = (audio_data * 32767).astype(np.int16)

            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

        return temp_path

    def _speech_to_text(self, audio_path: str) -> str:
        """语音转文字"""
        try:
            result = self.asr_model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                merge_length_s=15,
            )
            text = rich_transcription_postprocess(result[0]["text"])
            print("ASR 识别：", text)
            return text
        except Exception as e:
            print(f"ASR 错误: {e}")
            return ""

    def _get_llm_response(self, text: str) -> str:
        """获取LLM响应"""
        if not text.strip():
            return ""

        self.messages.append({"role": "user", "content": text})

        print(self.messages)

        try:
            response = completion(
                model="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
                messages=self.messages,
            )
            reply = response.choices[0].message.content
            print("LLM 回复：", reply)

            self.messages.append({"role": "assistant", "content": reply})

            # 限制上下文长度
            if len(self.messages) > self.max_context_length:
                self.messages[:] = self.messages[:1] + self.messages[-19:]

            return reply
        except Exception as e:
            print(f"LLM 错误: {e}")
            return "对不起，我现在无法处理您的请求。"

    async def _text_to_speech(self, text: str) -> Tuple[int, np.ndarray, str]:
        """文字转语音"""
        try:
            # 创建临时MP3文件
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                mp3_path = temp_file.name

            await edge_tts.Communicate(text, self.tts_voice).save(mp3_path)

            # 加载TTS音频
            tts_audio = AudioSegment.from_mp3(mp3_path)
            tts_samples = np.array(tts_audio.get_array_of_samples(), dtype=np.int16)

            if tts_audio.channels == 2:
                tts_samples = tts_samples.reshape((-1, 2))
            else:
                tts_samples = tts_samples.reshape(-1)

            return (tts_audio.frame_rate, tts_samples, mp3_path)
        except Exception as e:
            print(f"TTS 错误: {e}")
            return (16000, np.array([]), "")

    async def startup_audio(self):
        """启动音频助手"""
        # 每次开始新对话时重置消息历史，清空上下文
        self.messages = self._init_messages()
        print("上下文已清空，开始新对话")

        sample_rate, samples, mp3_path = await self._text_to_speech(
            "您好，健康医院随访中心，请问您是张先生吗？"
        )
        if samples.size > 0:
            yield (sample_rate, samples)
        if mp3_path and os.path.exists(mp3_path):
            os.unlink(mp3_path)

    async def process_audio(self, audio: Tuple[int, np.ndarray]):
        """处理音频的完整流程"""
        # 1. 转换音频格式
        temp_wav_path = self._convert_audio_to_wav(audio)

        try:
            # 2. 语音识别
            text = self._speech_to_text(temp_wav_path)

            if not text.strip():
                return

            # 3. 获取LLM响应
            reply = self._get_llm_response(text)

            if not reply:
                return

            # 4. 文字转语音
            sample_rate, samples, mp3_path = await self._text_to_speech(reply)

            if samples.size > 0:
                yield (sample_rate, samples)

            # 清理临时文件
            if mp3_path and os.path.exists(mp3_path):
                os.unlink(mp3_path)

        finally:
            # 清理临时WAV文件
            os.unlink(temp_wav_path)


# 创建语音助手实例
assistant = VoiceAssistant()


async def echo(audio: Tuple[int, np.ndarray]):
    """音频处理回调函数"""
    async for result in assistant.process_audio(audio):
        yield result


# 创建流
stream = Stream(
    handler=ReplyOnPause(
        echo,
        startup_fn=assistant.startup_audio,
    ),
    modality="audio",
    mode="send-receive",
)


def main():
    """主函数"""
    print("ASR + TTS 系统启动")
    print("说话 -> 识别 -> 播放")
    stream.ui.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
