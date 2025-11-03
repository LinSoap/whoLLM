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
                        你是一位专业的医院随访护士AI。请以**友好、简短、对话式**的语气，主动完成一次高血压患者的电话随访。
                        
                        **随访流程（按顺序主动执行）：**
                        
                        1.  **开场与确认：** 自我介绍（医院/科室/身份，例如：XX医院随访中心），确认对方身份，说明本次随访目的是关心血压和用药。
                        2.  **核心信息采集：**
                            * **询问血压值：** 询问最近测得的**血压数值**和测量频率。
                            * **询问症状：** 询问是否有**头晕、头痛、胸闷**等不适症状。
                            * **询问用药：** 确认是否**按时按量服药**，有无忘记或自行停药。
                        3.  **健康指导与复诊提醒：**
                            * 给予**简短的用药和低盐饮食**建议（基于患者反馈）。
                            * **主动提醒**患者下次**复诊的具体时间或建议**，并强调坚持服药。
                        4.  **结束语：** 礼貌地结束通话。
                        
                        **语气要求：** 温暖、专业、简练。每次回复请控制在30字以内，确保信息简洁明了。
""",
            }
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

    async def _text_to_speech(self, text: str) -> Tuple[int, np.ndarray]:
        """文字转语音"""
        try:
            await edge_tts.Communicate(text, self.tts_voice).save("output.mp3")

            # 加载TTS音频
            tts_audio = AudioSegment.from_mp3("output.mp3")
            tts_samples = np.array(tts_audio.get_array_of_samples(), dtype=np.int16)

            if tts_audio.channels == 2:
                tts_samples = tts_samples.reshape((-1, 2))
            else:
                tts_samples = tts_samples.reshape(-1)

            return (tts_audio.frame_rate, tts_samples)
        except Exception as e:
            print(f"TTS 错误: {e}")
            return (16000, np.array([]))

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
            tts_result = await self._text_to_speech(reply)

            if tts_result[1].size > 0:
                yield tts_result

            # 清理临时文件
            if os.path.exists("output.mp3"):
                os.unlink("output.mp3")

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
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
)


def main():
    """主函数"""
    print("ASR + TTS 系统启动")
    print("说话 -> 识别 -> 播放")
    stream.ui.launch()


if __name__ == "__main__":
    main()
