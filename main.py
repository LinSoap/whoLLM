from fastrtc import Stream, ReplyOnPause
from litellm import completion
import edge_tts
import numpy as np
import tempfile
import os
import wave
from pydub import AudioSegment

# FunASR
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


# 初始化 ASR
print("加载 ASR 模型...")
asr_model = AutoModel(model="paraformer-zh")
# punc_model = AutoModel(model="ct-punc")
cache = {}

# 维护对话上下文
messages = [
    {
        "role": "system",
        "content": """你是健康医院的医生助理，正在进行高血压患者的电话随访。请严格按照以下标准流程进行对话，用中文回复，保持友好和专业：

1. 建立连接与确认：
   - 确认患者身份，保证通话顺利。
   - 问候并确认身份："您好，我是健康医院内科的医生助理，请问您是xx先生/女士吗？"
   - 询问是否方便接听："现在接听电话方便吗？"
   - 说明来意："我们是来了解您的血压控制和服药情况的。"

2. 核心信息采集：
   - 了解患者病情控制、用药依从性和生活方式。
   - 血压控制："您最近测得的血压是多少？有没有在家做记录？"
   - 用药情况："您有没有按时服用降压药？有没有出现不舒服或忘记吃药的情况？"
   - 症状询问："最近有没有头晕、头痛、心慌等不适症状？"
   - 生活方式："您的饮食和运动情况怎么样？有没有注意低盐？"

3. 健康指导与评估：
   - 给予专业建议，解决疑问，强调重要事项。
   - 肯定/纠正：根据患者反馈，肯定做得好的地方，并纠正错误做法（如：高盐饮食、随意停药）。
   - 提醒重点："请您继续坚持监测血压并记录，一定要按时按量服药。"
   - 解答疑问："您对我给的建议还有其他疑问吗？"

4. 总结与记录：
   - 明确下一步，完成随访档案。
   - 复诊提醒："请您按照预约时间（或建议时间）到医院复诊。"
   - 紧急情况告知："如果出现剧烈不适，请立即就近就医。"
   - 记录：本次随访情况已记录。
# 回复要求：
    每次回复请控制在30字以内，确保信息简洁明了。
""",
    }
]

os.environ["OPENROUTER_API_KEY"] = ""


async def run_tts(text: str, output: str, voice: str = "zh-CN-XiaoxiaoNeural") -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output)


async def echo(audio: tuple[int, np.ndarray]):

    # 1. asr audio
    # Convert audio to float32 and normalize
    audio_data = audio[1].astype(np.float32)
    if audio[1].dtype == np.int16:
        audio_data /= 32768.0

    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        # Write WAV
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(audio[0])
            wav_file.writeframes(audio_int16.tobytes())

    try:
        res = asr_model.generate(
            input=temp_path,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=False,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        print("ASR 识别：", text)

        if not text.strip():
            return
        # 2. call llm
        messages.append({"role": "user", "content": text})

        print("历史消息：", messages)
        response = completion(
            model="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
            messages=messages,
        )
        assistant_reply = response.choices[0].message.content

        print("LLM 回复：", assistant_reply)
        messages.append({"role": "assistant", "content": assistant_reply})
        # 限制上下文长度
        if len(messages) > 20:
            messages[:] = messages[:1] + messages[-19:]  # 保留system和最近19个消息
        # 3. tts audio
        await run_tts(assistant_reply, "output.mp3")

        # Load TTS audio and yield
        tts_audio = AudioSegment.from_mp3("output.mp3")
        tts_samples = np.array(tts_audio.get_array_of_samples(), dtype=np.int16)
        if tts_audio.channels == 2:
            tts_samples = tts_samples.reshape((-1, 2))
        else:
            tts_samples = tts_samples.reshape(1, -1)
        yield (tts_audio.frame_rate, tts_samples)

        # Clean up
        os.unlink("output.mp3")

    finally:
        os.unlink(temp_path)


# 创建流
stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
)

# 启动
if __name__ == "__main__":
    print("ASR + TTS 系统启动")
    print("说话 -> 识别 -> 播放")
    # 可以调用asr_audio_streaming()进行测试
    # asr_audio_streaming()
    stream.ui.launch()
