#!/usr/bin/env python3
"""
Streamlit UI for OpenClip
Provides a web interface for video processing with AI-powered analysis
"""

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import os
import json
import re
import time
import threading
import tempfile
import copy
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from core.browser_preferences import (
    PREFERENCES_COOKIE_NAME,
    PREFERENCES_HYDRATED_FLAG,
    build_preferences_payload,
    deserialize_preferences_payload,
    merge_browser_preferences,
    serialize_preferences_payload,
)
from core.browser_session import (
    INPUT_TYPE_SERVER_PATH,
    INPUT_TYPE_UPLOAD,
    INPUT_TYPE_URL,
    normalize_input_type,
    reset_browser_state,
)
from core.clip_concat import concat_clips
from core.file_string_utils import FileStringUtils
from core.subtitle_burner import SubtitleBurner, SubtitleStyleConfig
# Import the video orchestrator
from video_orchestrator import VideoOrchestrator
from core.video_utils import VideoFileValidator
from core.config import API_KEY_ENV_VARS, DEFAULT_LLM_PROVIDER, DEFAULT_TITLE_STYLE, MAX_DURATION_MINUTES, WHISPER_MODEL, MAX_CLIPS, LLM_CONFIG, SUPPORTED_LLM_PROVIDERS
from core.clip_duration import CLIP_DURATION_PRESETS, DEFAULT_CLIP_LENGTH_PRESET, normalize_clip_length_preset
from core.transcript_generation_whisperx import WHISPERX_AVAILABLE
from core.editor import ensure_editor_service
from core.downloaders.bilibili_downloader import ImprovedBilibiliDownloader

# Import job manager for background processing
from job_manager import get_job_manager, JobStatus
from core.upload_staging import (
    OWNER_SESSION_QUERY_PARAM,
    SOURCE_KIND_SERVER_PATH,
    SOURCE_KIND_UPLOADED_FILE,
    SOURCE_KIND_URL,
    delete_upload_record,
    ensure_owner_session_id,
    list_uploads_for_owner,
    stage_uploaded_file,
    uploads_root_for_output_dir,
)


LOCAL_EDITOR_HOSTS = {'localhost', '127.0.0.1', '::1'}


def is_bilibili_url(url: str) -> bool:
    """Check if URL is a Bilibili URL"""
    if not url:
        return False
    bilibili_patterns = [
        r'https?://(?:www\.)?bilibili\.com/video/[Bb][Vv][0-9A-Za-z]+',
        r'https?://(?:www\.)?bilibili\.com/bangumi/',
        r'https?://(?:www\.)?b23\.tv/',
        r'https?://(?:m\.)?bilibili\.com/video/',
    ]
    return any(re.match(pattern, url) for pattern in bilibili_patterns)


def _is_local_editor_url(editor_url: str) -> bool:
    try:
        host = urlparse(editor_url).hostname or ''
    except Exception:
        return False
    return host.lower() in LOCAL_EDITOR_HOSTS


def _show_editor_launch(editor_url: str) -> None:
    st.session_state.editor_launch_url = editor_url
    if _is_local_editor_url(editor_url):
        webbrowser.open_new_tab(editor_url)
        st.success(f'Editor launched: {editor_url}')
    else:
        st.success('Editor is ready.')
        st.caption('Open the editor from this browser. The server will not open a tab for remote/LAN URLs.')
    st.markdown(f'[Open Clip Editor]({editor_url})')


async def get_bilibili_multi_parts(url: str, browser: Optional[str] = None, cookies_file: Optional[str] = None) -> list:
    """
    Get multi-part video information from Bilibili URL
    
    Returns:
        List of part information dicts, empty list if single video or error
    """
    try:
        downloader = ImprovedBilibiliDownloader(
            browser=browser,
            cookies=cookies_file
        )
        parts = await downloader.get_multi_part_info(url)
        return parts
    except Exception as e:
        st.warning(f"Could not detect multi-part video: {e}")
        return []


@st.cache_data(show_spinner=False)
def render_subtitle_style_preview(
    preset: str,
    font_size: str,
    vertical_position: str,
    background_style: str,
    subtitle_translation: Optional[str],
    ui_language: str,
) -> bytes | None:
    sample_original = (
        "这是一行原字幕预览效果。"
        if ui_language == "zh"
        else "This is an original subtitle preview line."
    )
    sample_translation = (
        "This is the translated subtitle preview."
        if ui_language == "zh"
        else "这是翻译字幕的预览效果。"
    )
    burner = SubtitleBurner(
        subtitle_style_config=SubtitleStyleConfig(
            preset=preset,
            font_size=font_size,
            vertical_position=vertical_position,
            bilingual_layout="auto",
            background_style=background_style,
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        preview_path = Path(tmpdir) / "subtitle_preview.png"
        ok = burner.generate_preview_image(
            preview_path,
            subtitle_translation=subtitle_translation,
            original_text=sample_original,
            translated_text=sample_translation,
        )
        if not ok or not preview_path.exists():
            return None
        return preview_path.read_bytes()

# Set page config
st.set_page_config(
    page_title="OpenClip",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# File Helpers (Refresh/Server Restart Safe)
# --------------------------
FILE_PATH = "persistent_data.json"

# Define translation dictionaries
TRANSLATIONS = {
    'en': {
        'app_title': 'OpenClip',
        'sidebar_title': '🎬 OpenClip',
        'input_type': 'Input Type',
        'video_url': 'Video URL',
        'local_file_path': 'Local Video File Path',
        'llm_provider': 'LLM Provider',
        'llm_model': 'LLM Model',
        'llm_base_url': 'LLM Base URL',
        'api_key': 'API Key',
        'title_style': 'Title Style',
        'language': 'Output Language',
        'output_dir': 'Output Directory',
        'cookie_settings': 'Cookie Settings',
        'cookie_mode': 'Cookie Mode',
        'cookie_mode_help': 'Choose how OpenClip should authenticate when downloading from Bilibili or YouTube.',
        'cookie_mode_browser': 'Browser cookies',
        'cookie_mode_file': 'Cookies file',
        'cookie_mode_none': 'No cookies',
        'cookie_browser': 'Cookie Browser',
        'cookie_browser_help': 'Use cookies extracted from this browser for remote video downloads.',
        'cookies_file': 'Cookies File Path (optional)',
        'use_background': 'Use Background Info',
        'user_intent': 'What are you looking for? (optional)',
        'user_intent_help': 'Describe what you want to find, e.g. "Messi\'s solo dribbles" or "all goals and saves". Leave blank to auto-detect highlights based on content type (e.g. goals, shots, saves for sports).',
        'user_intent_placeholder': 'e.g. all goals and key saves',
        'agentic_analysis': 'Enhanced Clip Quality',
        'agentic_analysis_help': 'Runs extra AI review and refinement passes to improve clip selection and boundaries. Slower, but usually produces cleaner standalone clips.',
        'visual_verification': 'Visual Verification',
        'visual_verification_help': 'Use video keyframes to verify highlights. Rejects moments where subtitles are exciting but the actual footage is mundane. Slower but more accurate.',
        'advanced_options': 'Advanced Options',
        'override_analysis_prompt': 'Override Analysis Prompt',
        'override_analysis_prompt_help': 'Replace the default analysis prompt entirely. For developers who want full control over how the LLM analyzes content.',
        'use_custom_prompt': 'Use Custom Highlight Analysis Prompt',
        'force_whisper': 'Force Local ASR Subtitles',
        'force_whisper_help': 'Ignore downloaded subtitles and generate a fresh local transcript.',
        'asr_backend': 'ASR Engine',
        'asr_backend_help': 'Choose speech recognition engine: local Whisper (free, slow) or LLM API (fast, costs tokens).',
        'long_video_acceleration': '⚡ Long Video Acceleration',
        'long_video_acceleration_help': 'Performs audio energy analysis first to identify exciting moments (commentary spikes, crowd roars), then only transcribes those segments. Best for sports, live events, and content with clear emotional peaks.',
        'generate_clips': 'Generate Clips',
        'max_clips': 'Max Clips',
        'clip_length': 'Max Clip Length',
        'clip_length_help': 'Maximum duration for each clip. Clips will not exceed this length.',
        'add_titles': 'Add Video Top Banner Title',
        'generate_cover': 'Generate Cover',
        'process_video': '🎬 Process Video',
        'background_info': 'Background Information',
        'custom_highlight_prompt': 'Custom Highlight Analysis Prompt',
        'save_background': 'Save Background Information',
        'save_custom_prompt': 'Save Custom Highlight Analysis Prompt',
        'background_info_notice': 'Please ensure your background information is in the `prompts/background/background.md` file.',
        'background_info_warning': 'The system will use the content of `prompts/background/background.md` for analysis.',
        'background_file_path': 'Background information is stored in:',
        'custom_prompt_editor': 'Custom Highlight Analysis Prompt Editor',
        'custom_prompt_info': 'Edit the prompt below to customize how engaging moments are analyzed.',
        'custom_prompt_help': 'Edit the prompt to customize engaging moments analysis. This will be used instead of the default prompt.',
        'current_saved_prompt': 'Current saved prompt file:',
        'results': '📊 Results',
        'saved_results': '📊 Saved Results',
        'clear_results': 'Clear Saved Results',
        'processing_success': '✅ Video processing completed successfully!',
        'processing_time': '⏱️ Processing time:',
        'video_information': '🎥 Video Information',
        'transcript_source': '📝 Transcript source:',
        'error': '❌ Unexpected error:',
        'reset_form': '🔄 Reset Form',
        'confirmation': 'Are you sure you want to reset all settings?',
        'yes_reset': 'Yes, Reset',
        'cancel': 'Cancel',
        'reset_success': '✅ Form has been reset!',
        'background_info_config': 'Background Information Configuration',
        'background_info_edit': 'Edit the background information to provide context about streamers, nicknames, or recurring themes for better analysis.',
        'background_info_help': 'Enter information about streamers, their nicknames, games, and common terms to improve AI analysis.',
        'background_save_success': 'Background information saved successfully!',
        'background_save_error': 'Failed to save background information:',
        'custom_prompt_save_success': 'Custom highlight analysis prompt saved successfully!',
        'custom_prompt_save_error': 'Failed to save custom highlight analysis prompt:',
        'select_input_type': 'Select input type',
        'enter_video_url': 'Enter Bilibili or YouTube URL',
        'video_url_help': 'Supports Bilibili (https://www.bilibili.com/video/BV...) and YouTube (https://www.youtube.com/watch?v=...) URLs',
        'upload_file': 'Upload File',
        'server_file_path': 'Local Path',
        'server_file_help': 'Enter a full path on the machine running OpenClip.',
        'local_file_help': 'Enter the full path to a local video file',
        'local_file_srt_notice': 'To use existing subtitles, place the .srt file in the same directory with the same filename (e.g. video.mp4 → video.srt).',
        'select_llm_provider': 'Select which AI provider to use for analysis',
        'llm_model_help': 'Override the model name for the selected provider. Leave blank to use the provider default.',
        'llm_base_url_help': 'Override the OpenAI-compatible chat completions endpoint for the selected provider. Leave blank to use the provider default.',
        'llm_model_unset': 'No default model is configured for this provider. Enter one below.',
        'enter_api_key': 'Enter API key or leave blank if set as environment variable',
        'api_key_help': 'You can also set the API_KEY environment variable',
        'custom_openai_api_key_help': 'For custom_openai, API key is optional. Leave it blank if your endpoint does not require Bearer authentication.',
        'select_title_style': 'Select the visual style for titles and covers',
        'select_language': 'Language for analysis and output',
        'enter_output_dir': 'Directory to save processed videos',
        'cookies_file_help': 'Optional path to a Netscape-format cookies.txt file. When provided, it overrides browser cookie extraction.',
        'force_whisper_help': 'Ignore platform subtitles and force local ASR generation. English uses Whisper; Chinese uses Paraformer.',
        'generate_clips_help': 'Generate video clips for engaging moments',
        'max_clips_help': 'Maximum number of highlight clips to generate',
        'add_titles_help': 'Add artistic titles to video clips (this step may be slow)',
        'burn_subtitles': 'Burn Subtitles into Clips',
        'burn_subtitles_help': 'Hard-burn SRT subtitles into clip videos (requires ffmpeg with libass)',
        'subtitle_translation': 'Subtitle Translation Language (optional)',
        'subtitle_translation_help': 'Translate subtitles to this language and burn bilingual subtitles. Select "None" for original language only.',
        'subtitle_translation_none': 'None',
        'subtitle_style_section': 'Subtitle Style',
        'subtitle_style_help': 'Preview and adjust burned subtitle appearance before processing.',
        'subtitle_style_preset': 'Subtitle Style',
        'subtitle_style_font_size': 'Subtitle Size',
        'subtitle_style_vertical_position': 'Vertical Position',
        'subtitle_style_background_style': 'Background Style',
        'subtitle_preview': 'Subtitle Preview',
        'subtitle_preview_help': 'Preview uses a built-in 1920x1080 sample background and the same ASS styling path as final burning.',
        'subtitle_preview_failed': 'Preview could not be rendered. Please verify ffmpeg with libass is available.',
        'generate_cover_help': 'Generate cover image for the video',
        'use_background_help': 'Use background information from prompts/background/background.md',
        'use_custom_prompt_help': 'Use custom prompt for highlight analysis',
        'advanced_config_notice': 'For advanced options (e.g. video split duration, Whisper model), edit `core/config.py`.',
        'speaker_references': 'Speaker References Directory (Preview)',
        'speaker_references_help': 'Directory of reference audio clips for speaker name mapping. Filename stem becomes the speaker name (e.g. references/Host.wav → "Host"). Requires HUGGINGFACE_TOKEN env var.',
        'speaker_references_unavailable': 'Speaker Identification (Preview) — requires extra dependencies: `uv sync --extra speakers`',
        'speaker_references_dir_not_found': '⚠️ Directory not found. Please check the path.',
        'speaker_references_token_warning': '⚠️ HUGGINGFACE_TOKEN is not set. Speaker identification will fail at runtime.',
    },
    'zh': {
        'app_title': 'OpenClip',
        'sidebar_title': '🎬 OpenClip',
        'input_type': '输入类型',
        'video_url': '视频链接',
        'local_file_path': '本地视频文件路径',
        'llm_provider': 'LLM 提供商',
        'llm_model': 'LLM 模型',
        'llm_base_url': 'LLM Base URL',
        'api_key': 'API 密钥',
        'title_style': '标题风格',
        'language': '输出语言',
        'output_dir': '输出目录',
        'cookie_settings': 'Cookie 设置',
        'cookie_mode': 'Cookie 模式',
        'cookie_mode_help': '选择 OpenClip 下载 Bilibili 或 YouTube 时如何进行身份验证。',
        'cookie_mode_browser': '浏览器 cookies',
        'cookie_mode_file': 'Cookies 文件',
        'cookie_mode_none': '不使用 cookies',
        'cookie_browser': 'Cookie 浏览器',
        'cookie_browser_help': '使用此浏览器中的 cookies 下载远程视频。',
        'cookies_file': 'Cookies 文件路径（可选）',
        'use_background': '使用背景信息提示词',
        'user_intent': '你想找什么？（可选）',
        'user_intent_help': '描述你想找的内容，例如"梅西的个人突破"或"所有进球和关键扑救"。留空则根据内容类型自动识别高光（如足球会找进球、射门、扑救等）。',
        'user_intent_placeholder': '例如：所有进球和关键扑救',
        'agentic_analysis': '深度优化',
        'agentic_analysis_help': '会增加额外的 AI 审查和优化步骤，以提升片段选择和边界质量。处理会更慢，但通常能得到更干净、独立性更强的片段。',
        'visual_verification': '视觉验证',
        'visual_verification_help': '通过视频关键帧验证高光时刻。会过滤掉字幕激动但画面平淡的片段。更准确但速度更慢。',
        'advanced_options': '高级选项',
        'override_analysis_prompt': '覆盖分析提示词',
        'override_analysis_prompt_help': '完全替换默认分析提示词。适合想完全控制LLM分析方式的开发者。',
        'use_custom_prompt': '使用自定义高光分析提示词',
        'force_whisper': '强制使用本地 ASR 生成字幕',
        'asr_backend': 'ASR 引擎',
        'asr_backend_help': '选择语音识别引擎：本地 Whisper（免费但慢）或 LLM API（快速，消耗 token）。',
        'long_video_acceleration': '⚡ 长视频加速',
        'long_video_acceleration_help': '先进行音频能量分析，识别解说激动/观众欢呼等高能量片段，仅对候选片段做语音转写。适合体育赛事、直播回放等有明显情绪波动的视频。',
        'generate_clips': '生成高光片段',
        'max_clips': '最大片段数',
        'clip_length': '片段时长上限',
        'clip_length_help': '每个片段的最大时长，不会超过此上限。',
        'add_titles': '添加视频上方横幅标题',
        'generate_cover': '生成封面',
        'process_video': '🎬 处理视频',
        'background_info': '背景信息',
        'custom_highlight_prompt': '自定义高光分析提示',
        'save_background': '保存背景信息',
        'save_custom_prompt': '保存自定义高光分析提示',
        'background_info_notice': '请确保您的背景信息在 `prompts/background/background.md` 文件中。',
        'background_info_warning': '系统将使用 `prompts/background/background.md` 文件的内容进行分析。',
        'background_file_path': '背景信息存储在：',
        'custom_prompt_editor': '自定义高光分析提示编辑器',
        'custom_prompt_info': '编辑下面的提示以自定义如何分析精彩时刻。',
        'custom_prompt_help': '编辑提示以自定义精彩时刻分析。这将替代默认提示。',
        'current_saved_prompt': '当前保存的提示文件：',
        'results': '📊 结果',
        'saved_results': '📊 保存的结果',
        'clear_results': '清除保存的结果',
        'processing_success': '✅ 视频处理成功完成！',
        'processing_time': '⏱️ 处理时间：',
        'video_information': '🎥 视频信息',
        'transcript_source': '📝 字幕来源：',
        'error': '❌ 意外错误：',
        'reset_form': '🔄 重置表单',
        'confirmation': '确定要重置所有设置吗？',
        'yes_reset': '是的，重置',
        'cancel': '取消',
        'reset_success': '✅ 表单已重置！',
        'background_info_config': '背景信息配置',
        'background_info_edit': '编辑背景信息以提供有关主播、昵称或 recurring themes 的上下文，以获得更好的分析。',
        'background_info_help': '输入有关主播、他们的昵称、游戏和常用术语的信息，以改善 AI 分析。',
        'background_save_success': '背景信息保存成功！',
        'background_save_error': '保存背景信息失败：',
        'custom_prompt_save_success': '自定义高光分析提示保存成功！',
        'custom_prompt_save_error': '保存自定义高光分析提示失败：',
        'select_input_type': '选择输入类型',
        'enter_video_url': '输入 B 站或 YouTube 链接',
        'video_url_help': '支持 B 站 (https://www.bilibili.com/video/BV...) 和 YouTube (https://www.youtube.com/watch?v=...) 链接',
        'upload_file': '上传文件',
        'server_file_path': '本地路径',
        'server_file_help': '输入运行 OpenClip 的那台机器上的完整文件路径。',
        'local_file_help': '输入本地视频文件的完整路径',
        'local_file_srt_notice': '如需使用已有字幕，请将 .srt 文件放在同目录下，文件名保持一致（如 video.mp4 → video.srt）。',
        'select_llm_provider': '选择用于分析的 AI 提供商',
        'llm_model_help': '覆盖当前提供商使用的模型名。留空则使用该提供商的默认配置。',
        'llm_base_url_help': '覆盖当前提供商使用的 OpenAI 兼容 chat completions 接口地址。留空则使用默认配置。',
        'llm_model_unset': '当前提供商没有默认模型，请在下方填写。',
        'enter_api_key': '输入 API 密钥或留空（如果已设置为环境变量）',
        'api_key_help': '您也可以设置 API_KEY 环境变量',
        'custom_openai_api_key_help': '对于 custom_openai，API Key 是可选的。如果你的兼容接口不需要 Bearer 鉴权，可以留空。',
        'select_title_style': '选择标题和封面的视觉风格',
        'select_language': '分析和输出的语言',
        'enter_output_dir': '保存处理后视频的目录',
        'cookies_file_help': '可选的 Netscape 格式 cookies.txt 文件路径。提供后会优先使用该文件，而不是从浏览器提取 cookie。',
        'force_whisper_help': '忽略平台字幕并强制使用本地 ASR 生成字幕。英文走 Whisper，中文走 Paraformer。',
        'generate_clips_help': '为精彩时刻生成视频片段',
        'max_clips_help': '生成高光片段的最大数量',
        'add_titles_help': '为视频片段添加艺术标题（此步骤可能较慢）',
        'burn_subtitles': '将字幕烧录到片段中',
        'burn_subtitles_help': '将 SRT 字幕硬烧到剪辑视频中（需要带 libass 的 ffmpeg）',
        'subtitle_translation': '字幕翻译语言（可选）',
        'subtitle_translation_help': '将字幕翻译为该语��并烧录双语字幕。选择"无"则仅烧录原语言字幕。',
        'subtitle_translation_none': '无',
        'subtitle_style_section': '字幕样式',
        'subtitle_style_help': '在正式处理前预览并调整烧录字幕的显示效果。',
        'subtitle_style_preset': '字幕风格',
        'subtitle_style_font_size': '字幕大小',
        'subtitle_style_vertical_position': '垂直位置',
        'subtitle_style_background_style': '背景样式',
        'subtitle_preview': '字幕预览',
        'subtitle_preview_help': '预览使用内置 1920x1080 示例背景，并复用最终字幕烧录的 ASS 样式路径。',
        'subtitle_preview_failed': '字幕预览生成失败，请确认 ffmpeg 已包含 libass。',
        'generate_cover_help': '为视频生成封面图像',
        'use_background_help': '������ prompts/background/background.md 中的背景信息',
        'use_custom_prompt_help': '使用自定义提示进行高光分析',
        'advanced_config_notice': '如需调整高级选项（如视频分割时长、Whisper 模型），请编辑 `core/config.py`。',
        'speaker_references': '说话人参考音频目录（预览版）',
        'speaker_references_help': '包含参考音频片段的目录，用于说话人姓名映射。文件名即说话人姓名（如 references/Host.wav → "Host"）。需要设置 HUGGINGFACE_TOKEN 环境变量。',
        'speaker_references_unavailable': '说话人识别（预览版）— 需要额外依赖：`uv sync --extra speakers`',
        'speaker_references_dir_not_found': '⚠️ 目录不存在，请检查路径。',
        'speaker_references_token_warning': '⚠️ 未设置 HUGGINGFACE_TOKEN，运行时说话人识别将失败。',
    }
}


def build_default_llm_provider_settings():
    return {
        provider: {
            "model": "",
            "base_url": "",
        }
        for provider in SUPPORTED_LLM_PROVIDERS
    }


def backfill_llm_provider_settings(saved: Dict[str, Any]) -> None:
    settings = saved.get("llm_provider_settings")
    if not isinstance(settings, dict):
        settings = {}

    defaults = build_default_llm_provider_settings()
    for provider, provider_defaults in defaults.items():
        current = settings.get(provider)
        if not isinstance(current, dict):
            current = {}
        for key, value in provider_defaults.items():
            current.setdefault(key, value)
        settings[provider] = current

    saved["llm_provider_settings"] = settings
    if saved.get("llm_provider") not in SUPPORTED_LLM_PROVIDERS:
        saved["llm_provider"] = DEFAULT_LLM_PROVIDER


# Define default data
DEFAULT_DATA = {
    # Checkboxes
    'use_background': False,
    'use_custom_prompt': False,
    'force_whisper': False,
    'asr_backend': 'whisper',
    'long_video_acceleration': False,
    'generate_clips': True,
    'max_clips': MAX_CLIPS,
    'clip_length_preset': DEFAULT_CLIP_LENGTH_PRESET,
    'add_titles': False,
    'burn_subtitles': False,
    'subtitle_translation': None,
    'subtitle_style_preset': 'default',
    'subtitle_style_font_size': 'medium',
    'subtitle_style_vertical_position': 'bottom',
    'subtitle_style_background_style': 'none',
    'generate_cover': True,
    # Other form elements
    'input_type': INPUT_TYPE_URL,
    'video_source': "",
    'llm_provider': DEFAULT_LLM_PROVIDER,
    'llm_provider_settings': build_default_llm_provider_settings(),
    'api_key': "",
    'title_style': DEFAULT_TITLE_STYLE,
    'language': "zh",
    'output_dir': "processed_videos",
    'cookie_mode': "none",
    'cookie_browser': "chrome",
    'cookies_file': "",
    'custom_prompt_file': None,
    'custom_prompt_text': "",
    'speaker_references_dir': "",
    'mode': 'engaging_moments',
    'user_intent': "",
    'agentic_analysis': False,
    # Language setting
    'ui_language': "zh",
    # Processing result
    'processing_result': None,
}

# Initialize file if it doesn't exist
if not os.path.exists(FILE_PATH):
    with open(FILE_PATH, "w") as f:
        json.dump(DEFAULT_DATA, f, indent=2)

def load_from_file():
    with open(FILE_PATH, "r") as f:
        saved = json.load(f)
    # Backfill any new default keys missing from older saved files
    for key, value in DEFAULT_DATA.items():
        if key not in saved:
            saved[key] = copy.deepcopy(value)
    backfill_llm_provider_settings(saved)
    saved['input_type'] = normalize_input_type(saved.get('input_type'))
    return saved

def _browser_preferences_bridge_script() -> str:
    return """
        const rootWindow = window.parent || window;
        const currentUrl = new URL(rootWindow.location.href);
    """


def _render_browser_preferences_writer(payload: dict[str, Any]) -> None:
    serialized = serialize_preferences_payload(payload)
    bridge = _browser_preferences_bridge_script()
    components.html(
        f"""
        <script>
        {bridge}
        rootWindow.document.cookie = {json.dumps(PREFERENCES_COOKIE_NAME)} + "=" + {json.dumps(serialized)} + "; path=/; max-age=31536000; SameSite=Lax";
        </script>
        """,
        height=0,
        width=0,
    )


def _render_browser_preferences_clearer() -> None:
    bridge = _browser_preferences_bridge_script()
    components.html(
        f"""
        <script>
        {bridge}
        rootWindow.document.cookie = {json.dumps(PREFERENCES_COOKIE_NAME)} + "=; path=/; max-age=0; SameSite=Lax";
        </script>
        """,
        height=0,
        width=0,
    )


# Load persistent data as startup seed only; browser sessions use their own runtime state.
persisted_data = load_from_file()

# Initialize reset counter in session state
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# Initialize processing state
if 'processing' not in st.session_state:
    st.session_state.processing = False
    st.session_state.cancel_event = threading.Event()
    st.session_state.processing_thread = None
    st.session_state.processing_outcome = {'result': None, 'error': None}
    st.session_state.progress_state = {'status': '', 'progress': 0}

# Initialize job manager
job_manager = get_job_manager()

# Initialize processing job tracking (supports multiple concurrent jobs)
if 'processing_job_ids' not in st.session_state:
    st.session_state.processing_job_ids = []
    st.session_state.processing = False

if 'browser_data' not in st.session_state:
    st.session_state.browser_data = reset_browser_state(DEFAULT_DATA)

if 'uploads_root' not in st.session_state:
    st.session_state.uploads_root = str(uploads_root_for_output_dir(DEFAULT_DATA.get('output_dir', 'processed_videos')))

if PREFERENCES_HYDRATED_FLAG not in st.session_state:
    st.session_state[PREFERENCES_HYDRATED_FLAG] = False
if 'remembered_preferences_payload' not in st.session_state:
    st.session_state['remembered_preferences_payload'] = None
if 'suspend_preference_writeback' not in st.session_state:
    st.session_state['suspend_preference_writeback'] = False

raw_preferences_cookie = st.context.cookies.get(PREFERENCES_COOKIE_NAME)
if not st.session_state[PREFERENCES_HYDRATED_FLAG]:
    payload = deserialize_preferences_payload(raw_preferences_cookie)
    st.session_state[PREFERENCES_HYDRATED_FLAG] = True
    if payload is not None:
        st.session_state.browser_data = merge_browser_preferences(DEFAULT_DATA, st.session_state.browser_data, payload)
        # Restore owner session ID from cookie if available
        saved_oc_session = (payload.get("prefs") or {}).get("oc_session")
        if saved_oc_session:
            st.session_state[OWNER_SESSION_QUERY_PARAM] = saved_oc_session
        st.session_state['remembered_preferences_payload'] = raw_preferences_cookie
        st.session_state['suspend_preference_writeback'] = False
    else:
        st.session_state['remembered_preferences_payload'] = None
        st.session_state['suspend_preference_writeback'] = True

data = st.session_state.browser_data
current_lang = data.get('ui_language', 'zh')
t = TRANSLATIONS[current_lang]
current_owner_session_id = ensure_owner_session_id(st.query_params, st.session_state)

# Don't auto-resume tracking on new tabs - let user choose via "Watch Progress" button
# This allows each tab to track different jobs independently

# Track if we just processed a video
just_processed = False

# Function to display results
def display_results(result):
    """Display processing results consistently"""
    if result.success:
        st.success(t['processing_success'])
        
        # Display processing time
        st.info(f"{t['processing_time']} {result.processing_time:.2f} seconds")
        
        # Display video info
        if result.video_info:
            with st.expander(t['video_information']):
                for key, value in result.video_info.items():
                    st.write(f"**{key.capitalize()}:** {value}")
        
        # Display transcript info
        if result.transcript_source:
            st.info(f"{t['transcript_source']} {result.transcript_source}")
        
        # Display analysis info
        if result.engaging_moments_analysis:
            analysis = result.engaging_moments_analysis
            with st.expander("🧠 Analysis Results"):
                st.write(f"Total parts analyzed: {analysis.get('total_parts_analyzed', 0)}")
                if analysis.get('top_moments'):
                    moments = analysis['top_moments']
                    if isinstance(moments, dict) and 'top_engaging_moments' in moments:
                        moments = moments['top_engaging_moments']
                    
                    if isinstance(moments, list):
                        st.write(f"Found {len(moments)} engaging moments")
                        for i, moment in enumerate(moments):
                            with st.container():
                                st.subheader(f"Rank {i+1}: {moment.get('title', 'Untitled')}")
                                if 'description' in moment:
                                    st.write(moment['description'])
                                if 'timestamp' in moment:
                                    st.write(f"Timestamp: {moment['timestamp']}")
        
        # Display clip info
        output_dir = None
        if result.clip_generation and result.clip_generation.get('success'):
            clips = result.clip_generation
            with st.expander("🎬 Generated Clips"):
                st.write(f"Generated {clips.get('total_clips', 0)} clips")
                if clips.get('clips_info'):
                    output_dir = Path(clips.get('output_dir', ''))
                    # Tag filter
                    all_tags = sorted(set(
                        tag for clip in clips['clips_info'] for tag in clip.get('tags', [])
                    ))
                    filtered_clips = clips['clips_info']
                    if all_tags:
                        selected_tags = st.multiselect("🏷 Filter by tag", all_tags, default=[], key="clip_tag_filter")
                        if selected_tags:
                            filtered_clips = [c for c in filtered_clips if any(t in selected_tags for t in c.get('tags', []))]
                    # Select all / deselect all
                    sel_col1, sel_col2 = st.columns(2)
                    with sel_col1:
                        select_all = st.button("全选", key="clip_select_all")
                    with sel_col2:
                        deselect_all = st.button("取消全选", key="clip_deselect_all")
                    if select_all:
                        for i in range(len(filtered_clips)):
                            st.session_state[f"select_clip_{i}"] = True
                    if deselect_all:
                        for i in range(len(filtered_clips)):
                            st.session_state[f"select_clip_{i}"] = False

                    # Create columns for side-by-side display (2 per row) with minimal gap
                    cols = st.columns(2, gap="xxsmall")
                    selected_clip_paths = []
                    for i, clip in enumerate(filtered_clips):
                        clip_filename = clip.get('filename')
                        if clip_filename:
                            clip_path = output_dir / clip_filename
                            if clip_path.exists():
                                with cols[i % 2]:
                                    checked = st.checkbox("选择", key=f"select_clip_{i}", value=False)
                                    if checked:
                                        selected_clip_paths.append(str(clip_path))
                                    st.video(str(clip_path), width=450)
                                    time_range = clip.get('time_range', '')
                                    tags_str = ", ".join(clip.get('tags', []))
                                    caption = f"**#{clip.get('rank', i+1)} {clip.get('title', 'Untitled')}**"
                                    if time_range:
                                        caption += f"  \n⏱ {time_range}"
                                    if tags_str:
                                        caption += f"  \n🏷 {tags_str}"
                                    st.caption(caption)

                    if selected_clip_paths:
                        # Reorder clips via drag-style number inputs
                        st.markdown("**调整顺序** (数字越小越靠前):")
                        order_cols = st.columns(min(len(selected_clip_paths), 4))
                        order_indices = []
                        for idx, sp in enumerate(selected_clip_paths):
                            with order_cols[idx % len(order_cols)]:
                                fname = Path(sp).stem[:20]
                                o = st.number_input(fname, min_value=1, max_value=len(selected_clip_paths), value=idx+1, key=f"clip_order_{idx}")
                                order_indices.append((o, sp))
                        order_indices.sort(key=lambda x: x[0])
                        ordered_paths = [p for _, p in order_indices]

                        if st.button("🎬 生成高光合集视频"):
                            compilation_path = str(output_dir / "highlight_compilation.mp4")
                            with st.spinner("正在拼接高光合集..."):
                                ok = concat_clips(ordered_paths, compilation_path)
                            if ok and Path(compilation_path).exists():
                                st.success("合集视频生成成功！")
                                st.video(compilation_path)
                            else:
                                st.error("合集视频生成失败，请检查日志。")

                    # Delete rejected clips
                    unselected = [
                        str(output_dir / clip.get('filename'))
                        for i, clip in enumerate(filtered_clips)
                        if clip.get('filename') and not st.session_state.get(f"select_clip_{i}", False)
                        and (output_dir / clip.get('filename')).exists()
                    ]
                    if unselected:
                        if st.button("🗑 删除未选中的片段", type="secondary"):
                            for p in unselected:
                                Path(p).unlink(missing_ok=True)
                                srt_p = Path(p).with_suffix('.srt')
                                srt_p.unlink(missing_ok=True)
                            st.success(f"已删除 {len(unselected)} 个片段")
                            st.rerun()
        
        # Display post-processing info (titles and/or subtitles)
        if getattr(result, 'post_processing', None) and result.post_processing.get('success'):
            titles = result.post_processing
            with st.expander("✨ Post-Processed Clips"):
                st.write(f"Post-processed {titles.get('total_clips', 0)} clips")
                post_dir = Path(titles.get('output_dir', ''))
                if titles.get('processed_clips'):
                    clips_to_show = [
                        (post_dir / c['filename'], c.get('title', 'Untitled'), c.get('time_range', ''))
                        for c in titles['processed_clips']
                        if c.get('filename')
                    ]
                elif post_dir.exists():
                    clips_to_show = sorted(
                        [(p, p.stem, '') for p in post_dir.glob('*.mp4') if not p.name.startswith('_')]
                    )
                else:
                    clips_to_show = []
                if clips_to_show:
                    cols = st.columns(2, gap="xxsmall")
                    for i, (clip_path, clip_title, time_range) in enumerate(clips_to_show):
                        if clip_path.exists():
                            with cols[i % 2]:
                                st.video(str(clip_path), width=450)
                                caption = f"**{clip_title}**"
                                if time_range:
                                    caption += f"  \n⏱ {time_range}"
                                st.caption(caption)
        
        # Display cover info
        if result.cover_generation and result.cover_generation.get('success'):
            covers = result.cover_generation
            with st.expander("🖼️ Generated Covers"):
                st.write(f"Generated {covers.get('total_covers', 0)} cover images")
                if covers.get('covers'):
                    cols = st.columns(2, gap="xxsmall")
                    for i, cover in enumerate(covers['covers']):
                        cover_path = cover.get('path')
                        if cover_path and Path(cover_path).exists():
                            with cols[i % 2]:
                                st.image(cover_path, caption=cover.get('title', 'Untitled'), width=450)
        
        # Display output directory
        if output_dir:
            st.info(f"📁 All outputs saved to: {output_dir}")

        editor_project = getattr(result, 'editor_project', None)
        if editor_project and editor_project.get('project_id'):
            st.header('🛠️ Clip Editor')
            st.caption('Open the post-generation editor for timeline, subtitle, and cover-title adjustments.')
            if st.button('🛠️ Open in Editor', key=f"open_editor_{editor_project.get('project_id')}"):
                try:
                    editor_url = ensure_editor_service(
                        editor_project['project_id'],
                        projects_root=editor_project.get('projects_root') or str(Path(editor_project['project_root']).parent),
                        jobs_dir=str((Path.cwd() / 'jobs').resolve()),
                        open_browser=False,
                    )
                    _show_editor_launch(editor_url)
                except Exception as exc:
                    st.warning(f'Editor unavailable: {exc}')
    else:
        st.error(f"{t['error']} {result.error_message}")


def _is_editor_rerender_job(job) -> bool:
    return (job.options or {}).get('kind') == 'editor_rerender'


def _launch_editor_for_job(job) -> None:
    project_id = (job.options or {}).get('project_id')
    projects_root = (job.options or {}).get('projects_root') or str((Path.cwd() / 'processed_videos').resolve())
    if not project_id:
        st.warning('Editor project ID missing for this job.')
        return
    try:
        editor_url = ensure_editor_service(
            project_id,
            projects_root=projects_root,
            jobs_dir=str((Path.cwd() / 'jobs').resolve()),
            open_browser=False,
        )
        _show_editor_launch(editor_url)
    except Exception as exc:
        st.warning(f'Editor unavailable: {exc}')

# Custom CSS (Slate light theme polish — core palette lives in .streamlit/config.toml)
st.markdown(
    """
<style>
    .stButton > button {
        border-radius: 6px;
    }
    .stFileUploader > label,
    .stTextInput > label,
    .stSelectbox > label,
    .stCheckbox > label {
        font-weight: 600;
    }
    .stMainBlockContainer .stCheckbox label p {
        font-size: 0.8rem !important;
    }
    .video-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }
    .stColumns > div { gap: 0.25rem !important; }
    .stColumn { padding: 0 !important; margin: 0 !important; }
    .stVideo {
        margin-bottom: 0.5rem !important;
        margin-right: 0 !important;
        margin-left: 0 !important;
    }
    .stMarkdown {
        margin-bottom: 0.5rem !important;
        margin-right: 0 !important;
        margin-left: 0 !important;
    }
    .streamlit-expanderContent { padding: 0.5rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.title("🎬 OpenClip")
st.markdown("""
A lightweight automated video processing pipeline that identifies and extracts the most engaging moments from long-form videos (especially livestream recordings). Uses AI-powered analysis to find highlights, generates clips, and adds artistic titles.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # UI Language Selector
    ui_language = st.selectbox(
        "UI Language",
        options=["English", "中文"],
        index=["English", "中文"].index("中文" if current_lang == "zh" else "English"),
        help="Select language for the user interface",
        key=f"ui_language_{st.session_state.reset_counter}"
    )
    new_lang = "zh" if ui_language == "中文" else "en"
    if new_lang != current_lang:
        data['ui_language'] = new_lang
        st.rerun()

    st.divider()
    
    # Video input options
    upload_option_label = t.get('upload_file', 'Upload File')
    server_path_option_label = t.get('server_file_path', 'Server File Path (host only)')
    input_type_options = [
        (INPUT_TYPE_URL, 'Video URL'),
        (INPUT_TYPE_SERVER_PATH, server_path_option_label),
        (INPUT_TYPE_UPLOAD, upload_option_label),
    ]
    current_input_type = normalize_input_type(data.get('input_type'))
    input_type = st.radio(
        t['input_type'],
        options=[value for value, _label in input_type_options],
        format_func=lambda value: dict(input_type_options).get(value, value),
        index=[value for value, _label in input_type_options].index(current_input_type),
        key=f"input_type_{st.session_state.reset_counter}"
    )
    data['input_type'] = input_type

    uploaded_file = None
    if input_type == INPUT_TYPE_URL:
        video_source = st.text_input(
            t['video_url'],
            value=data['video_source'],
            placeholder=t['enter_video_url'],
            help=t['video_url_help'],
            key=f"video_source_{st.session_state.reset_counter}"
        )
        data['video_source'] = video_source
    elif input_type == INPUT_TYPE_UPLOAD:
        uploaded_file = st.file_uploader(
            upload_option_label,
            type=sorted(ext.lstrip('.') for ext in VideoFileValidator.VIDEO_EXTENSIONS),
            key=f"upload_file_{st.session_state.reset_counter}",
            help=t.get('upload_file_help', 'Upload a video file from this browser. The file will be staged on the host until deleted.'),
        )
        video_source = uploaded_file.name if uploaded_file else ''
        data['video_source'] = video_source
        st.caption(t.get('upload_file_notice', 'Uploaded files are stored on the host until you delete them.'))
    else:
        video_source = st.text_input(
            server_path_option_label,
            value=data.get('video_source', "") if current_input_type == INPUT_TYPE_SERVER_PATH else "",
            help=t.get('server_file_help', 'Enter a full path on the host machine. This is intended for the server operator only.'),
            key=f"server_file_path_{st.session_state.reset_counter}"
        )
        st.caption(t['local_file_srt_notice'])
        data['video_source'] = video_source

    cookie_mode = data.get('cookie_mode', 'none')
    cookie_browser = data.get('cookie_browser', 'chrome')
    cookies_file = ""

    with st.expander(t['cookie_settings'], expanded=False):
        cookie_mode_options = [t['cookie_mode_none'], t['cookie_mode_browser'], t['cookie_mode_file']]
        cookie_mode_values = ['none', 'browser', 'file']
        current_cookie_mode_idx = cookie_mode_values.index(cookie_mode) if cookie_mode in cookie_mode_values else 0
        cookie_mode_label = st.selectbox(
            t['cookie_mode'],
            options=cookie_mode_options,
            index=current_cookie_mode_idx,
            help=t['cookie_mode_help'],
            key=f"cookie_mode_{st.session_state.reset_counter}",
        )
        cookie_mode = cookie_mode_values[cookie_mode_options.index(cookie_mode_label)]
        data['cookie_mode'] = cookie_mode

        if cookie_mode == 'browser':
            browser_options = ['chrome', 'firefox', 'edge', 'safari']
            current_browser_idx = browser_options.index(cookie_browser) if cookie_browser in browser_options else 0
            cookie_browser = st.selectbox(
                t['cookie_browser'],
                options=browser_options,
                index=current_browser_idx,
                help=t['cookie_browser_help'],
                key=f"cookie_browser_{st.session_state.reset_counter}",
            )
            data['cookie_browser'] = cookie_browser
            data['cookies_file'] = ""
            cookies_file = ""
        elif cookie_mode == 'file':
            cookies_file = st.text_input(
                t['cookies_file'],
                value=data.get('cookies_file', ''),
                help=t['cookies_file_help'],
                placeholder="cookies.txt",
                key=f"cookies_file_{st.session_state.reset_counter}"
            )
            data['cookies_file'] = cookies_file
            if cookies_file and not Path(cookies_file).is_file():
                st.caption("⚠️ Cookies file not found. Download will fail until this path is corrected.")
        else:
            data['cookies_file'] = ""
            cookies_file = ""
    
    # LLM provider selection
    llm_provider_options = list(SUPPORTED_LLM_PROVIDERS)
    llm_provider = st.selectbox(
        t['llm_provider'],
        options=llm_provider_options,
        index=llm_provider_options.index(data['llm_provider']),
        help=t['select_llm_provider'],
        key=f"llm_provider_{st.session_state.reset_counter}"
    )
    data['llm_provider'] = llm_provider

    provider_settings = data['llm_provider_settings'].setdefault(
        llm_provider,
        {"model": "", "base_url": ""},
    )
    provider_default_model = (LLM_CONFIG[llm_provider].get('default_model') or "").strip()
    provider_default_base_url = (LLM_CONFIG[llm_provider].get('base_url') or "").strip()

    llm_model = st.text_input(
        t['llm_model'],
        value=provider_settings.get('model', ''),
        placeholder=provider_default_model,
        help=t['llm_model_help'],
        key=f"llm_model_{st.session_state.reset_counter}"
    )
    provider_settings['model'] = llm_model.strip()

    llm_base_url = st.text_input(
        t['llm_base_url'],
        value=provider_settings.get('base_url', ''),
        placeholder=provider_default_base_url,
        help=t['llm_base_url_help'],
        key=f"llm_base_url_{st.session_state.reset_counter}"
    )
    provider_settings['base_url'] = llm_base_url.strip()

    resolved_llm_model = (provider_settings['model'] or provider_default_model).strip()
    resolved_llm_base_url = (provider_settings['base_url'] or provider_default_base_url).strip()

    if not resolved_llm_model:
        st.warning(t['llm_model_unset'])

    # API key input (optional, since it can be set via environment variable)
    api_key_env_var = API_KEY_ENV_VARS.get(llm_provider, "QWEN_API_KEY")
    api_key = st.text_input(
        f"{llm_provider.upper()} {t['api_key']}",
        value=data['api_key'],
        type="password",
        placeholder=t['enter_api_key'],
        help=t['api_key_help'],
        key=f"api_key_{st.session_state.reset_counter}"
    )
    data['api_key'] = api_key
    if llm_provider == "custom_openai":
        st.caption(t['custom_openai_api_key_help'])
    
    title_style = data['title_style']

    # Additional options
    languages = ["zh", "en", "vi"]
    language = st.selectbox(
        t['language'],
        options=languages,
        index=languages.index(data['language']),
        help=t['select_language'],
        key=f"language_{st.session_state.reset_counter}"
    )
    data['language'] = language

    # Clip generation options (always enabled)
    generate_clips = True
    data['generate_clips'] = generate_clips

    max_clips = st.number_input(
        t['max_clips'],
        min_value=1,
        max_value=100,
        value=int(data['max_clips']),
        step=1,
        help=t['max_clips_help'],
        key=f"max_clips_{st.session_state.reset_counter}"
    )
    data['max_clips'] = max_clips

    clip_length_options = list(CLIP_DURATION_PRESETS.keys())
    current_clip_length = normalize_clip_length_preset(data.get('clip_length_preset'))
    clip_length_preset = st.selectbox(
        t['clip_length'],
        options=clip_length_options,
        index=clip_length_options.index(current_clip_length),
        format_func=lambda preset: CLIP_DURATION_PRESETS[preset].label,
        help=t['clip_length_help'],
        key=f"clip_length_preset_{st.session_state.reset_counter}"
    )
    data['clip_length_preset'] = clip_length_preset

    # User intent
    user_intent = st.text_input(
        t['user_intent'],
        value=data.get('user_intent', ''),
        placeholder=t['user_intent_placeholder'],
        help=t['user_intent_help'],
        key=f"user_intent_{st.session_state.reset_counter}"
    )
    data['user_intent'] = user_intent

    # Output directory
    output_dir = st.text_input(
        t['output_dir'],
        value=data['output_dir'],
        help=t['enter_output_dir'],
        key=f"output_dir_{st.session_state.reset_counter}"
    )
    data['output_dir'] = output_dir

    # Checkboxes for additional options
    generate_cover = st.checkbox(
        t['generate_cover'],
        value=data['generate_cover'],
        help=t['generate_cover_help'],
        key=f"generate_cover_{st.session_state.reset_counter}"
    )
    data['generate_cover'] = generate_cover
    
    burn_subtitles = st.checkbox(
        t['burn_subtitles'],
        value=data.get('burn_subtitles', False),
        help=t['burn_subtitles_help'],
        key=f"burn_subtitles_{st.session_state.reset_counter}"
    )
    data['burn_subtitles'] = burn_subtitles

    agentic_analysis = st.checkbox(
        t['agentic_analysis'],
        value=bool(data.get('agentic_analysis', False)),
        help=t['agentic_analysis_help'],
        key=f"agentic_analysis_{st.session_state.reset_counter}"
    )
    data['agentic_analysis'] = agentic_analysis

    visual_verification = st.checkbox(
        t['visual_verification'],
        value=bool(data.get('visual_verification', False)),
        help=t['visual_verification_help'],
        key=f"visual_verification_{st.session_state.reset_counter}"
    )
    data['visual_verification'] = visual_verification

    if burn_subtitles:
        st.markdown(f"**{t['subtitle_style_section']}**")
        st.caption(t['subtitle_style_help'])

        # Map display labels to API values
        subtitle_lang_options = [t['subtitle_translation_none'], '中文', 'English']
        subtitle_lang_values = [None, 'Simplified Chinese', 'English']
        current_val = data.get('subtitle_translation', None)
        current_idx = subtitle_lang_values.index(current_val) if current_val in subtitle_lang_values else 0
        subtitle_lang_label = st.selectbox(
            t['subtitle_translation'],
            options=subtitle_lang_options,
            index=current_idx,
            help=t['subtitle_translation_help'],
            key=f"subtitle_translation_{st.session_state.reset_counter}"
        )
        subtitle_translation = subtitle_lang_values[subtitle_lang_options.index(subtitle_lang_label)]
        data['subtitle_translation'] = subtitle_translation

        subtitle_preset_values = ['default', 'clean', 'high_contrast', 'stream']
        subtitle_preset_labels = {
            'en': {'default': 'Default', 'clean': 'Clean', 'high_contrast': 'High Contrast', 'stream': 'Stream'},
            'zh': {'default': '默认', 'clean': '简洁', 'high_contrast': '高对比', 'stream': '直播'},
        }
        subtitle_size_values = ['small', 'medium', 'large']
        subtitle_size_labels = {
            'en': {'small': 'Small', 'medium': 'Medium', 'large': 'Large'},
            'zh': {'small': '小', 'medium': '中', 'large': '大'},
        }
        subtitle_position_values = ['bottom', 'lower_middle', 'middle']
        subtitle_position_labels = {
            'en': {'bottom': 'Bottom', 'lower_middle': 'Lower Middle', 'middle': 'Middle'},
            'zh': {'bottom': '底部', 'lower_middle': '偏下居中', 'middle': '居中'},
        }
        subtitle_background_values = ['none', 'light_box', 'solid_box']
        subtitle_background_labels = {
            'en': {
                'none': 'None',
                'light_box': 'Light Box',
                'solid_box': 'Solid Box',
            },
            'zh': {
                'none': '无',
                'light_box': '浅色底框',
                'solid_box': '实心底框',
            },
        }

        subtitle_style_preset = st.selectbox(
            t['subtitle_style_preset'],
            options=subtitle_preset_values,
            index=subtitle_preset_values.index(data.get('subtitle_style_preset', 'default'))
            if data.get('subtitle_style_preset', 'default') in subtitle_preset_values else 0,
            format_func=lambda value: subtitle_preset_labels[current_lang][value],
            key=f"subtitle_style_preset_{st.session_state.reset_counter}"
        )
        subtitle_style_font_size = st.selectbox(
            t['subtitle_style_font_size'],
            options=subtitle_size_values,
            index=subtitle_size_values.index(data.get('subtitle_style_font_size', 'medium'))
            if data.get('subtitle_style_font_size', 'medium') in subtitle_size_values else 1,
            format_func=lambda value: subtitle_size_labels[current_lang][value],
            key=f"subtitle_style_font_size_{st.session_state.reset_counter}"
        )
        subtitle_style_vertical_position = st.selectbox(
            t['subtitle_style_vertical_position'],
            options=subtitle_position_values,
            index=subtitle_position_values.index(data.get('subtitle_style_vertical_position', 'bottom'))
            if data.get('subtitle_style_vertical_position', 'bottom') in subtitle_position_values else 0,
            format_func=lambda value: subtitle_position_labels[current_lang][value],
            key=f"subtitle_style_vertical_position_{st.session_state.reset_counter}"
        )
        subtitle_style_background_style = st.selectbox(
            t['subtitle_style_background_style'],
            options=subtitle_background_values,
            index=subtitle_background_values.index(data.get('subtitle_style_background_style', 'none'))
            if data.get('subtitle_style_background_style', 'none') in subtitle_background_values else 0,
            format_func=lambda value: subtitle_background_labels[current_lang][value],
            key=f"subtitle_style_background_style_{st.session_state.reset_counter}"
        )

        data['subtitle_style_preset'] = subtitle_style_preset
        data['subtitle_style_font_size'] = subtitle_style_font_size
        data['subtitle_style_vertical_position'] = subtitle_style_vertical_position
        data['subtitle_style_background_style'] = subtitle_style_background_style

        st.caption(t['subtitle_preview_help'])
        preview_bytes = render_subtitle_style_preview(
            subtitle_style_preset,
            subtitle_style_font_size,
            subtitle_style_vertical_position,
            subtitle_style_background_style,
            subtitle_translation,
            current_lang,
        )
        if preview_bytes:
            st.image(preview_bytes, caption=t['subtitle_preview'], width='stretch')
        else:
            st.warning(t['subtitle_preview_failed'])
    else:
        subtitle_translation = None
        data['subtitle_translation'] = None
        data['subtitle_style_preset'] = 'default'
        data['subtitle_style_font_size'] = 'medium'
        data['subtitle_style_vertical_position'] = 'bottom'
        data['subtitle_style_background_style'] = 'none'

    with st.expander(t['advanced_options']):
        add_titles = st.checkbox(
            t['add_titles'],
            value=data['add_titles'],
            help=t['add_titles_help'],
            key=f"add_titles_{st.session_state.reset_counter}"
        )
        data['add_titles'] = add_titles

        use_background = st.checkbox(
            t['use_background'],
            value=data['use_background'],
            help=t['use_background_help'],
            key=f"use_background_{st.session_state.reset_counter}"
        )
        data['use_background'] = use_background

        if use_background:
            st.info(t['background_info_notice'])

        force_whisper = st.checkbox(
            t['force_whisper'],
            value=data['force_whisper'],
            help=t['force_whisper_help'],
            key=f"force_whisper_{st.session_state.reset_counter}"
        )
        data['force_whisper'] = force_whisper

        asr_backend_options = ["whisper", "llm"]
        asr_backend = st.selectbox(
            t['asr_backend'],
            options=asr_backend_options,
            index=asr_backend_options.index(data.get('asr_backend', 'whisper')),
            help=t['asr_backend_help'],
            key=f"asr_backend_{st.session_state.reset_counter}",
        )
        data['asr_backend'] = asr_backend

        long_video_acceleration = st.checkbox(
            t['long_video_acceleration'],
            value=data.get('long_video_acceleration', False),
            help=t['long_video_acceleration_help'],
            key=f"long_video_acceleration_{st.session_state.reset_counter}"
        )
        data['long_video_acceleration'] = long_video_acceleration

        use_custom_prompt = st.checkbox(
            t['override_analysis_prompt'],
            value=data.get('use_custom_prompt', False),
            help=t['override_analysis_prompt_help'],
            key=f"use_custom_prompt_{st.session_state.reset_counter}"
        )
        data['use_custom_prompt'] = use_custom_prompt

        # Initialize custom_prompt_text if not present
        if 'custom_prompt_text' not in data:
            data['custom_prompt_text'] = ""

        # Speaker Identification (Preview)
        if not WHISPERX_AVAILABLE:
            st.info(t['speaker_references_unavailable'])
            speaker_references_dir = ""
        else:
            speaker_references_dir = st.text_input(
                t['speaker_references'],
                value=data.get('speaker_references_dir', ''),
                help=t['speaker_references_help'],
                placeholder="references/",
                key=f"speaker_references_dir_{st.session_state.reset_counter}",
            )
            if speaker_references_dir:
                if not Path(speaker_references_dir).is_dir():
                    st.caption(t['speaker_references_dir_not_found'])
                elif not os.getenv('HUGGINGFACE_TOKEN'):
                    st.caption(t['speaker_references_token_warning'])
        data['speaker_references_dir'] = speaker_references_dir

        st.caption(t['advanced_config_notice'])

    # Browser runtime state lives in session state; do not persist peer changes globally.
    
    # ============================================================================
    # PROCESS VIDEO BUTTON (in sidebar)
    # ============================================================================
    st.divider()
    
    # Get API key from input or environment
    resolved_api_key = api_key or os.getenv(api_key_env_var)
    requires_api_key = llm_provider != "custom_openai"
    
    if st.session_state[PREFERENCES_HYDRATED_FLAG]:
        preferences_payload = build_preferences_payload(data)
        # Inject oc_session into preferences so it persists across restarts
        preferences_payload.setdefault("prefs", {})["oc_session"] = current_owner_session_id
        default_preferences_payload = build_preferences_payload(reset_browser_state(DEFAULT_DATA))
        if st.session_state['suspend_preference_writeback']:
            if preferences_payload != default_preferences_payload:
                st.session_state['suspend_preference_writeback'] = False
        if not st.session_state['suspend_preference_writeback']:
            serialized_preferences_payload = serialize_preferences_payload(preferences_payload)
            if serialized_preferences_payload != st.session_state['remembered_preferences_payload']:
                _render_browser_preferences_writer(preferences_payload)
                st.session_state['remembered_preferences_payload'] = serialized_preferences_payload

    # Check if we can process (allow concurrent jobs)
    source_ready = bool(video_source) if input_type != INPUT_TYPE_UPLOAD else uploaded_file is not None
    can_process = bool(
        source_ready
        and resolved_llm_model
        and resolved_llm_base_url
        and (resolved_api_key or not requires_api_key)
    )
    
    # Process Video and Reset Form buttons on same row
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        process_clicked = st.button(
            t['process_video'],
            disabled=not can_process,
            type="primary",
            use_container_width=True
        )
    
    with btn_col2:
        reset_clicked = st.button(
            t['reset_form'],
            use_container_width=True
        )
    
    # Handle reset button
    if reset_clicked:
        # Reset current form state and forget remembered browser preferences.
        st.session_state.browser_data = reset_browser_state(DEFAULT_DATA)
        st.session_state[PREFERENCES_HYDRATED_FLAG] = True
        st.session_state['remembered_preferences_payload'] = None
        st.session_state['suspend_preference_writeback'] = True
        _render_browser_preferences_clearer()
        # Increment reset counter to force widget recreation
        st.session_state.reset_counter += 1
        # Force a rerun
        st.rerun()

# Main content area

# ============================================================================
# WORKER FUNCTION FOR BACKGROUND PROCESSING
# ============================================================================
def process_video_worker(job, progress_callback):
    """
    Worker function that processes video for a job
    This runs in a background thread managed by JobManager
    """
    options = job.options
    
    orchestrator = VideoOrchestrator(
        output_dir=options['output_dir'],
        max_duration_minutes=options['max_duration_minutes'],
        whisper_model=options['whisper_model'],
        browser=options.get('browser'),
        cookies=options.get('cookies_file') or None,
        api_key=options['api_key'],
        llm_provider=options['llm_provider'],
        llm_model=options.get('llm_model'),
        llm_base_url=options.get('llm_base_url'),
        skip_analysis=False,
        generate_clips=options['generate_clips'],
        add_titles=options['add_titles'],
        title_style=options['title_style'],
        use_background=options['use_background'],
        generate_cover=options['generate_cover'],
        language=options['language'],
        debug=False,
        custom_prompt_file=options.get('custom_prompt_file'),
        max_clips=options['max_clips'],
        clip_length_preset=options.get('clip_length_preset', DEFAULT_CLIP_LENGTH_PRESET),
        enable_diarization=bool(options.get('speaker_references_dir')),
        speaker_references_dir=options.get('speaker_references_dir'),
        burn_subtitles=options.get('burn_subtitles', False),
        subtitle_translation=options.get('subtitle_translation') or None,
        subtitle_style_preset=options.get('subtitle_style_preset', 'default'),
        subtitle_style_font_size=options.get('subtitle_style_font_size', 'medium'),
        subtitle_style_vertical_position=options.get('subtitle_style_vertical_position', 'bottom'),
        subtitle_style_bilingual_layout='auto',
        subtitle_style_background_style=options.get('subtitle_style_background_style', 'none'),
        mode=options.get('mode', 'engaging_moments'),
        user_intent=options.get('user_intent') or None,
        agentic_analysis=options.get('agentic_analysis', False),
        normalize_boundaries=options.get('normalize_boundaries', True),
        long_video_acceleration=options.get('long_video_acceleration', False),
        visual_verification=options.get('visual_verification', False),
    )
    
    result = asyncio.run(orchestrator.process_video(
        job.video_source,
        force_whisper=options['force_whisper'],
        asr_backend=options.get('asr_backend', 'whisper'),
        skip_download=False,
        progress_callback=progress_callback,
    ))

    if not result.success:
        raise RuntimeError(getattr(result, 'error_message', None) or "Processing failed")
    
    # Convert result to dict for JSON serialization
    return {
        'success': result.success,
        'error_message': getattr(result, 'error_message', None),
        'processing_time': getattr(result, 'processing_time', None),
        'video_info': getattr(result, 'video_info', None),
        'transcript_source': getattr(result, 'transcript_source', None),
        'engaging_moments_analysis': getattr(result, 'engaging_moments_analysis', None),
        'clip_generation': getattr(result, 'clip_generation', None),
        'post_processing': getattr(result, 'post_processing', None),
        'cover_generation': getattr(result, 'cover_generation', None),
        'editor_project': getattr(result, 'editor_project', None),
        'video_root_dir': getattr(result, 'video_root_dir', None),
    }


# Define retry handler (needs process_video_worker to be defined first)
def handle_retry(job_id):
    """Handle retry button click for a job"""
    new_job_id = job_manager.retry_job(job_id)
    if new_job_id:
        # Start the new job
        job_manager.start_job(new_job_id, process_video_worker)
        
        # Auto-track if no jobs are being tracked
        if not st.session_state.processing_job_ids:
            st.session_state.processing_job_ids = [new_job_id]
            st.session_state.processing = True
        
        st.session_state.retry_success = f"{new_job_id[:8]}..."
        st.rerun()
    else:
        st.session_state.retry_error = True

@st.fragment(run_every="2s")
def render_runtime_dashboard():
    uploads_root = Path(st.session_state.uploads_root)

    completed_jobs = []
    for job_id in st.session_state.processing_job_ids[:]:
        job = job_manager.get_job(job_id)
        if job and job.status.value in ['completed', 'failed', 'cancelled']:
            completed_jobs.append(job)
            st.session_state.processing_job_ids.remove(job_id)

    for job in completed_jobs:
        if job.status.value == 'completed':
            st.success(f"✅ Job completed: {job.video_source[:50]}...")
            if not _is_editor_rerender_job(job):
                data['processing_result'] = job.result
        elif job.status.value == 'failed':
            st.error(f"❌ Job failed: {job.video_source[:50]}... - {job.error}")
        elif job.status.value == 'cancelled':
            st.warning(f"⏹️ Job cancelled: {job.video_source[:50]}...")

    st.session_state.processing = len(st.session_state.processing_job_ids) > 0

    owner_uploads = list_uploads_for_owner(uploads_root, current_owner_session_id)

    st.header("📁 My Uploaded Files")
    if owner_uploads:
        for upload in owner_uploads:
            upload_cols = st.columns([4, 2, 1])
            with upload_cols[0]:
                st.write(f"**{upload['original_filename']}**")
                st.caption(Path(upload['staged_path']).name)
            with upload_cols[1]:
                st.caption(f"Created: {upload.get('created_at', 'unknown')}")
                if not upload.get('exists', True):
                    st.warning('Missing on disk')
            with upload_cols[2]:
                delete_disabled = job_manager.has_active_upload_reference(upload['upload_id'])
                if st.button('🗑️ Delete Upload', key=f"delete_upload_{upload['upload_id']}", use_container_width=True, disabled=delete_disabled):
                    job_manager.mark_upload_deleted(upload['upload_id'])
                    delete_upload_record(upload)
                    st.rerun()
                if delete_disabled:
                    st.caption('In use by an active job')
    else:
        st.info('No uploaded files for this browser session yet.')

    st.divider()
    st.header("📋 Your Jobs")

    jobs = job_manager.list_jobs(limit=20, owner_session_id=current_owner_session_id)

    if jobs:
        stats = job_manager.get_stats(owner_session_id=current_owner_session_id)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", stats['total'])
        col2.metric("Processing", stats['processing'])
        col3.metric("Completed", stats['completed'])
        col4.metric("Failed", stats['failed'])

        st.divider()

        for job in jobs:
            is_editor_rerender = _is_editor_rerender_job(job)
            status_emoji = {
                'pending': '⏳',
                'processing': '🔄',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '⏹️'
            }.get(job.status.value, '❓')

            display_source = job.video_source if len(job.video_source) <= 60 else job.video_source[:57] + '...'

            with st.expander(f"{status_emoji} {job.status.value.upper()} - {display_source}", expanded=(job.status.value == 'processing')):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**Job ID:** `{job.id[:8]}...`")
                    st.write(f"**Created:** {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    duration_placeholder = st.empty()
                    if job.status.value in ['completed', 'failed', 'cancelled']:
                        if job.completed_at and job.started_at:
                            duration = (job.completed_at - job.started_at).total_seconds()
                            duration_placeholder.write(f"**Duration:** {duration:.1f}s")
                    else:
                        duration_placeholder.empty()

                with col2:
                    if job.status.value == 'processing':
                        st.progress(job.progress / 100)
                        st.caption(f"{job.current_step}")
                    elif job.status.value == 'completed':
                        st.success("Editor rerender completed!" if is_editor_rerender else "Processing completed!")
                        if is_editor_rerender:
                            operation = (job.options or {}).get('operation')
                            clip_id = (job.options or {}).get('clip_id')
                            if operation:
                                st.write(f"**Operation:** {operation}")
                            if clip_id:
                                st.caption(f"Clip: {clip_id[:8]}...")
                        elif job.result and job.result.get('processing_time'):
                            st.write(f"**Time:** {job.result['processing_time']:.1f}s")
                    elif job.status.value == 'failed':
                        st.error(f"Error: {job.error}")
                    elif job.status.value == 'cancelled':
                        st.warning("Job was cancelled")

                with col3:
                    button_placeholder = st.empty()

                    if job.status.value == 'completed':
                        with button_placeholder.container():
                            if is_editor_rerender:
                                if st.button("🛠️ Open in Editor", key=f"open_editor_job_{job.id}", use_container_width=True):
                                    _launch_editor_for_job(job)
                            else:
                                if st.button("📊 View", key=f"view_{job.id}", use_container_width=True):
                                    data['processing_result'] = job.result
                                    st.rerun()
                            if st.button("🗑️ Delete", key=f"delete_{job.id}", use_container_width=True):
                                job_manager.delete_job(job.id)
                                st.rerun()
                    elif job.status.value == 'processing':
                        with button_placeholder.container():
                            is_tracked = job.id in st.session_state.processing_job_ids

                            if is_tracked:
                                st.button("✓ Watching", key=f"watching_{job.id}", use_container_width=True, disabled=True)
                            else:
                                if st.button("👁️ Watch Progress", key=f"watch_{job.id}", use_container_width=True):
                                    st.session_state.processing_job_ids = [job.id]
                                    st.session_state.processing = True
                                    st.rerun()

                            if st.button("⏹️ Cancel", key=f"cancel_{job.id}", use_container_width=True):
                                job_manager.cancel_job(job.id)
                                st.rerun()
                    elif job.status.value in ['failed', 'cancelled', 'pending']:
                        with button_placeholder.container():
                            if job.status.value in ['failed', 'cancelled']:
                                source_deleted = bool((job.options or {}).get('source_deleted'))
                                if st.button("🔄 Retry", key=f"retry_{job.id}", use_container_width=True, on_click=handle_retry, args=(job.id,), disabled=source_deleted):
                                    pass
                                if source_deleted:
                                    st.caption('Retry unavailable: source upload was deleted.')

                            if st.button("🗑️ Delete", key=f"delete_{job.id}", use_container_width=True):
                                job_manager.delete_job(job.id)
                                st.rerun()
    else:
        st.info("No jobs yet. Process a video below to get started!")

    st.divider()

    if completed_jobs:
        st.rerun()


render_runtime_dashboard()

# ============================================================================
# CUSTOM PROMPT EDITOR (if enabled)
# ============================================================================
# Custom prompt editor (shown only if use_custom_prompt is checked)
custom_prompt_file = data.get('custom_prompt_file')
if use_custom_prompt:
    st.subheader(t['custom_prompt_editor'])
    st.info(t['custom_prompt_info'])
    
    # Load default prompt if custom prompt text is empty
    if not data.get('custom_prompt_text'):
        default_prompt_path = Path("./prompts/engaging_moments_part_requirement.md")
        if default_prompt_path.exists():
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                data['custom_prompt_text'] = f.read()
    
    # Text area for custom prompt
    custom_prompt_text = st.text_area(
        t['custom_highlight_prompt'],
        value=data['custom_prompt_text'],
        height=500,
        help=t['custom_prompt_help'],
        key=f"custom_prompt_text_{st.session_state.reset_counter}"
    )
    data['custom_prompt_text'] = custom_prompt_text
    
    # Save button for custom prompt
    if st.button("💾 Save Prompt", key=f"save_custom_prompt_{st.session_state.reset_counter}"):
        if custom_prompt_text:
            try:
                # Create temp directory if it doesn't exist
                temp_dir = Path("./temp_prompts")
                temp_dir.mkdir(exist_ok=True)
                
                # Generate unique filename with timestamp
                custom_prompt_file = str(temp_dir / f"custom_highlight_prompt_{int(time.time())}.md")
                
                # Write custom prompt to file
                with open(custom_prompt_file, "w", encoding='utf-8') as f:
                    f.write(custom_prompt_text)
                
                # Save file path to data
                data['custom_prompt_file'] = custom_prompt_file
                
                # Show success message
                st.success(f"✅ {t['custom_prompt_save_success']}")
                st.caption(f"Saved to: {custom_prompt_file}")
            except Exception as e:
                st.error(f"❌ {t['custom_prompt_save_error']} {str(e)}")
        else:
            st.warning("⚠️ Please enter a highlight analysis prompt before saving.")
    
    # Show current saved prompt file if exists
    if custom_prompt_file and Path(custom_prompt_file).exists():
        st.info(f"{t['current_saved_prompt']} {Path(custom_prompt_file).name}")
    
    st.divider()

# ============================================================================
# BUTTON CLICK HANDLERS
# ============================================================================

# Show retry success/error messages
if getattr(st.session_state, 'retry_success', None):
    st.success(f"✅ Job retried! New ID: `{st.session_state.retry_success}`")
    del st.session_state.retry_success
    time.sleep(1)
    st.rerun()

if getattr(st.session_state, 'retry_error', False):
    st.error("Failed to retry job")
    del st.session_state.retry_error

# --- Handle Start ---
if process_clicked:
    source_ready = bool(video_source) if input_type != INPUT_TYPE_UPLOAD else uploaded_file is not None
    if not source_ready:
        if input_type == INPUT_TYPE_UPLOAD:
            st.error('Please choose a video file to upload')
        elif input_type == INPUT_TYPE_SERVER_PATH:
            st.error('Please provide a server file path')
        else:
            st.error('Please provide a video URL')
    elif not resolved_llm_model:
        st.error("Please provide an LLM model name or configure the provider default model")
    elif not resolved_llm_base_url:
        st.error("Please provide an LLM base URL or configure the provider default base URL")
    elif requires_api_key and not resolved_api_key:
        st.error(f"Please provide {llm_provider.upper()} API key or set the {api_key_env_var} environment variable")
    else:
        source_kind = SOURCE_KIND_URL
        upload_metadata = None
        job_source = video_source
        if input_type == INPUT_TYPE_UPLOAD:
            upload_metadata = stage_uploaded_file(uploaded_file, uploads_root, current_owner_session_id)
            job_source = upload_metadata['staged_path']
            source_kind = SOURCE_KIND_UPLOADED_FILE
        elif input_type == INPUT_TYPE_SERVER_PATH:
            source_kind = SOURCE_KIND_SERVER_PATH

        # Create job options
        job_options = {
            'output_dir': output_dir,
            'max_duration_minutes': MAX_DURATION_MINUTES,
            'whisper_model': WHISPER_MODEL,
            'browser': cookie_browser if cookie_mode == 'browser' else None,
            'api_key': resolved_api_key or None,
            'llm_provider': llm_provider,
            'llm_model': provider_settings.get('model') or None,
            'llm_base_url': provider_settings.get('base_url') or None,
            'generate_clips': generate_clips,
            'add_titles': add_titles,
            'title_style': title_style,
            'use_background': use_background,
            'generate_cover': generate_cover,
            'language': language,
            'custom_prompt_file': custom_prompt_file,
            'max_clips': max_clips,
            'clip_length_preset': clip_length_preset,
            'force_whisper': force_whisper,
            'asr_backend': asr_backend,
            'long_video_acceleration': long_video_acceleration,
            'cookie_mode': cookie_mode,
            'cookies_file': (cookies_file or None) if cookie_mode == 'file' else None,
            'speaker_references_dir': speaker_references_dir or None,
            'burn_subtitles': burn_subtitles,
            'subtitle_translation': subtitle_translation or None,
            'subtitle_style_preset': data.get('subtitle_style_preset', 'default'),
            'subtitle_style_font_size': data.get('subtitle_style_font_size', 'medium'),
            'subtitle_style_vertical_position': data.get('subtitle_style_vertical_position', 'bottom'),
            'subtitle_style_background_style': data.get('subtitle_style_background_style', 'none'),
            'user_intent': user_intent or None,
            'agentic_analysis': agentic_analysis,
            'owner_session_id': current_owner_session_id,
            'source_kind': source_kind,
            'upload_id': upload_metadata['upload_id'] if upload_metadata else None,
            'source_deleted': False,
        }

        # Check if this is a Bilibili multi-part video
        created_job_ids = []
        if source_kind == SOURCE_KIND_URL and is_bilibili_url(job_source):
            with st.spinner("Checking for multi-part video..."):
                parts = asyncio.run(get_bilibili_multi_parts(
                    job_source,
                    browser=cookie_browser if cookie_mode == 'browser' else None,
                    cookies_file=(cookies_file or None) if cookie_mode == 'file' else None
                ))

            if parts and len(parts) > 1:
                # Multi-part video detected, create a job for each part
                st.info(f"📺 Detected multi-part video with {len(parts)} parts. Creating jobs for all parts...")

                for part in parts:
                    part_url = part['url']
                    part_options = job_options.copy()
                    # Append part info to output dir to keep them separate
                    part_options['output_dir'] = os.path.join(output_dir, f"P{part['index']}_{FileStringUtils.sanitize_filename(part['title'])[:30]}")

                    job_id = job_manager.create_job(part_url, part_options)
                    job_manager.start_job(job_id, process_video_worker)
                    created_job_ids.append(job_id)

                st.success(f"✅ Created {len(created_job_ids)} jobs for all parts!")
            else:
                # Single video, create one job
                job_id = job_manager.create_job(job_source, job_options)
                job_manager.start_job(job_id, process_video_worker)
                created_job_ids.append(job_id)
                st.success(f"✅ Job started! ID: `{job_id[:8]}...`")
        else:
            # Not Bilibili, create single job
            job_id = job_manager.create_job(job_source, job_options)
            job_manager.start_job(job_id, process_video_worker)
            created_job_ids.append(job_id)
            st.success(f"✅ Job started! ID: `{job_id[:8]}...`")
        
        # Auto-track first job if no jobs are currently being tracked
        if created_job_ids and not st.session_state.processing_job_ids:
            st.session_state.processing_job_ids = [created_job_ids[0]]
            st.session_state.processing = True
        
        # Show different message based on tracking state
        if len(created_job_ids) > 1:
            st.info(f"💡 {len(created_job_ids)} jobs are running in background. Click 'Watch Progress' in job cards to track them.")
        elif created_job_ids and created_job_ids[0] in st.session_state.processing_job_ids:
            st.info("💡 This job is being tracked. You can close this page and come back later.")
        elif created_job_ids:
            st.info("💡 Job is running in background. Click 'Watch Progress' in the job card to track it.")
        
        time.sleep(1)
        st.rerun()

# --- Helper to save and display final results ---
def _finalize_results(result):
    # Convert result object to dict if needed
    if not isinstance(result, dict):
        result = {
            'success': result.success,
            'error_message': getattr(result, 'error_message', None),
            'processing_time': getattr(result, 'processing_time', None),
            'video_info': getattr(result, 'video_info', None),
            'transcript_source': getattr(result, 'transcript_source', None),
            'engaging_moments_analysis': getattr(result, 'engaging_moments_analysis', None),
            'clip_generation': getattr(result, 'clip_generation', None),
            'post_processing': getattr(result, 'post_processing', None),
            'cover_generation': getattr(result, 'cover_generation', None),
            'editor_project': getattr(result, 'editor_project', None),
        }
    
    data['processing_result'] = result

# Display saved results if they exist and we didn't just process a video
if data['processing_result'] and not just_processed:
    header_col, action_col = st.columns([5, 1])
    with header_col:
        st.header("📊 Saved Results")
    with action_col:
        st.write("")
        st.write("")
        clear_saved_results = st.button("Clear Saved Results", key="clear_saved_results_header")
    if 'success' not in data['processing_result']:
        st.info("Saved results are only available for full processing jobs. Open editor rerender results from the job card instead.")
        if clear_saved_results:
            data['processing_result'] = None
            st.rerun()
    else:
    # Convert dictionary back to object-like structure
        class ResultObject:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)

        result_obj = ResultObject(data['processing_result'])
        display_results(result_obj)
        
        # Add a button to clear saved results
        if clear_saved_results:
            data['processing_result'] = None
            st.rerun()

# Footer
st.markdown("""
---
**Made with ❤️ for content creators**
""")

# GitHub buttons row
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    st.markdown("""
    <a href="https://github.com/linzzzzzz/openclip/issues" target="_blank" style="text-decoration: none;">
        <button style="
            background-color: transparent;
            color: #58a6ff;
            border: none;
            outline: none;
            box-shadow: none;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
        ">
            <span>🐛</span> Report Bug
        </button>
    </a>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <a href="https://github.com/linzzzzzz/openclip" target="_blank" style="text-decoration: none;">
        <button style="
            background-color: transparent;
            color: #f0883e;
            border: none;
            outline: none;
            box-shadow: none;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            white-space: nowrap;
        ">
            <span>⭐</span> Star on GitHub
        </button>
    </a>
    """, unsafe_allow_html=True)
