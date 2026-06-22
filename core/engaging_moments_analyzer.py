#!/usr/bin/env python3
"""
Engaging Moments Analyzer
Identifies engaging moments from video transcripts using LLM APIs
"""

import json
import logging
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from core.llm.qwen_api_client import QwenAPIClient, QwenMessage
from core.config import LLM_CONFIG, MAX_CLIPS, API_KEY_ENV_VARS, SUPPORTED_LLM_PROVIDERS, EVENT_CLUSTER_GAP_SECONDS, GROUNDING_WINDOW_SECONDS, MAX_CONCURRENT_GROUNDINGS, DISALLOWED_GOAL_WINDOW_SECONDS
from core.clip_duration import (
    build_clip_duration_prompt_section,
    get_clip_duration_preference,
)

logger = logging.getLogger(__name__)


class EngagingMomentsAnalyzer:
    """Analyzes video transcripts to identify engaging moments using LLM APIs"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "qwen",
        use_background: bool = False,
        language: str = "zh",
        debug: bool = False,
        custom_prompt_file: Optional[str] = None,
        max_clips: int = MAX_CLIPS,
        user_intent: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        clip_length_preset: Optional[str] = None,
        video_title: Optional[str] = None,
    ):
        """
        Initialize the analyzer

        Args:
            api_key: API key for the selected provider (optional, can use env var)
            provider: LLM provider to use ("qwen", "openrouter", "glm", "minimax", or "custom_openai")
            use_background: Whether to include background information in prompts
            language: Language for output ("zh" for Chinese, "en" for English)
            debug: Enable debug mode to export full prompts sent to LLM
            custom_prompt_file: Path to custom prompt file (optional)
            user_intent: Optional free-text description of what the user is looking for
        """
        self.custom_prompt_file = custom_prompt_file
        self.max_clips = max_clips
        self.user_intent = user_intent.strip() if user_intent else None
        self.clip_duration_preference = get_clip_duration_preference(clip_length_preset)
        self.provider = provider.lower()
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self.use_background = use_background
        self.background_content = None
        self.language = language
        self.debug = debug
        self.model = model.strip() if model else None
        self.base_url = base_url.strip() if base_url else None
        self.video_title = video_title.strip() if video_title else None
        self.match_teams = self._extract_teams_from_title(self.video_title) if self.video_title else None

        if self.provider == "custom_openai" and not (
            self.model or LLM_CONFIG["custom_openai"]["default_model"]
        ):
            raise ValueError(
                "custom_openai requires llm_model. Set CUSTOM_OPENAI_MODEL or provide llm_model."
            )
        
        # Initialize the appropriate LLM client
        if self.provider == "qwen":
            from core.llm.qwen_api_client import QwenAPIClient
            self.llm_client = QwenAPIClient(api_key, base_url=self.base_url)
        elif self.provider == "openrouter":
            from core.llm.openrouter_api_client import OpenRouterAPIClient
            self.llm_client = OpenRouterAPIClient(api_key, base_url=self.base_url)
        elif self.provider == "glm":
            from core.llm.glm_api_client import GLMAPIClient
            self.llm_client = GLMAPIClient(api_key, base_url=self.base_url)
        elif self.provider == "minimax":
            from core.llm.minimax_api_client import MiniMaxAPIClient
            self.llm_client = MiniMaxAPIClient(api_key, base_url=self.base_url)
        elif self.provider == "doubao":
            from core.llm.custom_openai_api_client import CustomOpenAIAPIClient
            doubao_cfg = LLM_CONFIG["doubao"]
            self.llm_client = CustomOpenAIAPIClient(
                api_key or os.getenv(API_KEY_ENV_VARS["doubao"]),
                base_url=self.base_url or doubao_cfg["base_url"],
                model=self.model or doubao_cfg["default_model"],
            )
        elif self.provider == "custom_openai":
            from core.llm.custom_openai_api_client import CustomOpenAIAPIClient
            self.llm_client = CustomOpenAIAPIClient(api_key, base_url=self.base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {', '.join(SUPPORTED_LLM_PROVIDERS)}")
        
        # Load background information if enabled
        if self.use_background:
            self._load_background_info()
    
    def _load_background_info(self):
        """Load background information from prompts/background/background.md"""
        try:
            background_path = self.prompts_dir / "background" / "background.md"
            if background_path.exists():
                with open(background_path, 'r', encoding='utf-8') as f:
                    self.background_content = f.read().strip()
                logger.info("📚 Background information loaded")
            else:
                logger.warning(f"Background file not found: {background_path}")
                self.use_background = False
        except Exception as e:
            logger.error(f"Error loading background information: {e}")
            self.use_background = False

    @staticmethod
    def _extract_teams_from_title(title: str) -> Optional[Dict[str, str]]:
        """Extract team names from video title like '德国2_1科特迪瓦' or '乌拉圭2-2佛得角'."""
        if not title:
            return None
        # Remove emoji
        cleaned = re.sub(r'[\U0001f300-\U0001f9ff\u200d\ufe0f]+', '', title).strip()
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'(全场回放|全场集锦|精彩回放|比赛回放|球迷之家回放|开门红)', '', cleaned).strip()
        # Match: TeamA <score> TeamB
        m = re.match(r'^(.+?)(\d+)[_\-:：](\d+)(.+?)$', cleaned)
        if m:
            home = m.group(1).strip()
            away = m.group(4).strip()
            # Further clean: take only the last CJK/letter word as team name
            # e.g. "xxx英格兰" -> "英格兰"
            home_m = re.search(r'([\u4e00-\u9fff\w]+)$', home)
            if home_m:
                home = home_m.group(1)
            return {'home': home, 'away': away}
        # Match: "England 4-2 Croatia" style
        m = re.match(r'^(.+?)\s+(\d+)\s*[-_:：]\s*(\d+)\s+(.+?)$', cleaned)
        if m:
            return {'home': m.group(1).strip(), 'away': m.group(4).strip()}
        return None

    def _build_team_context_prompt(self) -> str:
        """Build prompt section for team identification in tags."""
        if not self.match_teams:
            return ""
        home = self.match_teams['home']
        away = self.match_teams['away']
        return (
            f"\n\n## 球队信息\n\n"
            f"本场比赛双方球队：**{home}** vs **{away}**\n\n"
            f"**重要**：对于进球（goal）、射门（shot）、扑救（save）等事件，"
            f"请在 tags 中加入对应球队名称标签。规则：\n"
            f"- 进球/射门：加入射门方球队名（如 \"{home}\" 或 \"{away}\"）\n"
            f"- 扑救：加入扑救方门将所属球队名\n"
            f"- 犯规/红黄牌：加入犯规方球队名\n"
            f"- 如果无法从转录判断是哪支球队，则不加球队标签\n"
        )

    def set_video_title(self, title: str):
        """Set video title and re-extract team info (call after download when title becomes known)."""
        self.video_title = title.strip() if title else None
        self.match_teams = self._extract_teams_from_title(self.video_title) if self.video_title else None
        if self.match_teams:
            logger.info(f"⚽ Detected teams from title: {self.match_teams['home']} vs {self.match_teams['away']}")
    
    def _export_debug_prompt(self, prompt_content: str, prompt_type: str, part_name: Optional[str] = None):
        """
        Export full prompt content for debugging
        
        Args:
            prompt_content: The full prompt content to export
            prompt_type: Type of prompt ("part_analysis" or "aggregation")
            part_name: Name of the video part (for part analysis prompts)
        """
        if not self.debug:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create debug directory
            debug_dir = Path("debug_prompts")
            debug_dir.mkdir(exist_ok=True)
            
            # Generate filename
            if part_name:
                filename = f"{prompt_type}_{part_name}_{timestamp}.txt"
            else:
                filename = f"{prompt_type}_{timestamp}.txt"
            
            # Export prompt
            export_path = debug_dir / filename
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(f"=== DEBUG PROMPT - {prompt_type.upper()} ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Language: {self.language}\n")
                if part_name:
                    f.write(f"Video Part: {part_name}\n")
                f.write(f"Prompt Length: {len(prompt_content)} characters\n")
                f.write("=" * 60 + "\n\n")
                f.write(prompt_content)
            
            logger.info(f"🐛 Debug prompt exported: {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting debug prompt: {e}")
    
    def load_prompt_template(self, prompt_name: str) -> str:
        """
        Load prompt template from prompts directory
        
        Args:
            prompt_name: Name of the prompt file (without .md extension)
            
        Returns:
            Content of the prompt file
        """
        # Use custom prompt file if specified and this is the part requirement prompt
        if prompt_name == "engaging_moments_part_requirement" and self.custom_prompt_file:
            custom_prompt_path = Path(self.custom_prompt_file)
            if custom_prompt_path.exists():
                logger.info(f"📝 Using custom prompt file: {custom_prompt_path}")
                with open(custom_prompt_path, 'r', encoding='utf-8') as f:
                    prompt_content = f.read().strip()
            else:
                logger.warning(f"Custom prompt file not found: {custom_prompt_path}")
                logger.info(f"Falling back to default prompt: engaging_moments_part_requirement.md")
                # Fall back to default prompt
                base_prompt_path = self.prompts_dir / f"{prompt_name}.md"
                if not base_prompt_path.exists():
                    raise FileNotFoundError(f"Base prompt file not found: {base_prompt_path}")
                with open(base_prompt_path, 'r', encoding='utf-8') as f:
                    prompt_content = f.read().strip()
        else:
            # Load base prompt template (without language suffix)
            base_prompt_path = self.prompts_dir / f"{prompt_name}.md"
            
            if not base_prompt_path.exists():
                raise FileNotFoundError(f"Base prompt file not found: {base_prompt_path}")
            
            with open(base_prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
        
        # Load and append language-specific patch
        language_patch_path = self.prompts_dir / "language_patches" / f"{self.language}.md"
        
        if language_patch_path.exists():
            with open(language_patch_path, 'r', encoding='utf-8') as f:
                language_patch = f.read().strip()
            
            # Append language patch to the base prompt
            prompt_content += "\n\n" + language_patch
            logger.info(f"🌐 Language patch loaded for: {self.language}")
        else:
            logger.warning(f"Language patch not found: {language_patch_path}")
        
        return prompt_content
    
    def parse_srt_file(self, srt_path: str) -> List[Dict[str, Any]]:
        """
        Parse SRT file and extract subtitle entries
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            List of subtitle entries with timing and text
        """
        entries = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Split by double newlines to separate entries
            blocks = content.split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Parse timing line (format: 00:00:00,000 --> 00:00:02,000)
                    timing_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                    if timing_match:
                        start_time = timing_match.group(1)
                        end_time = timing_match.group(2)
                        text = ' '.join(lines[2:])  # Join all text lines
                        
                        entries.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'text': text
                        })
        
        except Exception as e:
            logger.error(f"Error parsing SRT file {srt_path}: {e}")
            
        return entries
    
    def time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format to seconds"""
        # Format: 00:01:30,500 or 00:01:30 (without milliseconds)
        if ',' in time_str:
            time_part, ms_part = time_str.split(',')
            ms = int(ms_part)
        else:
            time_part = time_str
            ms = 0
        
        parts = time_part.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, s = map(int, parts)
        else:
            raise ValueError(f"Unexpected time format: {time_str}")
        return h * 3600 + m * 60 + s + ms / 1000
    
    def seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    def create_transcript_context(self, entries: List[Dict[str, Any]]) -> str:
        """Create a formatted transcript context for Qwen analysis"""
        transcript_lines = []
        
        for entry in entries:
            transcript_lines.append(f"[{entry['start_time']} --> {entry['end_time']}] {entry['text']}")
        
        return '\n'.join(transcript_lines)
    
    def build_part_analysis_prompt(self, srt_path: str, part_name: str) -> str:
        """
        Build the analysis prompt for a video part without calling the LLM.

        Args:
            srt_path: Path to SRT file
            part_name: Name of the video part (e.g., "part01")

        Returns:
            The complete prompt string, or empty string if no entries found
        """
        entries = self.parse_srt_file(srt_path)
        if not entries:
            return ""

        transcript_context = self.create_transcript_context(entries)
        prompt_template = self.load_prompt_template("engaging_moments_part_requirement")

        prompt_parts = []
        if self.use_background and self.background_content:
            prompt_parts.append("## Additional Background Information\n\n")
            prompt_parts.append(self.background_content)
            prompt_parts.append("\n\n")

        prompt_parts.append(prompt_template)
        prompt_parts.append("\n\n")
        prompt_parts.append(build_clip_duration_prompt_section(self.clip_duration_preference.preset))
        prompt_parts.append(self._build_team_context_prompt())
        if self.user_intent:
            prompt_parts.append(f"\n\n## User Focus\n\nThe user is specifically looking for: {self.user_intent}\nPrioritize moments related to this when selecting and ranking clips.")
        prompt_parts.append(f"\n\n## Transcript Data for {part_name}\n\n")
        prompt_parts.append(transcript_context)
        prompt_parts.append("\n\nPlease analyze this transcript and identify engaging moments following the requirements above.")

        return "".join(prompt_parts)

    def build_aggregation_prompt(self, highlights_files: List[str]) -> str:
        """
        Build the aggregation prompt from highlights files without calling the LLM.

        Args:
            highlights_files: List of paths to highlights JSON files

        Returns:
            The complete aggregation prompt string
        """
        all_moments = []
        for file_path in highlights_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for moment in data.get('engaging_moments', []):
                    moment['source_part'] = data.get('video_part', 'unknown')
                    all_moments.append(moment)
            except Exception as e:
                logger.error(f"Error loading highlights file {file_path}: {e}")

        moments_context = self._create_moments_context(all_moments)
        prompt_template = self.load_prompt_template("engaging_moments_agg_requirement")

        prompt_parts = []
        if self.use_background and self.background_content:
            prompt_parts.append("## Background Information\n\n")
            prompt_parts.append(self.background_content)
            prompt_parts.append("\n\n")

        prompt_parts.append(prompt_template.replace("{max_clips}", str(self.max_clips)))
        prompt_parts.append("\n\n")
        prompt_parts.append(build_clip_duration_prompt_section(self.clip_duration_preference.preset))
        prompt_parts.append(
            "\n\nWhen ranking final clips, use duration fit as a preference after content quality, "
            "standalone clarity, and engagement strength."
        )
        if self.user_intent:
            prompt_parts.append(f"\n\n## User Focus\n\nThe user is specifically looking for: {self.user_intent}\nPrioritize moments related to this when selecting and ranking the final clips.")
        prompt_parts.append(f"\n\n## All Engaging Moments Data\n\n")
        prompt_parts.append(moments_context)
        prompt_parts.append(f"\n\nPlease select and rank the top {self.max_clips} most engaging moments following the requirements above.")

        return "".join(prompt_parts)

    async def analyze_part_for_engaging_moments(self, srt_path: str, part_name: str) -> Dict[str, Any]:
        """Analyze a single video part for engaging moments."""
        logger.info(f"🔍 Analyzing {part_name} for engaging moments...")

        entries = self.parse_srt_file(srt_path)
        if not entries:
            logger.warning(f"No entries found in {srt_path}")
            return self._create_empty_result(part_name)

        # Check if the SRT has context markers (from split_srt_for_analysis)
        with open(srt_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        if '[TARGET:' in raw_content:
            return await self._analyze_context_marked_chunk(raw_content, entries, part_name, srt_path)

        return await self._analyze_entries_chunk(entries, part_name, srt_path)

    async def _analyze_context_marked_chunk(
        self, raw_content: str, entries: List[Dict[str, Any]], chunk_name: str, srt_path: str
    ) -> Dict[str, Any]:
        """Analyze a chunk that contains [CONTEXT]/[TARGET] markers — pass raw content to LLM."""
        prompt_template = self.load_prompt_template("engaging_moments_part_requirement")

        prompt_parts = []
        if self.use_background and self.background_content:
            prompt_parts.append("## Additional Background Information\n\n")
            prompt_parts.append(self.background_content)
            prompt_parts.append("\n\n")

        prompt_parts.append(prompt_template)
        prompt_parts.append("\n\n")
        prompt_parts.append(self._build_team_context_prompt())
        if self.user_intent:
            prompt_parts.append(f"\n\n## 用户关注点\n\n用户特别关注：{self.user_intent}\n请优先标注与此相关的时刻。")
        prompt_parts.append(f"\n\n## Transcript Data for {chunk_name}\n\n")
        prompt_parts.append(raw_content)
        prompt_parts.append("\n\nPlease analyze this transcript and identify engaging moments. "
                           "IMPORTANT: Only output moments whose timestamps fall within the [TARGET] range.")

        analysis_prompt = "".join(prompt_parts)
        self._export_debug_prompt(analysis_prompt, "part_analysis", chunk_name)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.llm_client.simple_chat(analysis_prompt, model=self.model)
            )
            try:
                result = self._extract_and_parse_json(response, chunk_name, entries)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for {chunk_name}: {e}")
                result = self._create_empty_result(chunk_name)
        except Exception as e:
            logger.error(f"Error calling LLM API for {chunk_name}: {e}")
            result = self._create_empty_result(chunk_name)

        return result

    async def _analyze_entries_chunk(self, entries: List[Dict[str, Any]], chunk_name: str, srt_path: str) -> Dict[str, Any]:
        """Analyze a single chunk of entries."""
        transcript_context = self.create_transcript_context(entries)
        prompt_template = self.load_prompt_template("engaging_moments_part_requirement")

        prompt_parts = []
        if self.use_background and self.background_content:
            prompt_parts.append("## Additional Background Information\n\n")
            prompt_parts.append(self.background_content)
            prompt_parts.append("\n\n")

        prompt_parts.append(prompt_template)
        prompt_parts.append("\n\n")
        prompt_parts.append(self._build_team_context_prompt())
        if self.user_intent:
            prompt_parts.append(f"\n\n## 用户关注点\n\n用户特别关注：{self.user_intent}\n请优先标注与此相关的时刻。")
        prompt_parts.append(f"\n\n## Transcript Data for {chunk_name}\n\n")
        prompt_parts.append(transcript_context)
        prompt_parts.append("\n\nPlease analyze this transcript and identify engaging moments following the requirements above.")

        analysis_prompt = "".join(prompt_parts)
        self._export_debug_prompt(analysis_prompt, "part_analysis", chunk_name)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.llm_client.simple_chat(analysis_prompt, model=self.model)
            )
            try:
                result = self._extract_and_parse_json(response, chunk_name, entries)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for {chunk_name}: {e}")
                result = self._create_empty_result(chunk_name)
        except Exception as e:
            logger.error(f"Error calling LLM API for {chunk_name}: {e}")
            result = self._create_empty_result(chunk_name)

        return result
    
    def _extract_and_parse_json(self, response: str, part_name: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and parse JSON from AI response with AI-powered error handling
        
        Args:
            response: Raw AI response
            part_name: Video part name
            entries: SRT entries for validation
            
        Returns:
            Parsed and validated JSON result
        """
        # First try standard JSON parsing
        try:
            # Try direct parsing first
            result = json.loads(response.strip())
            logger.debug("Successfully parsed response as direct JSON")
            return self._validate_and_clean_result(result, part_name, entries)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                logger.debug("Successfully parsed JSON from code block")
                return self._validate_and_clean_result(result, part_name, entries)
            except json.JSONDecodeError:
                pass
        
        # If standard parsing fails, use AI to fix the JSON
        logger.info("Standard JSON parsing failed, using AI to fix JSON...")
        try:
            fixed_json = self._ai_fix_json(response, part_name)
            result = json.loads(fixed_json)
            logger.debug("Successfully parsed AI-fixed JSON")
            return self._validate_and_clean_result(result, part_name, entries)
        except Exception as e:
            logger.error(f"AI JSON fixing failed: {e}")
            # Export raw and fixed responses for debugging
            self._export_failed_responses(response, part_name, fixed_json if 'fixed_json' in locals() else None, e)
            return self._create_empty_result(part_name)
    
    def _ai_fix_json(self, malformed_response: str, part_name: str) -> str:
        """
        Use AI to fix malformed JSON response
        
        Args:
            malformed_response: The malformed JSON response
            part_name: Video part name for context
            
        Returns:
            Fixed JSON string
        """
        fix_prompt = f"""
You are a JSON repair expert. I have a malformed JSON response that needs to be fixed. 

The response should follow this structure:
{{
  "video_part": "{part_name}",
  "engaging_moments": [
    {{
      "title": "七人接力鉴定假发造型，现场即兴互动引爆弹幕高潮！",
      "start_time": "00:01:30",
      "end_time": "00:02:45",
      "duration_seconds": 75,
      "summary": "Brief 1-2 sentence description of what happens in this moment.",
      "engagement_details": {{
        "engagement_level": "high"
      }},
      "why_engaging": "多人互动环节，现场气氛热烈，弹幕互动频繁，具有很强的娱乐性和观赏价值",
      "tags": ["进球", "赛事高光", "明星球员", "精彩集锦"]
    }}
  ],
  "total_moments": 1,
  "analysis_timestamp": "2024-01-01T12:00:00Z"
}}

IMPORTANT:
- start_time and end_time should be in simple format (HH:MM:SS or MM:SS), NOT SRT format with milliseconds
- Remove any "engagement_score" field if present
- Ensure "why_engaging" is shorter than 100 characters
- Use only approved tags: ["进球", "射门", "头球", "点球", "红牌", "黄牌", "扑救", "任意球", "角球", "越位", "VAR", "换人", "助攻", "反击", "庆祝", "绝杀", "逆转", "乌龙球", "无效进球", "明星球员", "战术亮点", "赛事高光", "战术解析", "球星表现", "历史纪录", "裁判争议", "数据分析", "教练布置", "阵型变化", "体能管理", "赛前分析", "赛后总结", "经典对决", "名嘴金句", "精彩集锦", "深度解析", "争议"]

Here is the malformed response:
{malformed_response}

Please fix the JSON and return ONLY the valid JSON, no explanations:
"""
        
        try:
            # Use a simpler model for JSON fixing to avoid recursion
            fixed_response = self.llm_client.simple_chat(fix_prompt, model=self.model)
            
            # Extract JSON from the fixed response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', fixed_response, re.DOTALL)
            if json_match:
                return json_match.group(1)
            
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', fixed_response, re.DOTALL)
            if json_match:
                return json_match.group()
            
            # If no JSON found, return the entire response
            return fixed_response.strip()
            
        except Exception as e:
            logger.error(f"Error in AI JSON fixing: {e}")
            raise
    
    def _clean_json_text(self, json_text: str) -> str:
        """
        Clean common JSON formatting issues
        
        Args:
            json_text: Raw JSON text
            
        Returns:
            Cleaned JSON text
        """
        # Remove leading/trailing whitespace
        json_text = json_text.strip()
        
        # Remove markdown code block markers if present
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        
        # Fix common trailing comma issues
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Fix missing commas between objects/arrays (basic fix)
        json_text = re.sub(r'}\s*{', '},{', json_text)
        json_text = re.sub(r']\s*\[', '],[', json_text)
        
        # Convert SRT timestamp format to simple format if present
        # Convert HH:MM:SS,mmm to HH:MM:SS
        json_text = re.sub(r'(\d{2}:\d{2}:\d{2}),\d{3}', r'\1', json_text)
        
        return json_text
    
    def _validate_and_clean_result(self, result: Dict[str, Any], part_name: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and clean up the analysis result"""
        
        # Ensure required fields
        if 'engaging_moments' not in result:
            result['engaging_moments'] = []
        
        result['video_part'] = part_name
        result['total_moments'] = len(result['engaging_moments'])
        result['analysis_timestamp'] = datetime.now().isoformat() + 'Z'
        
        # Handle detected_content_type field
        if 'detected_content_type' not in result:
            result['detected_content_type'] = 'unknown'
        
        # Validate each moment
        valid_moments = []
        for moment in result['engaging_moments']:
            if self._validate_moment(moment, entries):
                valid_moments.append(moment)
        
        result['engaging_moments'] = valid_moments
        result['total_moments'] = len(valid_moments)
        
        return result
    
    def _validate_moment(self, moment: Dict[str, Any], entries: List[Dict[str, Any]]) -> bool:
        """Validate a single engaging moment (anchor_time format)"""

        # Support both old (start_time/end_time) and new (anchor_time) format
        if 'anchor_time' in moment:
            try:
                self.time_to_seconds(moment['anchor_time'])
            except Exception as e:
                logger.warning(f"Invalid anchor_time: {moment.get('anchor_time')!r}, error={e}")
                return False
            if 'title' not in moment:
                logger.warning("Missing required field: title")
                return False
            if 'importance' not in moment:
                moment['importance'] = 'medium'
            if 'event_type' not in moment:
                moment['event_type'] = 'other'
            if 'summary' not in moment:
                moment['summary'] = ""
            if 'tags' not in moment:
                moment['tags'] = []
            return True

        # Legacy format with start_time/end_time
        required_fields = ['title', 'start_time', 'end_time']
        for field in required_fields:
            if field not in moment:
                logger.warning(f"Missing required field: {field}")
                return False
        
        try:
            # Validate timing
            start_time = moment['start_time']
            end_time = moment['end_time']
            start_seconds = self.time_to_seconds(start_time)
            end_seconds = self.time_to_seconds(end_time)
            duration = end_seconds - start_seconds
            
            min_seconds = self.clip_duration_preference.min_seconds
            max_seconds = self.clip_duration_preference.max_seconds
            if duration < min_seconds or duration > max_seconds:
                logger.warning(
                    "Invalid duration: "
                    f"title={moment.get('title', '<untitled>')!r}, "
                    f"start_time={start_time!r} ({start_seconds:.3f}s), "
                    f"end_time={end_time!r} ({end_seconds:.3f}s), "
                    f"duration={duration:.3f}s, "
                    f"expected_range={min_seconds}-{max_seconds}s"
                )
                return False
            
            moment['duration_seconds'] = int(duration)
            
            # Ensure other fields exist
            if 'summary' not in moment:
                moment['summary'] = ""
            if 'engagement_details' not in moment:
                moment['engagement_details'] = {"engagement_level": "medium"}
            elif 'engagement_level' not in moment['engagement_details']:
                moment['engagement_details']['engagement_level'] = "medium"
            if 'tags' not in moment:
                moment['tags'] = []
                
        except Exception as e:
            logger.warning(
                "Error validating moment timing: "
                f"title={moment.get('title', '<untitled>')!r}, "
                f"start_time={moment.get('start_time')!r}, "
                f"end_time={moment.get('end_time')!r}, "
                f"error={e}"
            )
            return False
        
        return True
    
    def _create_empty_result(self, part_name: str) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            "video_part": part_name,
            "engaging_moments": [],
            "total_moments": 0,
            "analysis_timestamp": datetime.now().isoformat() + 'Z'
        }

    def _retag_disallowed_goals(self, moments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect goal events followed by VAR/offside within window, re-tag as 无效进球."""
        var_keywords = {"越位", "VAR"}
        goal_indices = []
        var_indices = []

        for i, m in enumerate(moments):
            tags = set(m.get('tags', []))
            if "进球" in tags:
                goal_indices.append(i)
            elif tags & var_keywords and m.get('event_type') == 'var':
                var_indices.append(i)

        remove_set = set()
        for gi in goal_indices:
            goal_m = moments[gi]
            goal_time = self.time_to_seconds(
                goal_m.get('anchor_time') or goal_m.get('start_time', '00:00:00')
            )
            for vi in var_indices:
                if vi in remove_set:
                    continue
                var_m = moments[vi]
                var_time = self.time_to_seconds(
                    var_m.get('anchor_time') or var_m.get('start_time', '00:00:00')
                )
                if 0 < (var_time - goal_time) <= DISALLOWED_GOAL_WINDOW_SECONDS:
                    # Re-tag as disallowed goal
                    tags = [t for t in goal_m.get('tags', []) if t != '进球']
                    if '无效进球' not in tags:
                        tags.insert(0, '无效进球')
                    if '射门' not in tags:
                        tags.append('射门')
                    for vt in var_m.get('tags', []):
                        if vt in var_keywords and vt not in tags:
                            tags.append(vt)
                    goal_m['tags'] = tags
                    goal_m['event_type'] = 'var'
                    title = goal_m.get('title', '')
                    if '无效' not in title and '取消' not in title and '越位' not in title:
                        goal_m['title'] = title.rstrip('！!。') + '被判无效'
                    remove_set.add(vi)
                    logger.info(f"🚫 Disallowed goal: '{goal_m['title']}' (VAR at {var_m.get('anchor_time')})")
                    break

        if remove_set:
            moments = [m for i, m in enumerate(moments) if i not in remove_set]
        return moments

    async def aggregate_top_moments(self, highlights_files: List[str], output_dir: str) -> Dict[str, Any]:
        """
        Aggregate engaging moments from multiple chunks with event clustering.
        Merges moments within EVENT_CLUSTER_GAP_SECONDS into single events.
        """
        logger.info("🔄 Aggregating top engaging moments with event clustering...")

        all_moments = []
        for file_path in highlights_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                video_part = data.get('video_part', 'unknown')
                for moment in data.get('engaging_moments', []):
                    moment['_source_video_part'] = video_part
                    all_moments.append(moment)
            except Exception as e:
                logger.error(f"Error loading highlights file {file_path}: {e}")

        if not all_moments:
            logger.warning("No engaging moments found to aggregate")
            return self._create_empty_aggregation_result()

        # Sort by anchor_time (or start_time for legacy)
        def _get_time(m):
            t = m.get('anchor_time') or m.get('start_time', '00:00:00')
            return self.time_to_seconds(t)

        all_moments.sort(key=_get_time)

        # Event clustering: merge moments within gap threshold
        clusters = []
        current_cluster = [all_moments[0]]
        for m in all_moments[1:]:
            prev_time = _get_time(current_cluster[-1])
            curr_time = _get_time(m)
            if curr_time - prev_time <= EVENT_CLUSTER_GAP_SECONDS:
                current_cluster.append(m)
            else:
                clusters.append(current_cluster)
                current_cluster = [m]
        clusters.append(current_cluster)

        # Pick representative from each cluster (highest importance)
        importance_rank = {'high': 3, 'medium': 2, 'low': 1}
        deduplicated = []
        for cluster in clusters:
            best = max(cluster, key=lambda m: importance_rank.get(
                m.get('importance') or m.get('engagement_details', {}).get('engagement_level', 'low'), 0
            ))
            deduplicated.append(best)

        logger.info(f"📊 Clustered {len(all_moments)} moments → {len(deduplicated)} unique events")

        # Post-process: detect disallowed goals (进球 followed by VAR/越位 within window)
        deduplicated = self._retag_disallowed_goals(deduplicated)

        top_moments = deduplicated[:self.max_clips]
        for i, moment in enumerate(top_moments):
            moment['rank'] = i + 1
            moment['timing'] = {
                'video_part': moment.pop('_source_video_part', 'unknown'),
                'anchor_time': moment.get('anchor_time') or moment.get('start_time', '00:00:00'),
            }

        logger.info(f"✅ Aggregated {len(top_moments)} events (clustered, sorted by time)")
        return {
            "top_engaging_moments": top_moments,
            "total_moments": len(top_moments),
            "analysis_timestamp": datetime.now().isoformat() + 'Z',
            "aggregation_criteria": f"Clustered (gap={EVENT_CLUSTER_GAP_SECONDS}s), top {self.max_clips}",
            "analysis_summary": {
                "highest_engagement_themes": [],
                "total_engaging_content_time": "N/A",
                "recommendation": "Two-stage event grounding"
            },
            "honorable_mentions": []
        }

    async def ground_events(self, moments: List[Dict[str, Any]], srt_path: str) -> List[Dict[str, Any]]:
        """Fine grounding: determine precise start/end for each event using LLM."""
        if not moments:
            return []

        entries = self.parse_srt_file(srt_path)
        if not entries:
            logger.warning("No SRT entries for grounding, returning moments without grounding")
            return moments

        prompt_template = self.load_prompt_template("event_grounding_requirement")
        sem = asyncio.Semaphore(MAX_CONCURRENT_GROUNDINGS)

        async def _ground_one(moment: Dict[str, Any]) -> Dict[str, Any]:
            anchor = (moment.get('anchor_time')
                      or (moment.get('timing') or {}).get('anchor_time')
                      or moment.get('start_time', '00:00:00'))
            anchor_sec = self.time_to_seconds(anchor)
            srt_window = self._extract_srt_window(entries, anchor_sec, GROUNDING_WINDOW_SECONDS)
            if not srt_window:
                return moment

            min_sec = self.clip_duration_preference.min_seconds
            max_sec = self.clip_duration_preference.max_seconds
            prompt = prompt_template.replace("{title}", moment.get('title', ''))
            prompt = prompt.replace("{event_type}", moment.get('event_type', 'other'))
            prompt = prompt.replace("{anchor_time}", anchor)
            prompt = prompt.replace("{importance}", moment.get('importance', 'medium'))
            prompt = prompt.replace("{srt_window}", srt_window)
            window_start = self.seconds_to_time(max(0, anchor_sec - GROUNDING_WINDOW_SECONDS))
            window_end = self.seconds_to_time(anchor_sec + GROUNDING_WINDOW_SECONDS)
            prompt = prompt.replace("{window_start}", window_start)
            prompt = prompt.replace("{window_end}", window_end)
            prompt = prompt.replace("{min_duration}", str(min_sec))
            prompt = prompt.replace("{max_duration}", str(max_sec))

            async with sem:
                try:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, lambda: self.llm_client.simple_chat(prompt, model=self.model)
                    )
                    grounding = self._parse_grounding_response(response)
                    if grounding:
                        moment['start_time'] = grounding['start_time']
                        moment['end_time'] = grounding['end_time']
                        moment['duration_seconds'] = grounding['duration_seconds']
                    else:
                        logger.warning(f"Grounding parse failed for '{moment.get('title', '')[:30]}', response: {response[:200] if response else 'empty'}")
                except Exception as e:
                    logger.warning(f"Grounding failed for '{moment.get('title', '')}': {e}")
            return moment

        results = await asyncio.gather(*[_ground_one(m) for m in moments])
        # Filter out moments that failed grounding (no start_time)
        grounded = [m for m in results if 'start_time' in m and 'end_time' in m]
        logger.info(f"✅ Grounded {len(grounded)}/{len(moments)} events")
        return grounded

    def _extract_srt_window(self, entries: List[Dict[str, Any]], anchor_sec: float, window_sec: float) -> str:
        """Extract SRT entries within anchor ± window_sec."""
        start_bound = max(0, anchor_sec - window_sec)
        end_bound = anchor_sec + window_sec
        lines = []
        for entry in entries:
            try:
                entry_start = self.time_to_seconds(entry.get('start_time') or entry.get('start', ''))
            except Exception:
                continue
            if entry_start < start_bound:
                continue
            if entry_start > end_bound:
                break
            st = entry.get('start_time') or entry.get('start', '')
            et = entry.get('end_time') or entry.get('end', '')
            lines.append(f"{st} --> {et}\n{entry['text']}")
        return "\n\n".join(lines)

    def _parse_grounding_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM grounding response JSON."""
        try:
            text = response.strip()
            if '```' in text:
                text = re.sub(r'```(?:json)?\s*', '', text)
                text = text.replace('```', '').strip()
            data = json.loads(text)
            start = data.get('start_time')
            end = data.get('end_time')
            if not start or not end:
                return None
            start_sec = self.time_to_seconds(start)
            end_sec = self.time_to_seconds(end)
            duration = end_sec - start_sec
            if duration <= 0:
                return None
            return {'start_time': start, 'end_time': end, 'duration_seconds': int(duration)}
        except Exception:
            return None

    def build_pre_verify_pool(self, highlights_files: List[str], pool_size: int) -> Dict[str, Any]:
        """
        Build a lightweight deterministic pre-verification pool for agentic analysis.

        This intentionally avoids the heavier LLM-based global aggregation used by the
        non-agentic path. It concatenates part-level candidates, normalizes their shape,
        drops exact duplicate windows, and caps the pool deterministically.
        """
        logger.info("🧺 Building deterministic pre-verification pool...")

        if pool_size <= 0:
            return self._create_empty_aggregation_result()

        all_moments: List[Dict[str, Any]] = []
        seen_windows: set[tuple[str, str, str]] = set()

        for file_path in highlights_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading highlights file {file_path}: {e}")
                continue

            video_part = data.get('video_part', 'unknown')
            for idx, raw_moment in enumerate(data.get('engaging_moments', []), start=1):
                moment = dict(raw_moment)
                start_time = moment.get('start_time', '00:00:00')
                end_time = moment.get('end_time', '00:00:00')
                source_key = (video_part, start_time, end_time)
                if source_key in seen_windows:
                    continue
                seen_windows.add(source_key)

                moment['_source_video_part'] = video_part
                moment['_source_rank'] = idx
                if 'timing' not in moment:
                    moment['timing'] = {
                        'video_part': video_part,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': f"{moment.get('duration_seconds', 0)}s",
                    }
                all_moments.append(moment)

        if not all_moments:
            logger.warning("No engaging moments found for pre-verification pool")
            return self._create_empty_aggregation_result()

        top_moments = all_moments[:pool_size]
        for i, moment in enumerate(top_moments, start=1):
            moment['rank'] = i
            if 'timing' not in moment:
                moment['timing'] = {
                    'video_part': moment.pop('_source_video_part', 'unknown'),
                    'start_time': moment.get('start_time', '00:00:00'),
                    'end_time': moment.get('end_time', '00:00:00'),
                    'duration': f"{moment.get('duration_seconds', 0)}s",
                }

        return {
            "top_engaging_moments": top_moments,
            "total_moments": len(top_moments),
            "analysis_timestamp": datetime.now().isoformat() + 'Z',
            "aggregation_criteria": (
                "Deterministic pre-verification pool built from per-part engaging moments"
            ),
            "analysis_summary": {
                "highest_engagement_themes": [],
                "total_engaging_content_time": "N/A",
                "recommendation": "Pre-verification pool assembled deterministically before standalone review",
            },
            "honorable_mentions": [],
        }
    
    def _extract_and_parse_aggregation_json(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from aggregation AI response with AI-powered fixing
        
        Args:
            response: Raw AI response
            
        Returns:
            Parsed and validated JSON result
        """
        # First try standard JSON parsing
        try:
            # Try direct parsing first
            result = json.loads(response.strip())
            logger.debug("Successfully parsed aggregation response as direct JSON")
            return self._validate_aggregation_result(result)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                logger.debug("Successfully parsed aggregation JSON from code block")
                return self._validate_aggregation_result(result)
            except json.JSONDecodeError:
                pass
        
        # If standard parsing fails, use AI to fix the JSON
        logger.info("Standard aggregation JSON parsing failed, using AI to fix JSON...")
        try:
            fixed_json = self._ai_fix_aggregation_json(response)
            result = json.loads(fixed_json)
            logger.debug("Successfully parsed AI-fixed aggregation JSON")
            return self._validate_aggregation_result(result)
        except Exception as e:
            logger.error(f"AI aggregation JSON fixing failed: {e}")
            # Export raw and fixed responses for debugging
            self._export_failed_aggregation_responses(response, fixed_json if 'fixed_json' in locals() else None, e)
            raise json.JSONDecodeError("Could not extract valid JSON from aggregation response", response, 0)
    
    def _ai_fix_aggregation_json(self, malformed_response: str) -> str:
        """
        Use AI to fix malformed aggregation JSON response
        
        Args:
            malformed_response: The malformed JSON response
            
        Returns:
            Fixed JSON string
        """
        fix_prompt = f"""
You are a JSON repair expert. I have a malformed JSON response for video moment aggregation that needs to be fixed.

The response should follow this structure:
{{
  "top_engaging_moments": [
    {{
      "rank": 1,
      "title": "七人接力鉴定假发造型，现场即兴互动引爆弹幕高潮！",
      "timing": {{
        "video_part": "part02",
        "start_time": "00:15:30",
        "end_time": "00:17:15",
        "duration": 105
      }},
      "summary": "Brief 1-2 sentence description of what happens in this moment.",
      "engagement_details": {{
        "engagement_level": "high"
      }},
      "why_engaging": "本场首粒进球，比分改写，凯恩展现巨星本色",
      "tags": ["进球", "赛事高光", "明星球员", "精彩集锦"]
    }}
  ],
  "total_moments": 5,
  "analysis_timestamp": "2024-01-01T12:00:00Z",
  "aggregation_criteria": "Selected based on engagement score, duration, and content quality",
  "analysis_summary": {{
    "highest_engagement_themes": ["进球", "战术解析", "赛事高光"],
    "total_engaging_content_time": "8 minutes 45 seconds",
    "recommendation": "These moments represent the most entertaining and shareable content from the livestream"
  }},
  "honorable_mentions": []
}}

IMPORTANT:
- start_time and end_time should be in simple format (HH:MM:SS or MM:SS), NOT SRT format with milliseconds
- Ensure all timing information is preserved accurately
- Use only approved tags: ["进球", "射门", "头球", "点球", "红牌", "黄牌", "扑救", "任意球", "角球", "越位", "VAR", "换人", "助攻", "反击", "庆祝", "绝杀", "逆转", "乌龙球", "无效进球", "明星球员", "战术亮点", "赛事高光", "战术解析", "球星表现", "历史纪录", "裁判争议", "数据分析", "教练布置", "阵型变化", "体能管理", "赛前分析", "赛后总结", "经典对决", "名嘴金句", "精彩集锦", "深度解析", "争议"]

Here is the malformed response:
{malformed_response}

Please fix the JSON and return ONLY the valid JSON, no explanations:
"""

        try:
            # Use a simpler model for JSON fixing
            fixed_response = self.llm_client.simple_chat(fix_prompt, model=self.model)
            
            # Extract JSON from the fixed response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', fixed_response, re.DOTALL)
            if json_match:
                return json_match.group(1)
            
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', fixed_response, re.DOTALL)
            if json_match:
                return json_match.group()
            
            # If no JSON found, return the entire response
            return fixed_response.strip()
            
        except Exception as e:
            logger.error(f"Error in AI aggregation JSON fixing: {e}")
            raise
    
    def _export_failed_responses(self, raw_response: str, part_name: str, fixed_response: Optional[str], error: Exception):
        """
        Export raw and AI-fixed responses when JSON parsing fails for debugging
        
        Args:
            raw_response: Original AI response
            part_name: Video part name
            fixed_response: AI-fixed response (if available)
            error: The parsing error that occurred
        """
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create debug directory
            debug_dir = Path("debug_responses")
            debug_dir.mkdir(exist_ok=True)
            
            # Export raw response
            raw_file = debug_dir / f"{part_name}_raw_response_{timestamp}.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"=== RAW AI RESPONSE FOR {part_name.upper()} ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {str(error)}\n")
                f.write(f"Response Length: {len(raw_response)} characters\n")
                f.write("=" * 50 + "\n\n")
                f.write(raw_response)
            
            # Export AI-fixed response if available
            if fixed_response:
                fixed_file = debug_dir / f"{part_name}_ai_fixed_response_{timestamp}.txt"
                with open(fixed_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== AI-FIXED RESPONSE FOR {part_name.upper()} ===\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Original Error: {str(error)}\n")
                    f.write(f"Fixed Response Length: {len(fixed_response)} characters\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(fixed_response)
            
            logger.info(f"📁 Exported failed responses to debug_responses/ directory")
            logger.info(f"   Raw response: {raw_file.name}")
            if fixed_response:
                logger.info(f"   AI-fixed response: {fixed_file.name}")
                
        except Exception as export_error:
            logger.error(f"Failed to export debug responses: {export_error}")
    
    def _export_failed_aggregation_responses(self, raw_response: str, fixed_response: Optional[str], error: Exception):
        """
        Export raw and AI-fixed aggregation responses when JSON parsing fails
        
        Args:
            raw_response: Original AI response
            fixed_response: AI-fixed response (if available)
            error: The parsing error that occurred
        """
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create debug directory
            debug_dir = Path("debug_responses")
            debug_dir.mkdir(exist_ok=True)
            
            # Export raw response
            raw_file = debug_dir / f"aggregation_raw_response_{timestamp}.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write("=== RAW AI AGGREGATION RESPONSE ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Error: {str(error)}\n")
                f.write(f"Response Length: {len(raw_response)} characters\n")
                f.write("=" * 50 + "\n\n")
                f.write(raw_response)
            
            # Export AI-fixed response if available
            if fixed_response:
                fixed_file = debug_dir / f"aggregation_ai_fixed_response_{timestamp}.txt"
                with open(fixed_file, 'w', encoding='utf-8') as f:
                    f.write("=== AI-FIXED AGGREGATION RESPONSE ===\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Original Error: {str(error)}\n")
                    f.write(f"Fixed Response Length: {len(fixed_response)} characters\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(fixed_response)
            
            logger.info(f"📁 Exported failed aggregation responses to debug_responses/ directory")
            logger.info(f"   Raw response: {raw_file.name}")
            if fixed_response:
                logger.info(f"   AI-fixed response: {fixed_file.name}")
                
        except Exception as export_error:
            logger.error(f"Failed to export debug aggregation responses: {export_error}")
    
    def _create_moments_context(self, moments: List[Dict[str, Any]]) -> str:
        """Create formatted context of all moments for aggregation"""
        context_lines = []
        
        for i, moment in enumerate(moments, 1):
            engagement_level = moment.get('engagement_details', {}).get('engagement_level', 'unknown')
            context_lines.append(f"""
Moment {i}:
- Part: {moment.get('source_part', 'unknown')}
- Title: {moment.get('title', 'No title')}
- Time: {moment.get('start_time', '')} --> {moment.get('end_time', '')}
- Duration: {moment.get('duration_seconds', 0)} seconds
- Engagement Level: {engagement_level}
- Tags: {', '.join(moment.get('tags', []))}
- Summary: {moment.get('summary', '')}
""")
        
        return '\n'.join(context_lines)
    
    def _validate_aggregation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean up aggregation result"""
        
        if 'top_engaging_moments' not in result:
            result['top_engaging_moments'] = []
        
        # Ensure proper ranking
        for i, moment in enumerate(result['top_engaging_moments']):
            moment['rank'] = i + 1
        
        result['total_moments'] = len(result['top_engaging_moments'])
        result['analysis_timestamp'] = datetime.now().isoformat() + 'Z'
        
        if 'aggregation_criteria' not in result:
            result['aggregation_criteria'] = "Selected based on engagement score, duration, and content quality"
        
        return result
    
    def _create_empty_aggregation_result(self) -> Dict[str, Any]:
        """Create empty aggregation result"""
        return {
            "top_engaging_moments": [],
            "total_moments": 0,
            "analysis_timestamp": datetime.now().isoformat() + 'Z',
            "aggregation_criteria": "No engaging moments found"
        }
    
    def _create_fallback_aggregation(self, all_moments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback aggregation without sorting - just take first N moments"""

        # Take first N moments (no sorting - LLM should have already ranked them)
        top_moments = all_moments[:self.max_clips]

        # Add ranking and ensure timing wrapper expected by clip_generator
        for i, moment in enumerate(top_moments):
            if 'rank' not in moment:
                moment['rank'] = i + 1
            if 'timing' not in moment:
                moment['timing'] = {
                    'video_part': moment.pop('_source_video_part', 'unknown'),
                    'start_time': moment.get('start_time', '00:00:00'),
                    'end_time': moment.get('end_time', '00:00:00'),
                    'duration': f"{moment.get('duration_seconds', 0)}s",
                }
        
        return {
            "top_engaging_moments": top_moments,
            "total_moments": len(top_moments),
            "analysis_timestamp": datetime.now().isoformat() + 'Z',
            "aggregation_criteria": f"Fallback selection - first {self.max_clips} moments",
            "analysis_summary": {
                "highest_engagement_themes": [],
                "total_engaging_content_time": "N/A",
                "recommendation": "Fallback aggregation used due to parsing error"
            },
            "honorable_mentions": []
        }
    
    async def save_highlights_to_file(self, highlights: Dict[str, Any], output_path: str):
        """Save highlights analysis to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(highlights, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Highlights saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving highlights to {output_path}: {e}")
            raise
