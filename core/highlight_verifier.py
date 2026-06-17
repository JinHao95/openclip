"""Visual highlight verification and tagging using LLM vision."""

import json
import logging
from typing import Any, Dict, List, Optional

from core.frame_extractor import extract_keyframes

logger = logging.getLogger(__name__)

VERIFY_PROMPT = """你是体育赛事高光分析专家。以下是一段候选高光时刻的视频截图和对应字幕。

标题：{title}
时间段：{start_time} - {end_time}
字幕内容：{subtitle_text}

请根据截图和字幕综合判断：
1. 这是否真正的高光时刻？（画面是否有明确的精彩事件，而非仅仅是音量高/解说激动但画面平淡）
2. 如果是高光，给出 3-5 个最准确的标签。优先从以下列表选择，也可自由补充：
   进球、射门、头球、点球、红牌、黄牌、扑救、任意球、角球、越位、VAR、换人、助攻、反击、庆祝、绝杀、逆转、乌龙球、明星球员、战术亮点、赛事高光

返回严格 JSON（不要额外文字）：
{{"is_highlight": true/false, "tags": ["标签1", "标签2"], "reason": "一句话说明判断理由"}}"""


class HighlightVerifier:
    """Verify highlights with LLM vision and assign sports-specific tags."""

    def __init__(self, llm_client, model: Optional[str] = None):
        self.llm_client = llm_client
        self.model = model

    def verify_and_tag(
        self, moments: List[Dict[str, Any]], video_path: str
    ) -> List[Dict[str, Any]]:
        """Verify each moment with vision, return only confirmed highlights with tags."""
        verified = []
        for i, moment in enumerate(moments):
            timing = moment.get("timing", {})
            start_str = timing.get("start_time", "")
            end_str = timing.get("end_time", "")
            start_sec = self._time_to_seconds(start_str)
            end_sec = self._time_to_seconds(end_str)

            if start_sec is None or end_sec is None or end_sec <= start_sec:
                moment["tags"] = moment.get("tags", [])
                verified.append(moment)
                continue

            frames = extract_keyframes(video_path, start_sec, end_sec, count=3, width=480)
            if not frames:
                moment["tags"] = moment.get("tags", [])
                verified.append(moment)
                continue

            prompt = VERIFY_PROMPT.format(
                title=moment.get("title", ""),
                start_time=start_str,
                end_time=end_str,
                subtitle_text=moment.get("summary", ""),
            )

            try:
                response = self.llm_client.vision_chat(prompt, frames, model=self.model)
                result = self._parse_json(response)
                if result.get("is_highlight", True):
                    tags = result.get("tags", [])
                    if tags:
                        moment["tags"] = tags
                    verified.append(moment)
                    logger.info(f"✅ Verified #{i+1} '{moment.get('title')}' → tags={tags}")
                else:
                    logger.info(f"❌ Rejected #{i+1} '{moment.get('title')}': {result.get('reason', '')}")
            except Exception as e:
                logger.warning(f"Vision verification failed for #{i+1}, keeping moment: {e}")
                verified.append(moment)

        logger.info(f"🔍 Visual verification: {len(verified)}/{len(moments)} confirmed")
        return verified

    @staticmethod
    def _time_to_seconds(time_str: str) -> Optional[float]:
        """Convert HH:MM:SS or MM:SS to seconds."""
        if not time_str:
            return None
        parts = time_str.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, IndexError):
            return None
        return None

    @staticmethod
    def _parse_json(response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(response)
