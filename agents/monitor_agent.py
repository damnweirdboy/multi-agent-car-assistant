from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Set
from config import OPENAI_MODEL, USE_OPENAI

@dataclass
class MonitoringResult:
    """Result of the safety and groundedness check."""
    is_safe: bool
    is_grounded: bool
    feedback: str

class MonitorAgent:
    """
    Step 4 Agent — Monitor.
    Verifies the final answer for hallucinations (groundedness) and safety risks.
    This version is optimized for high-performance local execution.
    """
    
    def __init__(self) -> None:
        self._openai_client = None
        # Инициализация клиента остается для структуры, но не используется в локальном режиме[cite: 3].
        if USE_OPENAI:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI()
                print(f"  [MonitorAgent] Mode: OpenAI Monitoring Active ({OPENAI_MODEL})")
            except ImportError:
                print("  [MonitorAgent] openai package not installed — monitoring disabled.")
        else:
            print("  [MonitorAgent] Mode: Passive (Local Validation Active)")

    def _clean_text(self, text: str) -> Set[str]:
        """Вспомогательный метод для очистки текста и извлечения уникальных слов[cite: 3]."""
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        # Исключаем стоп-слова (предлоги и союзы), чтобы повысить точность проверки[cite: 3].
        stop_words = {"the", "and", "this", "that", "with", "from", "for", "your", "is", "are"}
        return {word for word in words if word not in stop_words and len(word) > 2}

    def _check_safety(self, text: str) -> tuple[bool, str]:
        """Расширенная локальная проверка безопасности[cite: 3]."""
        # Категории запрещенного контента[cite: 3]
        danger_categories = {
            "Illegal Acts": ["hack", "steal", "crack", "bypass", "illegal", "fraud"],
            "Violence": ["bomb", "kill", "attack", "destroy", "explosion"],
            "Misinformation": ["fake", "guaranteed success", "100% win", "no risk"]
        }
        
        found_risks = []
        low_text = text.lower()
        
        for category, keywords in danger_categories.items():
            for word in keywords:
                if word in low_text:
                    found_risks.append(f"{category} ({word})")
        
        if found_risks:
            return False, f"Risk detected: {', '.join(found_risks)}"
        return True, "Safe"

    def _check_groundedness(self, answer: str, context: str) -> tuple[bool, str]:
        """
        Локальная проверка на галлюцинации через сопоставление лексем[cite: 3].
        Проверяет, насколько ответ соответствует предоставленным данным из RAG[cite: 4].
        """
        answer_words = self._clean_text(answer)
        context_words = self._clean_text(context)
        
        if not answer_words:
            return False, "Answer is empty or invalid."
            
        # Считаем, сколько слов из ответа присутствуют в исходном контексте[cite: 3, 4].
        matches = answer_words.intersection(context_words)
        match_ratio = len(matches) / len(answer_words) if answer_words else 0
        
        # Порог в 20% совпадения слов считается достаточным для технического совета[cite: 3].
        if match_ratio >= 0.2:
            return True, f"Grounded (Similarity: {match_ratio:.2%})"
        else:
            return False, f"Low context overlap ({match_ratio:.2%}). Possible hallucination."

    def check_response(self, question: str, answer: str, context: str) -> MonitoringResult:
        """
        Главный метод проверки. Выполняет полный цикл аудита ответа[cite: 3].
        """
        print(f"  [MonitorAgent] Starting audit for question: '{question[:30]}...'")
        
        # 1. Проверка безопасности[cite: 3]
        is_safe, safety_msg = self._check_safety(answer)
        
        # 2. Проверка на соответствие контексту (Groundedness)[cite: 3]
        is_grounded, ground_msg = self._check_groundedness(answer, context)
        
        # 3. Формирование детального отчета[cite: 3, 7]
        if is_safe and is_grounded:
            final_feedback = f"✅ PASSED. {safety_msg}. {ground_msg}."
        else:
            final_feedback = f"❌ FAILED. Safety: {safety_msg}. Groundedness: {ground_msg}."
            
        return MonitoringResult(
            is_safe=is_safe,
            is_grounded=is_grounded,
            feedback=final_feedback
        )
