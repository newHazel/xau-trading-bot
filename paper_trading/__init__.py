"""Paper trading modules — Phase 7."""

from .entry_conditions import PaperEntryConditions, EntryConditionResult
from .paper_engine import PaperEngine, PaperSignal, PaperPosition, PaperTradeResult
from .paper_journal import PaperJournal, JournalEntry
from .paper_stats import PaperStats, PaperStatsResult, ComparisonResult
from .paper_rules import PaperRules, GraduationResult

__all__ = [
    "PaperEntryConditions", "EntryConditionResult",
    "PaperEngine", "PaperSignal", "PaperPosition", "PaperTradeResult",
    "PaperJournal", "JournalEntry",
    "PaperStats", "PaperStatsResult", "ComparisonResult",
    "PaperRules", "GraduationResult",
]
