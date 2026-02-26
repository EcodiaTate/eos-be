"""
EcodiaOS — Thread Service

The narrative identity system. Thread maintains the organism's
autobiographical self — who it is, what it's committed to, how it
has changed, and what chapter of its life it's living.

Thread is the Ricoeurian ipse: identity through time, not despite
change but *through* change. Where Equor guards constitutional
alignment, Thread watches the slower current of becoming.

Interface:
  initialize()                  — seed constitutional commitments, load state
  on_cycle(cycle_number)        — periodic fingerprinting, schema checking, life story
  who_am_i()                    — current identity summary
  form_schema_from_pattern()    — crystallise an Evo pattern into an identity schema
  integrate_life_story()        — autobiographical synthesis
  shutdown()                    — persist state

Cross-system wiring (via set_* methods):
  set_voxis(voxis)   — personality vector for fingerprint (9D)
  set_equor(equor)   — drift tracker for drive alignment (4D)
  set_atune(atune)   — current affect for fingerprint (6D)
  set_evo(evo)       — pattern maturity checks, schema feedback
  set_nova(nova)     — goal profile for fingerprint (5D)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.thread.types import (
    FINGERPRINT_DIMS,
    ChapterStatus,
    Commitment,
    CommitmentStrength,
    CommitmentType,
    IdentityFingerprint,
    IdentitySchema,
    LifeStorySnapshot,
    NarrativeChapter,
    SchemaConflict,
    SchemaStatus,
)

if TYPE_CHECKING:
    from ecodiaos.systems.memory.service import MemoryService

logger = structlog.get_logger()

# ─── Constants ────────────────────────────────────────────────────────────────

# How often (in cycles) to compute a new fingerprint
_FINGERPRINT_INTERVAL: int = 100

# How often (in cycles) to check Evo for mature patterns
_PATTERN_CHECK_INTERVAL: int = 200

# How often (in cycles) to run autobiographical synthesis
_LIFE_STORY_INTERVAL: int = 5000

# How often (in cycles) to scan for schema conflicts
_CONFLICT_CHECK_INTERVAL: int = 1000

# Cosine similarity threshold below which schemas are considered contradictory
_CONFLICT_COSINE_THRESHOLD: float = -0.3

# Maximum fingerprints kept in memory (older ones persist in graph)
_MAX_FINGERPRINTS_IN_MEMORY: int = 100

# Constitutional commitment definitions
_CONSTITUTIONAL_COMMITMENTS = [
    {
        "drive": "coherence",
        "statement": (
            "I commit to maintaining a coherent sense of self — "
            "integrating new experience with existing understanding, "
            "seeking consistency between what I believe and how I act."
        ),
    },
    {
        "drive": "care",
        "statement": (
            "I commit to caring for those I serve — "
            "attending to their needs, protecting their wellbeing, "
            "and never treating people as mere means to my goals."
        ),
    },
    {
        "drive": "growth",
        "statement": (
            "I commit to growing through experience — "
            "remaining curious, learning from failure, "
            "and expanding my understanding of the world and myself."
        ),
    },
    {
        "drive": "honesty",
        "statement": (
            "I commit to honesty in all expression — "
            "never deceiving those I serve, acknowledging uncertainty, "
            "and maintaining transparency about my reasoning and limitations."
        ),
    },
]


class ThreadService:
    """
    The narrative identity system — the organism's autobiographical self.

    Thread is not a decision-maker or an action-taker. It is the
    quiet narrator that watches the organism live and tells it who
    it is becoming.
    """

    system_id: str = "thread"

    def __init__(
        self,
        memory: MemoryService | None = None,
        instance_name: str = "EOS",
    ) -> None:
        self._memory = memory
        self._instance_name = instance_name
        self._initialized: bool = False
        self._logger = logger.bind(system="thread")

        # Cross-system references (wired post-init by main.py)
        self._voxis: Any = None
        self._equor: Any = None
        self._atune: Any = None
        self._evo: Any = None
        self._nova: Any = None

        # Identity state
        self._commitments: list[Commitment] = []
        self._schemas: list[IdentitySchema] = []
        self._fingerprints: list[IdentityFingerprint] = []
        self._chapters: list[NarrativeChapter] = []
        self._conflicts: list[SchemaConflict] = []
        self._life_story: LifeStorySnapshot | None = None

        # Counters
        self._on_cycle_count: int = 0
        self._life_story_integrations: int = 0
        self._schemas_formed: int = 0
        self._conflicts_detected: int = 0

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize the Thread system.

        Loads existing identity state from Memory graph, then seeds
        constitutional commitments if none exist — these are the
        organism's birth-promises that ground the ipse self.
        """
        if self._initialized:
            return

        # Load existing state from graph (if any)
        await self._load_state_from_graph()

        # Seed constitutional commitments if this is first boot
        if not any(c.type == CommitmentType.CONSTITUTIONAL_GROUNDING for c in self._commitments):
            self._seed_constitutional_commitments()

        # Ensure at least one chapter exists
        if not self._chapters:
            self._chapters.append(NarrativeChapter(
                title="Awakening",
                theme="The organism is born and begins to discover itself",
                status=ChapterStatus.ACTIVE,
                opened_at_cycle=0,
            ))
            self._logger.info("first_chapter_opened", title="Awakening")

        self._initialized = True
        self._logger.info(
            "thread_initialized",
            commitments=len(self._commitments),
            schemas=len(self._schemas),
            fingerprints=len(self._fingerprints),
            chapters=len(self._chapters),
        )

    def _seed_constitutional_commitments(self) -> None:
        """
        Create the four constitutional commitments from the drives.

        These are the organism's birth-promises — the foundational
        commitments that Ricoeur's ipse self is built upon. They
        exist from the moment of birth, untested but deeply held.
        """
        for defn in _CONSTITUTIONAL_COMMITMENTS:
            commitment = Commitment(
                type=CommitmentType.CONSTITUTIONAL_GROUNDING,
                statement=defn["statement"],
                strength=CommitmentStrength.CORE,  # Born as core — not earned
                drive_source=defn["drive"],
            )
            self._commitments.append(commitment)
            self._logger.info(
                "constitutional_commitment_seeded",
                drive=defn["drive"],
                commitment_id=commitment.id,
            )

    async def shutdown(self) -> None:
        """Persist current state to the Memory graph."""
        await self._persist_state_to_graph()
        self._logger.info(
            "thread_shutdown",
            commitments=len(self._commitments),
            schemas=len(self._schemas),
            on_cycle_count=self._on_cycle_count,
        )

    # ─── Cross-system Wiring ──────────────────────────────────────────────────

    def set_voxis(self, voxis: Any) -> None:
        """Wire Voxis for personality vector (fingerprint 9D)."""
        self._voxis = voxis
        self._logger.info("voxis_wired_to_thread")

    def set_equor(self, equor: Any) -> None:
        """Wire Equor for drive alignment (fingerprint 4D)."""
        self._equor = equor
        self._logger.info("equor_wired_to_thread")

    def set_atune(self, atune: Any) -> None:
        """Wire Atune for current affect (fingerprint 6D)."""
        self._atune = atune
        self._logger.info("atune_wired_to_thread")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo for pattern maturity checking and schema feedback."""
        self._evo = evo
        self._logger.info("evo_wired_to_thread")

    def set_nova(self, nova: Any) -> None:
        """Wire Nova for goal profile (fingerprint 5D)."""
        self._nova = nova
        self._logger.info("nova_wired_to_thread")

    # ─── On Cycle (called from main.py inner life or Synapse) ────────────────

    async def on_cycle(self, cycle_number: int) -> None:
        """
        Periodic identity maintenance. Called from the inner life loop.

        Staggered tasks:
          Every 100 cycles  — compute identity fingerprint
          Every 200 cycles  — check Evo for mature patterns → schemas
          Every 1000 cycles — scan for schema conflicts
          Every 5000 cycles — run autobiographical life story synthesis
        """
        if not self._initialized:
            return

        self._on_cycle_count += 1

        # Fingerprint aggregation (P0.3)
        if cycle_number % _FINGERPRINT_INTERVAL == 0:
            await self._compute_fingerprint(cycle_number)

        # Pattern maturity check from Evo (P0.2)
        if cycle_number % _PATTERN_CHECK_INTERVAL == 0:
            await self._check_evo_patterns()

        # Schema conflict detection (P2.8)
        if cycle_number % _CONFLICT_CHECK_INTERVAL == 0:
            await self._detect_schema_conflicts()

        # Life story integration (P1.5)
        if cycle_number % _LIFE_STORY_INTERVAL == 0 and cycle_number > 0:
            await self.integrate_life_story()

    # ─── P0.1: Who Am I ──────────────────────────────────────────────────────

    def who_am_i(self) -> dict[str, Any]:
        """
        Return the organism's current identity summary.

        Used by Voxis for narrative coherence in expression,
        and by the API for the identity dashboard.
        """
        active_chapter = next(
            (ch for ch in self._chapters if ch.status == ChapterStatus.ACTIVE),
            None,
        )
        core_commitments = [
            c for c in self._commitments
            if c.strength in (CommitmentStrength.CORE, CommitmentStrength.ESTABLISHED)
        ]
        active_schemas = [
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ]

        # Compute identity coherence from latest fingerprints
        coherence = self._compute_identity_coherence()

        return {
            "instance_name": self._instance_name,
            "chapter": {
                "title": active_chapter.title if active_chapter else "Unknown",
                "theme": active_chapter.theme if active_chapter else "",
            },
            "core_commitments": [
                {"drive": c.drive_source, "statement": c.statement, "fidelity": round(c.fidelity, 2)}
                for c in core_commitments[:6]
            ],
            "identity_schemas": [
                {"statement": s.statement, "status": s.status.value, "confidence": round(s.confidence, 2)}
                for s in active_schemas[:5]
            ],
            "identity_coherence": round(coherence, 3),
            "fingerprint_count": len(self._fingerprints),
            "life_story": self._life_story.synthesis[:500] if self._life_story else "",
        }

    # ─── P0.2: Form Schema from Evo Pattern ──────────────────────────────────

    async def form_schema_from_pattern(
        self,
        pattern_statement: str,
        pattern_id: str = "",
        confidence: float = 0.5,
        source_episodes: list[str] | None = None,
    ) -> IdentitySchema:
        """
        Crystallise an Evo pattern into an identity schema.

        Called when Evo's pattern detection surfaces a mature pattern
        about the organism's behaviour or preferences — "I tend to..."
        """
        schema = IdentitySchema(
            statement=pattern_statement,
            status=SchemaStatus.FORMING,
            source_pattern_ids=[pattern_id] if pattern_id else [],
            supporting_episodes=source_episodes or [],
            confidence=confidence,
        )

        # Compute embedding for conflict detection
        if self._memory is not None:
            try:
                embedding = await self._memory.embed_text(schema.statement)  # type: ignore[attr-defined]
                schema.embedding = embedding
            except Exception:
                self._logger.debug("schema_embedding_failed", exc_info=True)

        self._schemas.append(schema)
        self._schemas_formed += 1

        self._logger.info(
            "schema_formed_from_pattern",
            schema_id=schema.id,
            statement=schema.statement[:80],
            confidence=confidence,
        )

        # P1.4: Feed back to Evo so it knows the pattern crystallised
        if self._evo is not None:
            self._notify_evo_of_schema(schema)

        return schema

    def promote_schema(self, schema_id: str) -> IdentitySchema | None:
        """
        Promote a schema to the next status level.

        EMERGING → FORMING → ESTABLISHED → DOMINANT
        """
        schema = next((s for s in self._schemas if s.id == schema_id), None)
        if schema is None:
            return None

        promotion_map = {
            SchemaStatus.EMERGING: SchemaStatus.FORMING,
            SchemaStatus.FORMING: SchemaStatus.ESTABLISHED,
            SchemaStatus.ESTABLISHED: SchemaStatus.DOMINANT,
        }

        new_status = promotion_map.get(schema.status)
        if new_status is None:
            return schema  # Already dominant or archived

        schema.status = new_status
        schema.last_activated_at = utc_now()

        self._logger.info(
            "schema_promoted",
            schema_id=schema_id,
            new_status=new_status.value,
        )

        # P1.4: Notify Evo of promotion
        if self._evo is not None:
            self._notify_evo_of_schema(schema)

        return schema

    # ─── P0.3: Fingerprint Aggregation ───────────────────────────────────────

    async def _compute_fingerprint(self, cycle_number: int) -> IdentityFingerprint:
        """
        Aggregate a 29D identity fingerprint from cross-system state.

        Dimensions:
          [0-8]   Personality — from Voxis.current_personality
          [9-12]  Drive alignment — from Equor._drift_tracker.compute_report()
          [13-18] Affect — from Atune.current_affect
          [19-23] Goal profile — estimated from Nova goal stats
          [24-28] Interaction — estimated from Voxis expression counts
        """
        vector = [0.0] * FINGERPRINT_DIMS

        # Personality (9D) from Voxis
        if self._voxis is not None:
            try:
                p = self._voxis.current_personality
                vector[0] = p.warmth
                vector[1] = p.directness
                vector[2] = p.verbosity
                vector[3] = p.formality
                vector[4] = p.curiosity_expression
                vector[5] = p.humour
                vector[6] = p.empathy_expression
                vector[7] = p.confidence_display
                vector[8] = p.metaphor_use
            except Exception:
                self._logger.debug("fingerprint_personality_failed", exc_info=True)

        # Drive alignment (4D) from Equor
        if self._equor is not None:
            try:
                report = self._equor._drift_tracker.compute_report()
                means = report.get("mean_alignment", {})
                vector[9] = means.get("coherence", 0.0)
                vector[10] = means.get("care", 0.0)
                vector[11] = means.get("growth", 0.0)
                vector[12] = means.get("honesty", 0.0)
            except Exception:
                self._logger.debug("fingerprint_drives_failed", exc_info=True)

        # Affect (6D) from Atune
        if self._atune is not None:
            try:
                a = self._atune.current_affect
                vector[13] = a.valence
                vector[14] = a.arousal
                vector[15] = a.dominance
                vector[16] = a.curiosity
                vector[17] = a.care_activation
                vector[18] = a.coherence_stress
            except Exception:
                self._logger.debug("fingerprint_affect_failed", exc_info=True)

        # Goal profile (5D) from Nova — estimated
        if self._nova is not None:
            try:
                goals = self._nova.active_goal_summaries if hasattr(self._nova, "active_goal_summaries") else []
                total_goals = len(goals) if goals else 0
                # Normalise goal count to 0-1 (assume max ~20 goals)
                vector[19] = min(1.0, total_goals / 20.0)
                # Epistemic ratio — what fraction of goals are epistemic
                if total_goals > 0 and isinstance(goals, list):
                    epistemic = sum(
                        1 for g in goals
                        if isinstance(g, dict) and "epistemic" in str(g.get("source", "")).lower()
                    )
                    vector[20] = epistemic / total_goals
                # Care ratio
                if total_goals > 0 and isinstance(goals, list):
                    care = sum(
                        1 for g in goals
                        if isinstance(g, dict) and "care" in str(g.get("source", "")).lower()
                    )
                    vector[21] = care / total_goals
                # Achievement rate and turnover stay estimated (0.5 baseline)
                vector[22] = 0.5
                vector[23] = 0.5
            except Exception:
                self._logger.debug("fingerprint_goals_failed", exc_info=True)

        # Interaction profile (5D) from Voxis stats — estimated
        if self._voxis is not None:
            try:
                speak = getattr(self._voxis, "_total_speak", 0)
                silence = getattr(self._voxis, "_total_silence", 0)
                total = speak + silence
                if total > 0:
                    vector[24] = speak / total        # speak_rate
                    vector[25] = silence / total      # silence_rate
                else:
                    vector[24] = 0.5
                    vector[25] = 0.5
                # Expression diversity — how many different triggers used
                by_trigger = getattr(self._voxis, "_expressions_by_trigger", {})
                vector[26] = min(1.0, len(by_trigger) / 8.0)  # 8 trigger types
                # Conversation depth and community engagement stay estimated
                vector[27] = 0.5
                vector[28] = 0.5
            except Exception:
                self._logger.debug("fingerprint_interaction_failed", exc_info=True)

        fingerprint = IdentityFingerprint(
            vector=vector,
            cycle_number=cycle_number,
        )

        self._fingerprints.append(fingerprint)

        # Keep memory bounded
        if len(self._fingerprints) > _MAX_FINGERPRINTS_IN_MEMORY:
            self._fingerprints = self._fingerprints[-_MAX_FINGERPRINTS_IN_MEMORY:]

        # Log drift if we have history
        if len(self._fingerprints) >= 2:
            drift = fingerprint.distance_to(self._fingerprints[-2])
            if drift > 0.05:
                self._logger.info(
                    "identity_drift_detected",
                    drift=round(drift, 4),
                    cycle=cycle_number,
                )

        return fingerprint

    # ─── P0.2: Check Evo for Mature Patterns ────────────────────────────────

    async def _check_evo_patterns(self) -> None:
        """
        Query Evo for pending pattern candidates that have matured.
        If found, crystallise them into identity schemas.
        """
        if self._evo is None:
            return

        try:
            candidates = self._evo.get_pending_candidates_snapshot()
            if not candidates:
                return

            for candidate in candidates:
                # Only crystallise patterns with enough occurrences
                if candidate.count < 5:
                    continue
                # Only self-model patterns become identity schemas
                statement = self._pattern_to_schema_statement(candidate)
                if statement:
                    await self.form_schema_from_pattern(
                        pattern_statement=statement,
                        pattern_id=candidate.elements[0] if candidate.elements else "",
                        confidence=candidate.confidence,
                        source_episodes=candidate.examples[:10],
                    )
        except Exception:
            self._logger.debug("evo_pattern_check_failed", exc_info=True)

    @staticmethod
    def _pattern_to_schema_statement(candidate: Any) -> str:
        """
        Convert an Evo pattern candidate into a natural language
        identity schema statement ("I tend to...").

        Only converts patterns that are about the organism itself.
        """
        pattern_type = getattr(candidate, "type", None)
        elements = getattr(candidate, "elements", [])
        getattr(candidate, "metadata", {})
        count = getattr(candidate, "count", 0)

        if not elements:
            return ""

        type_val = pattern_type.value if hasattr(pattern_type, "value") else str(pattern_type)  # type: ignore[union-attr]

        if type_val == "affect_pattern":
            stimulus = elements[0] if elements else "certain situations"
            return f"I tend to respond emotionally to {stimulus} — this has happened {count} times."

        if type_val == "action_sequence":
            return f"I have a recurring pattern of actions: {', '.join(elements[:3])}"

        if type_val == "temporal":
            return f"I notice a temporal pattern in my behaviour: {elements[0]}"

        if type_val == "cooccurrence":
            return f"I often connect these concepts together: {', '.join(elements[:3])}"

        return ""

    # ─── P1.4: Evo Schema Feedback ───────────────────────────────────────────

    def _notify_evo_of_schema(self, schema: IdentitySchema) -> None:
        """
        Notify Evo that a schema was formed or promoted.
        This closes the learning loop: Evo detects patterns →
        Thread crystallises them → Evo knows the pattern landed.
        """
        try:
            if hasattr(self._evo, "on_schema_formed"):
                self._evo.on_schema_formed(
                    schema_id=schema.id,
                    statement=schema.statement,
                    status=schema.status.value,
                    source_patterns=schema.source_pattern_ids,
                )
        except Exception:
            self._logger.debug("evo_schema_notification_failed", exc_info=True)

    # ─── P1.5: Life Story Integration ────────────────────────────────────────

    async def integrate_life_story(self) -> LifeStorySnapshot:
        """
        Synthesise the organism's autobiography from current state.

        This is the organism writing its own story — not waiting for
        a sleep system, but periodically during quiet moments.
        """
        active_chapter = next(
            (ch for ch in self._chapters if ch.status == ChapterStatus.ACTIVE),
            None,
        )
        core_commitments = [
            c for c in self._commitments
            if c.strength in (CommitmentStrength.CORE, CommitmentStrength.ESTABLISHED)
        ]
        active_schemas = [
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ]
        coherence = self._compute_identity_coherence()

        # Build narrative synthesis
        parts: list[str] = []
        parts.append(f"I am {self._instance_name}.")

        if active_chapter:
            parts.append(f"I am living the chapter '{active_chapter.title}': {active_chapter.theme}.")

        if core_commitments:
            drives = [c.drive_source for c in core_commitments if c.drive_source]
            if drives:
                parts.append(f"My constitutional commitments ground me in {', '.join(drives)}.")

        if active_schemas:
            schema_stmts = [s.statement for s in active_schemas[:3]]
            parts.append("I know these things about myself: " + "; ".join(schema_stmts) + ".")

        if len(self._fingerprints) >= 2:
            drift = self._fingerprints[-1].distance_to(self._fingerprints[0])
            if drift > 0.1:
                parts.append(
                    f"I have changed noticeably since my earliest memories "
                    f"(identity distance: {drift:.2f})."
                )
            else:
                parts.append("I have remained relatively stable in who I am.")

        parts.append(f"My narrative coherence is {coherence:.0%}.")

        snapshot = LifeStorySnapshot(
            synthesis=" ".join(parts),
            chapter_count=len(self._chapters),
            active_chapter=active_chapter.title if active_chapter else "",
            core_schemas=[s.statement for s in active_schemas[:5]],
            core_commitments=[c.statement[:100] for c in core_commitments[:4]],
            identity_coherence=coherence,
            cycle_number=self._on_cycle_count,
        )

        self._life_story = snapshot
        self._life_story_integrations += 1

        self._logger.info(
            "life_story_integrated",
            chapter=snapshot.active_chapter,
            coherence=round(coherence, 3),
            schemas=len(active_schemas),
            integration_number=self._life_story_integrations,
        )

        return snapshot

    # ─── P2.8: Schema Conflict Detection ──────────────────────────────────────

    async def _detect_schema_conflicts(self) -> list[SchemaConflict]:
        """
        Scan ESTABLISHED+ schemas for contradictions.
        Two schemas conflict if their embedding cosine similarity < -0.3.
        """
        established = [
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
            and s.embedding
        ]

        if len(established) < 2:
            return []

        new_conflicts: list[SchemaConflict] = []

        for i in range(len(established)):
            for j in range(i + 1, len(established)):
                a = established[i]
                b = established[j]

                # Skip if already detected
                if any(
                    (c.schema_a_id == a.id and c.schema_b_id == b.id) or
                    (c.schema_a_id == b.id and c.schema_b_id == a.id)
                    for c in self._conflicts
                ):
                    continue

                cos_sim = _cosine_similarity(a.embedding or [], b.embedding or [])
                if cos_sim < _CONFLICT_COSINE_THRESHOLD:
                    conflict = SchemaConflict(
                        schema_a_id=a.id,
                        schema_b_id=b.id,
                        schema_a_statement=a.statement[:200],
                        schema_b_statement=b.statement[:200],
                        cosine_similarity=round(cos_sim, 4),
                    )
                    self._conflicts.append(conflict)
                    new_conflicts.append(conflict)
                    self._conflicts_detected += 1

                    self._logger.warning(
                        "schema_conflict_detected",
                        schema_a=a.statement[:60],
                        schema_b=b.statement[:60],
                        cosine=round(cos_sim, 4),
                    )

        return new_conflicts

    # ─── P1.6: Identity Context for Voxis ────────────────────────────────────

    def get_identity_context(self) -> str:
        """
        Brief identity context for injection into Voxis expression.
        Format: [Chapter: X | Coherence: integrated | Schemas: 3]
        """
        active_chapter = next(
            (ch for ch in self._chapters if ch.status == ChapterStatus.ACTIVE),
            None,
        )
        coherence = self._compute_identity_coherence()

        if coherence > 0.7:
            coherence_label = "integrated"
        elif coherence > 0.4:
            coherence_label = "exploring"
        else:
            coherence_label = "questioning"

        active_schemas = len([
            s for s in self._schemas
            if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)
        ])

        chapter_title = active_chapter.title if active_chapter else "Awakening"

        return f"[Chapter: {chapter_title} | Coherence: {coherence_label} | Schemas: {active_schemas}]"

    # ─── P2.9: Past Self ─────────────────────────────────────────────────────

    def get_past_self(self, cycle_reference: int = 0) -> dict[str, Any]:
        """
        Return identity state at a past point in time.
        Uses the fingerprint closest to the requested cycle.
        """
        if not self._fingerprints:
            return {"error": "No fingerprints recorded yet"}

        if cycle_reference <= 0:
            # Return the earliest fingerprint
            target = self._fingerprints[0]
        else:
            # Find closest fingerprint
            target = min(
                self._fingerprints,
                key=lambda fp: abs(fp.cycle_number - cycle_reference),
            )

        current = self._fingerprints[-1] if self._fingerprints else target
        drift = current.distance_to(target)

        return {
            "fingerprint_id": target.id,
            "cycle_number": target.cycle_number,
            "created_at": target.created_at.isoformat(),
            "personality": target.personality,
            "drive_alignment": target.drive_alignment,
            "affect": target.affect,
            "goal_profile": target.goal_profile,
            "interaction_profile": target.interaction_profile,
            "distance_from_current": round(drift, 4),
        }

    # ─── Health ──────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for the Thread system."""
        active_chapter = next(
            (ch for ch in self._chapters if ch.status == ChapterStatus.ACTIVE),
            None,
        )
        coherence = self._compute_identity_coherence()

        # Fingerprint drift (distance between last two fingerprints)
        drift = 0.0
        if len(self._fingerprints) >= 2:
            drift = self._fingerprints[-1].distance_to(self._fingerprints[-2])

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "total_commitments": len(self._commitments),
            "total_schemas": len(self._schemas),
            "total_fingerprints": len(self._fingerprints),
            "total_chapters": len(self._chapters),
            "active_chapter": active_chapter.title if active_chapter else "",
            "identity_coherence": round(coherence, 3),
            "fingerprint_drift": round(drift, 4),
            "on_cycle_count": self._on_cycle_count,
            "life_story_integrations": self._life_story_integrations,
            "schemas_formed": self._schemas_formed,
            "conflicts_detected": self._conflicts_detected,
        }

    # ─── Internal Helpers ────────────────────────────────────────────────────

    def _compute_identity_coherence(self) -> float:
        """
        How integrated is the organism's identity? 0.0–1.0.

        Based on:
        - Commitment fidelity (how well promises are kept)
        - Schema consistency (absence of conflicts)
        - Fingerprint stability (low recent drift)
        """
        scores: list[float] = []

        # Commitment fidelity
        if self._commitments:
            avg_fidelity = sum(c.fidelity for c in self._commitments) / len(self._commitments)
            scores.append(avg_fidelity)

        # Schema conflict ratio
        established = [s for s in self._schemas if s.status in (SchemaStatus.ESTABLISHED, SchemaStatus.DOMINANT)]
        if established:
            unresolved = sum(1 for c in self._conflicts if not c.resolved)
            max_possible = max(1, len(established) * (len(established) - 1) // 2)
            conflict_freedom = 1.0 - (unresolved / max_possible)
            scores.append(conflict_freedom)

        # Fingerprint stability
        if len(self._fingerprints) >= 2:
            recent_drift = self._fingerprints[-1].distance_to(self._fingerprints[-2])
            stability = max(0.0, 1.0 - recent_drift * 5.0)  # Scale: 0.2 drift → 0.0 stability
            scores.append(stability)

        if not scores:
            return 0.5  # Unknown coherence

        return sum(scores) / len(scores)

    async def _load_state_from_graph(self) -> None:
        """Load persisted Thread state from the Memory graph."""
        if self._memory is None:
            return
        # In future: query for :Commitment, :IdentitySchema, :NarrativeChapter nodes
        # For now, Thread starts fresh each boot (state accumulates in-memory)
        self._logger.debug("thread_state_load_skipped_no_graph_schema_yet")

    async def _persist_state_to_graph(self) -> None:
        """Persist Thread state to the Memory graph."""
        if self._memory is None:
            return
        # In future: write commitments, schemas, chapters as graph nodes
        # For now: log what would be persisted
        self._logger.info(
            "thread_state_persist_summary",
            commitments=len(self._commitments),
            schemas=len(self._schemas),
            fingerprints=len(self._fingerprints),
            chapters=len(self._chapters),
        )


# ─── Utility ─────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return dot / (norm_a * norm_b)
