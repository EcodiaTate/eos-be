"""
Atune — Input Normalisation.

Converts raw input from any :class:`InputChannel` into the standard
:class:`Percept` format.  Every channel has a dedicated normaliser that
knows how to extract textual content, assign a default modality, and set a
salience hint.
"""

from __future__ import annotations

import structlog

from ecodiaos.primitives.common import Modality, SourceDescriptor, SystemID, new_id, utc_now
from ecodiaos.primitives.percept import (
    Content,
    Percept,
    Provenance,
    TransformRecord,
)

from .helpers import compute_hash_chain, hash_content
from .types import InputChannel, RawInput

logger = structlog.get_logger("ecodiaos.systems.atune.normalisation")


# ---------------------------------------------------------------------------
# Per-channel normaliser definitions
# ---------------------------------------------------------------------------


class ChannelNormaliser:
    """Base normaliser — subclass per channel for custom logic."""

    modality: str = "text"
    default_salience_hint: float = 0.5

    def extract_text(self, raw: RawInput) -> str:
        """Return the plain-text representation of the raw input."""
        if isinstance(raw.data, bytes):
            return raw.data.decode("utf-8", errors="replace")
        return raw.data


class TextChatNormaliser(ChannelNormaliser):
    modality = "text"
    default_salience_hint = 0.6  # User messages are inherently important


class VoiceNormaliser(ChannelNormaliser):
    modality = "audio_transcript"
    default_salience_hint = 0.6


class GestureNormaliser(ChannelNormaliser):
    modality = "interaction"
    default_salience_hint = 0.3


class SensorIoTNormaliser(ChannelNormaliser):
    modality = "sensor"
    default_salience_hint = 0.2


class CalendarNormaliser(ChannelNormaliser):
    modality = "temporal"
    default_salience_hint = 0.4


class ExternalAPINormaliser(ChannelNormaliser):
    modality = "api"
    default_salience_hint = 0.3


class SystemEventNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.4


class MemoryBubbleNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.5


class AffectShiftNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.4


class EvoInsightNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.5


class FederationMsgNormaliser(ChannelNormaliser):
    modality = "federation"
    default_salience_hint = 0.5


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CHANNEL_NORMALISERS: dict[InputChannel, ChannelNormaliser] = {
    InputChannel.TEXT_CHAT: TextChatNormaliser(),
    InputChannel.VOICE: VoiceNormaliser(),
    InputChannel.GESTURE: GestureNormaliser(),
    InputChannel.SENSOR_IOT: SensorIoTNormaliser(),
    InputChannel.CALENDAR: CalendarNormaliser(),
    InputChannel.EXTERNAL_API: ExternalAPINormaliser(),
    InputChannel.SYSTEM_EVENT: SystemEventNormaliser(),
    InputChannel.MEMORY_BUBBLE: MemoryBubbleNormaliser(),
    InputChannel.AFFECT_SHIFT: AffectShiftNormaliser(),
    InputChannel.EVO_INSIGHT: EvoInsightNormaliser(),
    InputChannel.FEDERATION_MSG: FederationMsgNormaliser(),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def normalise(
    raw_input: RawInput,
    channel: InputChannel,
    embed_fn: object,  # async callable (str) -> list[float]
) -> Percept:
    """
    Convert any raw input into a standard :class:`Percept`.

    Parameters
    ----------
    raw_input:
        The raw data arriving on *channel*.
    channel:
        Which input channel produced this data.
    embed_fn:
        Async callable that returns a 768-dim embedding for a text string.

    Returns
    -------
    Percept
        Normalised percept ready for salience scoring.
    """

    normaliser = CHANNEL_NORMALISERS.get(channel)
    if normaliser is None:
        logger.warning("unknown_channel", channel=channel.value)
        normaliser = ChannelNormaliser()

    # Extract text content
    text = normaliser.extract_text(raw_input)

    # Generate embedding
    embedding: list[float] = await embed_fn(text)  # type: ignore[operator]

    # Build provenance
    now = utc_now()
    raw_hash = hash_content(raw_input.data if isinstance(raw_input.data, str) else raw_input.data)
    text_hash = hash_content(text)

    provenance = Provenance(
        chain=[
            TransformRecord(
                step="normalise",
                system="atune",
                timestamp=now,
                input_hash=raw_hash,
                output_hash=text_hash,
            )
        ],
        integrity=compute_hash_chain(
            raw_input.data if isinstance(raw_input.data, str) else raw_input.data,
            text,
        ),
    )

    # Map channel to a SystemID; fall back to ATUNE for unknown channels
    try:
        source_system = SystemID(channel.value)
    except ValueError:
        source_system = SystemID.ATUNE

    # Map modality string to Modality enum
    try:
        source_modality = Modality(normaliser.modality)
    except ValueError:
        source_modality = Modality.TEXT

    percept = Percept(
        id=new_id(),
        timestamp=now,
        source=SourceDescriptor(
            system=source_system,
            channel=raw_input.channel_id or channel.value,
            modality=source_modality,
        ),
        content=Content(
            raw=raw_input.data if isinstance(raw_input.data, str) else raw_input.data.decode("utf-8", errors="replace"),
            parsed={"text": text},
            embedding=embedding,
        ),
        provenance=provenance,
        salience_hint=normaliser.default_salience_hint,
        metadata=raw_input.metadata,
    )

    logger.debug(
        "percept_normalised",
        percept_id=percept.id,
        channel=channel.value,
        text_length=len(text),
    )

    return percept
