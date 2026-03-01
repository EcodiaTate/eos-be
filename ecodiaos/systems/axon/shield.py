"""
EcodiaOS — Transaction Shield (Layer 1: Economic Immune System)

Pre-execution filter that evaluates transactions before they are broadcast
on-chain. Unlike sentinels (which detect and report), the shield PREVENTS
bad transactions from ever reaching the network.

Checks performed:
  1. Destination address blacklist
  2. Simulated slippage enforcement (max 50 bps)
  3. Gas cost vs expected ROI validation
  4. MEV risk heuristics (optional, via eth_call simulation)

The shield is wired into ExecutionPipeline as Stage 5.5 -- after context
assembly but before step execution. It only activates for financial
executors (defi_yield, wallet_transfer).
"""

from __future__ import annotations

from typing import Any

import structlog

from ecodiaos.systems.thymos.types import (
    AddressBlacklistEntry,
    SimulationResult,
)

logger = structlog.get_logger()

# Maximum slippage the shield will permit (basis points)
_DEFAULT_MAX_SLIPPAGE_BPS: int = 50

# Minimum expected ROI (USD) to justify gas costs
_MIN_ROI_TO_GAS_RATIO: float = 2.0

# Executors that require shield evaluation
SHIELDED_EXECUTORS: frozenset[str] = frozenset({"defi_yield", "wallet_transfer"})


class TransactionShield:
    """
    Pre-execution transaction filter.

    Evaluates financial transactions before broadcast. Rejects transactions
    that fail blacklist checks, slippage limits, or gas/ROI analysis.

    NOT a sentinel -- it prevents, not detects. Lives in Axon, not in the
    immune system module.
    """

    def __init__(
        self,
        wallet: Any = None,
        max_slippage_bps: int = _DEFAULT_MAX_SLIPPAGE_BPS,
    ) -> None:
        self._wallet = wallet
        self._max_slippage_bps = max_slippage_bps
        self._blacklist: dict[str, AddressBlacklistEntry] = {}
        self._logger = logger.bind(system="axon", component="transaction_shield")

        # Metrics
        self._total_evaluated: int = 0
        self._total_rejected: int = 0

    # -- Public API ----------------------------------------------------------

    async def evaluate(
        self,
        action_type: str,
        params: dict[str, Any],
        context: Any = None,
    ) -> SimulationResult:
        """
        Evaluate a financial transaction before execution.

        Returns a SimulationResult. If ``passed`` is False, the pipeline
        should abort with TRANSACTION_SHIELD_REJECTED.

        Only evaluates executors in SHIELDED_EXECUTORS. Others pass through.
        """
        if action_type not in SHIELDED_EXECUTORS:
            return SimulationResult(passed=True)

        self._total_evaluated += 1
        warnings: list[str] = []

        # -- Check 1: Blacklist -----------------------------------------------
        destination = self._extract_destination(action_type, params)
        chain_id = params.get("chain_id", 8453)

        if destination and self.is_blacklisted(destination, chain_id):
            self._total_rejected += 1
            entry = self._blacklist.get(self._blacklist_key(destination, chain_id))
            reason = entry.reason if entry else "unknown"
            self._logger.warning(
                "shield_blacklisted_address",
                address=destination,
                reason=reason,
            )
            return SimulationResult(
                passed=False,
                revert_reason=f"Destination {destination} is blacklisted: {reason}",
                warnings=[f"Blacklisted address: {destination}"],
            )

        # -- Check 2: Slippage enforcement ------------------------------------
        slippage_bps = params.get("max_slippage_bps", 0)
        if slippage_bps > self._max_slippage_bps:
            self._total_rejected += 1
            self._logger.warning(
                "shield_slippage_exceeded",
                requested_bps=slippage_bps,
                max_bps=self._max_slippage_bps,
            )
            return SimulationResult(
                passed=False,
                slippage_bps=slippage_bps,
                revert_reason=(
                    f"Slippage {slippage_bps} bps exceeds maximum "
                    f"{self._max_slippage_bps} bps"
                ),
                warnings=[f"Slippage {slippage_bps}bps > {self._max_slippage_bps}bps cap"],
            )

        # -- Check 3: Gas cost vs expected ROI --------------------------------
        expected_roi_usd = float(params.get("expected_roi_usd", 0))
        estimated_gas_usd = float(params.get("estimated_gas_usd", 0))

        if estimated_gas_usd > 0 and expected_roi_usd > 0:
            ratio = expected_roi_usd / estimated_gas_usd
            if ratio < _MIN_ROI_TO_GAS_RATIO:
                self._total_rejected += 1
                self._logger.warning(
                    "shield_gas_exceeds_roi",
                    roi_usd=expected_roi_usd,
                    gas_usd=estimated_gas_usd,
                    ratio=ratio,
                )
                return SimulationResult(
                    passed=False,
                    gas_used=int(estimated_gas_usd * 1e6),  # approximate
                    value_delta_usd=expected_roi_usd - estimated_gas_usd,
                    revert_reason=(
                        f"Gas cost ${estimated_gas_usd:.4f} exceeds ROI "
                        f"${expected_roi_usd:.4f} (ratio={ratio:.2f}, "
                        f"minimum={_MIN_ROI_TO_GAS_RATIO})"
                    ),
                    warnings=["Gas cost exceeds expected return"],
                )

        # -- Check 4: Simulation (best-effort) --------------------------------
        if destination and self._wallet is not None:
            sim_result = await self._try_simulate(destination, params)
            if sim_result is not None and not sim_result.passed:
                self._total_rejected += 1
                return sim_result

        # -- All checks passed ------------------------------------------------
        if warnings:
            self._logger.info(
                "shield_passed_with_warnings",
                warnings=warnings,
                action_type=action_type,
            )

        return SimulationResult(
            passed=True,
            slippage_bps=slippage_bps,
            warnings=warnings,
        )

    def is_blacklisted(self, address: str, chain_id: int = 8453) -> bool:
        """Check if an address is on the blacklist."""
        key = self._blacklist_key(address, chain_id)
        return key in self._blacklist

    def add_to_blacklist(self, entry: AddressBlacklistEntry) -> None:
        """Add an address to the blacklist."""
        key = self._blacklist_key(entry.address, entry.chain_id)
        self._blacklist[key] = entry
        self._logger.info(
            "shield_address_blacklisted",
            address=entry.address,
            chain_id=entry.chain_id,
            reason=entry.reason,
            threat_type=entry.threat_type,
        )

    # -- Internal ------------------------------------------------------------

    def _extract_destination(
        self,
        action_type: str,
        params: dict[str, Any],
    ) -> str:
        """Extract the destination address from executor params."""
        if action_type == "wallet_transfer":
            return params.get("to", params.get("destination", ""))
        if action_type == "defi_yield":
            return params.get("protocol_address", params.get("pool_address", ""))
        return ""

    @staticmethod
    def _blacklist_key(address: str, chain_id: int) -> str:
        return f"{address.lower()}:{chain_id}"

    async def _try_simulate(
        self,
        to: str,
        params: dict[str, Any],
    ) -> SimulationResult | None:
        """
        Attempt to simulate the transaction via eth_call.

        Best-effort: returns None if simulation is not available (e.g., the
        RPC node does not support state overrides). The shield does NOT block
        transactions just because simulation fails -- it only blocks if
        simulation explicitly reveals a problem.
        """
        try:
            # If the wallet client exposes a simulate method, use it
            if hasattr(self._wallet, "simulate_transaction"):
                result = await self._wallet.simulate_transaction(to=to, params=params)
                if result and isinstance(result, dict):
                    if result.get("reverted"):
                        return SimulationResult(
                            passed=False,
                            revert_reason=result.get("revert_reason", "Transaction would revert"),
                            gas_used=result.get("gas_used", 0),
                            warnings=["Simulation detected revert"],
                        )
            return None
        except Exception as exc:
            # Simulation failure is not a rejection -- log and continue
            self._logger.debug(
                "shield_simulation_unavailable",
                error=str(exc),
            )
            return None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_evaluated": self._total_evaluated,
            "total_rejected": self._total_rejected,
            "blacklist_size": len(self._blacklist),
            "max_slippage_bps": self._max_slippage_bps,
        }
