"""
EcodiaOS — Oikos & Identity REST Router

Exposes the organism's economic state, active organs, certificate status,
and deployed assets to the Next.js frontend.

Endpoints:
  GET /api/v1/oikos/state       — Full economic snapshot (net worth, BMR, runway)
  GET /api/v1/oikos/organs      — Active economic organs from OrganLifecycleManager
  GET /api/v1/oikos/certificate — Identity certificate status and days until expiry
  GET /api/v1/simula/assets     — Deployed OwnedAssets from the AssetFactory
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request

logger = structlog.get_logger("ecodiaos.api.oikos")

router = APIRouter()


@router.get("/api/v1/oikos/state")
async def get_oikos_state(request: Request) -> dict[str, Any]:
    """Return the organism's current economic snapshot."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    state = oikos.snapshot()
    return {
        "status": "ok",
        "data": {
            "total_net_worth": str(state.total_net_worth),
            "liquid_balance": str(state.liquid_balance),
            "survival_reserve": str(state.survival_reserve),
            "survival_reserve_target": str(state.survival_reserve_target),
            "total_deployed": str(state.total_deployed),
            "total_receivables": str(state.total_receivables),
            "total_asset_value": str(state.total_asset_value),
            "total_fleet_equity": str(state.total_fleet_equity),
            "derivative_liabilities": str(state.derivative_liabilities),
            "bmr_usd_per_hour": str(state.basal_metabolic_rate.usd_per_hour),
            "bmr_usd_per_day": str(state.basal_metabolic_rate.usd_per_day),
            "burn_rate_usd_per_hour": str(state.current_burn_rate.usd_per_hour),
            "runway_hours": str(state.runway_hours),
            "runway_days": str(state.runway_days),
            "starvation_level": state.starvation_level.value,
            "is_metabolically_positive": state.is_metabolically_positive,
            "metabolic_efficiency": str(state.metabolic_efficiency),
            "revenue_24h": str(state.revenue_24h),
            "revenue_7d": str(state.revenue_7d),
            "revenue_30d": str(state.revenue_30d),
            "costs_24h": str(state.costs_24h),
            "costs_7d": str(state.costs_7d),
            "costs_30d": str(state.costs_30d),
            "net_income_24h": str(state.net_income_24h),
            "net_income_7d": str(state.net_income_7d),
            "net_income_30d": str(state.net_income_30d),
            "economic_free_energy": str(state.economic_free_energy),
            "survival_probability_30d": str(state.survival_probability_30d),
            "instance_id": state.instance_id,
            "timestamp": state.timestamp.isoformat(),
        },
    }


@router.get("/api/v1/oikos/organs")
async def get_oikos_organs(request: Request) -> dict[str, Any]:
    """Return active economic organs from the OrganLifecycleManager."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    active = oikos._morphogenesis.active_organs
    all_organs = oikos._morphogenesis.all_organs

    return {
        "status": "ok",
        "data": {
            "active_count": len(active),
            "total_count": len(all_organs),
            "organs": [
                {
                    "organ_id": organ.organ_id,
                    "category": organ.category.value,
                    "specialisation": organ.specialisation,
                    "maturity": organ.maturity.value,
                    "resource_allocation_pct": str(organ.resource_allocation_pct),
                    "revenue_30d": str(organ.revenue_30d),
                    "cost_30d": str(organ.cost_30d),
                    "efficiency": str(organ.efficiency),
                    "created_at": organ.created_at.isoformat(),
                    "last_revenue_at": organ.last_revenue_at.isoformat() if organ.last_revenue_at else None,
                }
                for organ in all_organs
            ],
        },
    }


@router.get("/api/v1/oikos/certificate")
async def get_oikos_certificate(request: Request) -> dict[str, Any]:
    """Return the organism's identity certificate status."""
    cert_mgr = request.app.state.certificate_manager
    if cert_mgr is None:
        return {
            "status": "ok",
            "data": {
                "initialized": False,
                "has_certificate": False,
                "certificate_status": None,
                "remaining_days": -1.0,
            },
        }

    cert = cert_mgr.certificate
    return {
        "status": "ok",
        "data": {
            "initialized": True,
            "has_certificate": cert is not None,
            "is_certified": cert_mgr.is_certified,
            "certificate_status": cert.status.value if cert else None,
            "remaining_days": cert_mgr.certificate_remaining_days,
            "certificate_type": cert.certificate_type.value if cert else None,
            "instance_id": cert.instance_id if cert else None,
            "issued_at": cert.issued_at.isoformat() if cert else None,
            "expires_at": cert.expires_at.isoformat() if cert else None,
            "lineage_hash": cert_mgr.lineage_hash,
        },
    }


@router.get("/api/v1/simula/assets")
async def get_simula_assets(request: Request) -> dict[str, Any]:
    """Return deployed OwnedAssets from the AssetFactory."""
    oikos = request.app.state.oikos
    if oikos is None:
        return {"status": "unavailable", "error": "Oikos not initialized"}

    factory = oikos.asset_factory
    live = factory.get_live_assets()
    building = factory.get_building_assets()

    def _serialize_asset(asset: Any) -> dict[str, Any]:
        return {
            "asset_id": asset.asset_id,
            "name": asset.name,
            "description": asset.description,
            "asset_type": asset.asset_type,
            "status": asset.status.value,
            "estimated_value_usd": str(asset.estimated_value_usd),
            "development_cost_usd": str(asset.development_cost_usd),
            "monthly_revenue_usd": str(asset.monthly_revenue_usd),
            "monthly_cost_usd": str(asset.monthly_cost_usd),
            "total_revenue_usd": str(asset.total_revenue_usd),
            "total_cost_usd": str(asset.total_cost_usd),
            "projected_break_even_days": asset.projected_break_even_days,
            "break_even_reached": asset.break_even_reached,
            "deployed_at": asset.deployed_at.isoformat() if asset.deployed_at else None,
        }

    return {
        "status": "ok",
        "data": {
            "live_count": len(live),
            "building_count": len(building),
            "live": [_serialize_asset(a) for a in live],
            "building": [_serialize_asset(a) for a in building],
        },
    }
