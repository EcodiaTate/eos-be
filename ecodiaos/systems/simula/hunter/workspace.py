"""
EcodiaOS — Hunter Target Workspace

Abstraction for a target codebase (internal EOS or externally cloned repo).
Handles lifecycle: clone → analyze → cleanup.

Iron Rule: Hunter NEVER writes to the EOS source tree. All external targets
live in temporary directories that are cleaned up after the hunt.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger().bind(system="simula.hunter.workspace")


class TargetWorkspace:
    """Abstraction for a target codebase (internal or external)."""

    def __init__(
        self,
        root: Path,
        workspace_type: Literal["internal_eos", "external_repo"],
        temp_directory: Path | None = None,
    ) -> None:
        """
        Initialize workspace.

        Args:
            root: Absolute path to the codebase root.
            workspace_type: Whether this is the internal EOS tree or a cloned repo.
            temp_directory: If set, the parent temp dir to clean up on exit.

        Raises:
            FileNotFoundError: If root does not exist.
            NotADirectoryError: If root is not a directory.
        """
        resolved = root.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Workspace root is not a directory: {resolved}")

        self.root = resolved
        self.workspace_type = workspace_type
        self.temp_directory = temp_directory

        logger.info(
            "workspace_created",
            root=str(self.root),
            workspace_type=self.workspace_type,
            is_temp=self.temp_directory is not None,
        )

    @property
    def is_external(self) -> bool:
        """True if this workspace targets an external (non-EOS) codebase."""
        return self.workspace_type == "external_repo"

    def cleanup(self) -> None:
        """Remove temp directory if present. No-op for internal workspaces."""
        if self.temp_directory is not None and self.temp_directory.exists():
            shutil.rmtree(self.temp_directory, ignore_errors=True)
            logger.info(
                "workspace_cleaned_up",
                temp_directory=str(self.temp_directory),
            )

    @classmethod
    async def from_github_url(cls, github_url: str) -> TargetWorkspace:
        """
        Clone a GitHub repo into a temp directory and return a TargetWorkspace.

        Args:
            github_url: HTTPS URL of the repository to clone.

        Returns:
            A TargetWorkspace pointing at the cloned repo root.

        Raises:
            RuntimeError: If git clone fails.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="hunter_"))
        clone_target = temp_dir / "repo"

        logger.info("cloning_repo", url=github_url, target=str(clone_target))

        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", github_url, str(clone_target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Clean up on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"git clone failed (exit {proc.returncode}): {error_msg}"
            )

        logger.info("clone_complete", url=github_url, root=str(clone_target))

        return cls(
            root=clone_target,
            workspace_type="external_repo",
            temp_directory=temp_dir,
        )

    @classmethod
    def from_local_path(cls, path: Path) -> TargetWorkspace:
        """
        Create a workspace from an existing local directory.

        Useful for analyzing local repos without cloning.
        """
        return cls(
            root=path,
            workspace_type="external_repo",
            temp_directory=None,
        )

    @classmethod
    def internal(cls, eos_root: Path) -> TargetWorkspace:
        """Create a workspace pointing at the internal EOS codebase."""
        return cls(
            root=eos_root,
            workspace_type="internal_eos",
            temp_directory=None,
        )

    def __repr__(self) -> str:
        return (
            f"TargetWorkspace(root={self.root!r}, "
            f"type={self.workspace_type!r}, "
            f"temp={self.temp_directory is not None})"
        )
