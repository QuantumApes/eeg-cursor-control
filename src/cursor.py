"""
Cursor Controller: Maps classified mental states to screen cursor movement.

Supports:
    - Continuous mode: Smooth cursor movement while in a state
    - Discrete mode: Fixed-step movement on state transitions
    - Exponential smoothing to reduce jitter
    - Confidence gating to prevent accidental movement
    - Click actions via sustained focus
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pyautogui

from .classifier import ClassificationResult

logger = logging.getLogger(__name__)

# Safety: don't let pyautogui throw on screen edge
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0  # No delay between actions


@dataclass
class CursorConfig:
    """Cursor control parameters."""
    mode: str = "continuous"            # "continuous" or "discrete"
    speed: int = 10                     # Pixels per update
    smoothing: float = 0.3             # 0 = no smoothing, 1 = max smoothing
    confidence_threshold: float = 0.6   # Min confidence to act
    click_hold_seconds: float = 2.0     # Sustained focus to click
    screen_width: int = 1920
    screen_height: int = 1080


# ── State → Movement mapping ──────────────────────────────────────

@dataclass
class MovementVector:
    """Directional movement command."""
    dx: float = 0.0
    dy: float = 0.0
    click: bool = False


STATE_MOVEMENT = {
    "rest":        MovementVector(0, 0),
    "focus":       MovementVector(0, -1),       # UP
    "relax":       MovementVector(0, 1),         # DOWN
    "left_think":  MovementVector(-1, 0),        # LEFT
    "right_think": MovementVector(1, 0),         # RIGHT
}


class CursorController:
    """
    Translates EEG classifications into cursor actions.

    Usage:
        ctrl = CursorController(CursorConfig())
        ctrl.update(classification_result)  # Call at ~30 Hz
    """

    def __init__(self, config: Optional[CursorConfig] = None):
        self.config = config or CursorConfig()

        # Current smoothed position delta
        self._smooth_dx = 0.0
        self._smooth_dy = 0.0

        # Click detection
        self._focus_start: Optional[float] = None
        self._last_state = "rest"

        # Metrics
        self._updates = 0
        self._movements = 0
        self._clicks = 0

        # Position tracking (for headless/test mode)
        self._virtual_x = self.config.screen_width / 2
        self._virtual_y = self.config.screen_height / 2
        self._headless = False

        logger.info(
            "CursorController: mode=%s, speed=%d, threshold=%.2f",
            self.config.mode, self.config.speed, self.config.confidence_threshold,
        )

    # ── Main update loop ──────────────────────────────────────────

    def update(self, result: ClassificationResult) -> dict:
        """
        Process a classification result and move cursor accordingly.

        Args:
            result: ClassificationResult from the classifier

        Returns:
            Status dict with position, movement, confidence info
        """
        self._updates += 1

        state = result.predicted_class
        confidence = result.confidence

        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            state = "rest"  # Gate low-confidence predictions

        # Get base movement vector
        movement = STATE_MOVEMENT.get(state, MovementVector(0, 0))

        # Scale by speed and confidence
        raw_dx = movement.dx * self.config.speed * confidence
        raw_dy = movement.dy * self.config.speed * confidence

        # Apply exponential smoothing
        alpha = 1.0 - self.config.smoothing
        self._smooth_dx = alpha * raw_dx + self.config.smoothing * self._smooth_dx
        self._smooth_dy = alpha * raw_dy + self.config.smoothing * self._smooth_dy

        # Apply movement
        dx = int(round(self._smooth_dx))
        dy = int(round(self._smooth_dy))

        if dx != 0 or dy != 0:
            self._move(dx, dy)
            self._movements += 1

        # Click detection: sustained focus
        click = self._check_click(state)
        if click:
            self._click()
            self._clicks += 1

        self._last_state = state

        return {
            "state": state,
            "confidence": confidence,
            "dx": dx,
            "dy": dy,
            "click": click,
            "position": self.position,
            "total_movements": self._movements,
            "total_clicks": self._clicks,
        }

    # ── Click detection ───────────────────────────────────────────

    def _check_click(self, state: str) -> bool:
        """Detect click via sustained 'focus' state."""
        if state == "focus":
            if self._focus_start is None:
                self._focus_start = time.time()
            elif time.time() - self._focus_start >= self.config.click_hold_seconds:
                self._focus_start = None
                return True
        else:
            self._focus_start = None
        return False

    # ── Movement execution ────────────────────────────────────────

    def _move(self, dx: int, dy: int) -> None:
        """Move cursor by (dx, dy) pixels."""
        if self._headless:
            self._virtual_x = max(0, min(
                self.config.screen_width, self._virtual_x + dx
            ))
            self._virtual_y = max(0, min(
                self.config.screen_height, self._virtual_y + dy
            ))
        else:
            try:
                pyautogui.moveRel(dx, dy, _pause=False)
            except Exception as e:
                logger.warning("pyautogui move failed: %s", e)

    def _click(self) -> None:
        """Execute a mouse click."""
        logger.info("CLICK at position %s", self.position)
        if not self._headless:
            try:
                pyautogui.click(_pause=False)
            except Exception as e:
                logger.warning("pyautogui click failed: %s", e)

    # ── Properties ────────────────────────────────────────────────

    @property
    def position(self) -> tuple[int, int]:
        """Current cursor position."""
        if self._headless:
            return (int(self._virtual_x), int(self._virtual_y))
        try:
            pos = pyautogui.position()
            return (pos.x, pos.y)
        except Exception:
            return (0, 0)

    def set_headless(self, headless: bool = True) -> None:
        """Enable headless mode (no actual cursor movement)."""
        self._headless = headless
        logger.info("Headless mode: %s", headless)

    def center_cursor(self) -> None:
        """Move cursor to screen center."""
        cx = self.config.screen_width // 2
        cy = self.config.screen_height // 2
        if self._headless:
            self._virtual_x = cx
            self._virtual_y = cy
        else:
            pyautogui.moveTo(cx, cy)

    @property
    def stats(self) -> dict:
        return {
            "total_updates": self._updates,
            "total_movements": self._movements,
            "total_clicks": self._clicks,
            "movement_rate": self._movements / max(self._updates, 1),
        }
