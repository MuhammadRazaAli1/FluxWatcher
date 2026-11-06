"""Online evaluation against the simulator's ground truth."""
from __future__ import annotations

from typing import Dict, List, Optional


def _ratio(num: float, den: float) -> Optional[float]:
    return None if den <= 0 else num / den


class OnlineMetrics:
    """Point-wise and event-wise detection metrics, updated tick by tick.

    * Point-wise: confusion matrix over ticks. Ticks within ``grace`` ticks after an
      episode are ignored because the signal is still recovering from the anomaly.
    * Event-wise: an episode is *detected* if an alert is active at any tick of it;
      the detection delay is the number of ticks until the first alert.
      A *false alert event* is an alert that starts outside any episode / grace period.
    """

    def __init__(self, grace: int = 10):
        self.grace = grace
        self.reset()

    def reset(self) -> None:
        self.ticks = 0
        self.tp = self.fp = self.fn = self.tn = 0
        self.episodes = 0
        self.detected = 0
        self.delays: List[int] = []
        self.false_alert_events = 0
        self.alert_events = 0
        self._prev_label = 0
        self._prev_alert = False
        self._ep_start = 0
        self._ep_detected = False
        self._grace_left = 0

    def update(self, label: int, alert: bool, t: Optional[int] = None) -> None:
        t = self.ticks if t is None else t
        self.ticks += 1
        label = 1 if label else 0
        alert = bool(alert)

        if label and not self._prev_label:            # episode starts
            self._ep_start, self._ep_detected = t, False
        if label and alert and not self._ep_detected:
            self._ep_detected = True
            self.delays.append(t - self._ep_start)
        if not label and self._prev_label:            # episode ends
            self.episodes += 1
            self.detected += int(self._ep_detected)
            self._grace_left = self.grace

        if alert and not self._prev_alert:            # alert rising edge
            self.alert_events += 1
            if not label and self._grace_left <= 0:
                self.false_alert_events += 1

        if label:
            if alert:
                self.tp += 1
            else:
                self.fn += 1
        elif self._grace_left > 0:
            self._grace_left -= 1                     # recovering: ignored
        elif alert:
            self.fp += 1
        else:
            self.tn += 1

        self._prev_label, self._prev_alert = label, alert

    def snapshot(self) -> Dict[str, object]:
        precision = _ratio(self.tp, self.tp + self.fp)
        recall = _ratio(self.tp, self.tp + self.fn)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        mean_delay = sum(self.delays) / len(self.delays) if self.delays else None
        return {
            "ticks": self.ticks,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "episodes": self.episodes,
            "detected": self.detected,
            "event_recall": _ratio(self.detected, self.episodes),
            "mean_delay": mean_delay,
            "alert_events": self.alert_events,
            "false_alert_events": self.false_alert_events,
        }
