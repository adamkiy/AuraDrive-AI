"""Ten-minute history, exact agent rendezvous and final-decision audit logs."""
from __future__ import annotations
import asyncio, json, time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

WINDOW_MS = 600_000

class SensorDB:
    def __init__(self, path: str = "sensor_log.jsonl") -> None:
        self._records: deque[Dict[str, Any]] = deque()
        self._lock = asyncio.Lock()
        self._fh = Path(path).open("a", encoding="utf-8")
    async def insert(self, record: Dict[str, Any]) -> None:
        item = dict(record)
        if "frame_id" not in item or "timestamp_ms" not in item: raise ValueError("frame_id and timestamp_ms required")
        async with self._lock:
            self._records.append(item); self._fh.write(json.dumps(item, ensure_ascii=False)+"\n"); self._fh.flush()
            cutoff = int(time.time()*1000)-WINDOW_MS
            while self._records and int(self._records[0]["timestamp_ms"]) < cutoff: self._records.popleft()
    async def window(self) -> List[Dict[str, Any]]:
        async with self._lock: return [dict(x) for x in self._records]
    def close(self) -> None:
        try: self._fh.close()
        except Exception: pass

class AgentDecisionLog:
    def __init__(self, path: str = "agent_rendezvous_log.jsonl") -> None:
        self._entries: Dict[int, Dict[str, Any]] = {}
        self._condition = asyncio.Condition(); self._fh = Path(path).open("a", encoding="utf-8")
    async def add(self, frame_id: int, status: str, decision: Dict[str, Any]) -> None:
        entry={"frame_id":int(frame_id),"status":str(status),"decision":dict(decision),"decided_at_ms":int(time.time()*1000)}
        async with self._condition:
            self._entries[int(frame_id)] = entry
            self._fh.write(json.dumps(entry, ensure_ascii=False)+"\n"); self._fh.flush(); self._condition.notify_all()
    async def wait(self, frame_id: int, timeout_s: float) -> Optional[Dict[str, Any]]:
        deadline=time.monotonic()+timeout_s
        async with self._condition:
            while int(frame_id) not in self._entries:
                remaining=deadline-time.monotonic()
                if remaining<=0: return None
                try: await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError: return None
            return dict(self._entries[int(frame_id)])
    def close(self) -> None:
        try: self._fh.close()
        except Exception: pass

class FinalDecisionLog:
    def __init__(self, path: str = "final_decision_log.jsonl") -> None: self._fh=Path(path).open("a",encoding="utf-8"); self._lock=asyncio.Lock()
    async def add(self, decision: Dict[str, Any], *, frame_id: Optional[int], event: str) -> None:
        row={"timestamp_ms":int(time.time()*1000),"frame_id":frame_id,"event":event,"decision":dict(decision)}
        async with self._lock: self._fh.write(json.dumps(row,ensure_ascii=False)+"\n"); self._fh.flush()
    def close(self)->None:
        try:self._fh.close()
        except Exception:pass
