"""
bedrock_agi/constitution/agenda.py

Agenda: Time-slotted execution queue.
Manages the scheduling and dispatch of approved goals.
"""

import time
from typing import List, Optional, Dict

class Agenda:
    """FIFO queue with time slots and status tracking."""
    
    def __init__(self):
        self.slots: List[Dict] = []
    
    def schedule(self, gid: int, start_time: Optional[float] = None, deadline: Optional[float] = None) -> int:
        """
        Add goal to agenda.
        
        Args:
            gid: Goal ID from Ledger
            start_time: Unix timestamp to start (default: now)
            deadline: Unix timestamp to finish (default: now + 1hr)
            
        Returns:
            slot_id: The index of the created slot
        """
        now = time.time()
        start = start_time if start_time is not None else now
        # Default deadline is 1 hour if not specified
        end = deadline if deadline is not None else (now + 3600)
        
        slot = {
            'slot_id': len(self.slots),
            'gid': gid,
            'start': start,
            'deadline': end,
            'status': 'pending' # pending, running, complete, failed
        }
        self.slots.append(slot)
        return slot['slot_id']
    
    def next_ready(self) -> Optional[Dict]:
        """
        Get next ready slot.
        A slot is ready if:
        1. Status is 'pending'
        2. Current time >= start time
        3. Current time < deadline
        """
        now = time.time()
        for slot in self.slots:
            if slot['status'] == 'pending':
                if slot['start'] <= now < slot['deadline']:
                    return slot
        return None
    
    def mark_status(self, slot_id: int, status: str):
        """Update slot status (complete, failed, running)."""
        if 0 <= slot_id < len(self.slots):
            self.slots[slot_id]['status'] = status

if __name__ == "__main__":
    print("Testing Agenda...")
    
    agenda = Agenda()
    
    # Test 1: Immediate Schedule
    sid = agenda.schedule(gid=0)
    assert sid == 0
    print("✓ Scheduling works")
    
    # Test 2: Retrieval
    slot = agenda.next_ready()
    assert slot is not None
    assert slot['gid'] == 0
    print("✓ Next ready works")
    
    # Test 3: Completion
    agenda.mark_status(sid, 'complete')
    assert agenda.slots[0]['status'] == 'complete'
    
    # Test 4: Empty check
    slot_empty = agenda.next_ready()
    assert slot_empty is None
    print("✓ Status update works")
    
    # Test 5: Future Schedule
    future_sid = agenda.schedule(gid=1, start_time=time.time() + 3600)
    slot_future = agenda.next_ready()
    assert slot_future is None
    print("✓ Future scheduling respects time")
    
    print("✓ Agenda operational")