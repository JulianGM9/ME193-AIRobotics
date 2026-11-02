"""Bookkeeping for the live sample stream: gaps and board restarts."""


class SeqTracker:
    """Counts received and missing samples from their seq numbers.

    A new run id, or seq going backwards, means the board app or the sketch restarted:
    that is counted as a restart, not as a gap.
    """

    def __init__(self):
        self.run = None
        self.last_seq = None
        self.received = 0
        self.missing = 0
        self.restarts = 0

    def update(self, run, seqs) -> int:
        """Feed one batch's seq numbers. Returns how many samples were missing before it."""
        new_missing = 0
        if run != self.run or (self.last_seq is not None and seqs[0] <= self.last_seq):
            if self.run is not None:
                self.restarts += 1
            self.run = run
            self.last_seq = None
        for seq in seqs:
            seq = int(seq)
            if self.last_seq is not None and seq > self.last_seq + 1:
                new_missing += seq - self.last_seq - 1
            self.last_seq = seq
        self.received += len(seqs)
        self.missing += new_missing
        return new_missing
