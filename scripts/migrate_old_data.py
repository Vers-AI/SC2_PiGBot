"""
Migrate old rush detection logs to include scout timing features.

Adds last_nat_scout_time=-1 and nat_present_on_last_scout=-1 to entries missing them.
"""

import json
from pathlib import Path

LOG_FILE = Path("data/rush_detection_log.jsonl")

def migrate():
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}")
        return
    
    lines = LOG_FILE.read_text().splitlines()
    migrated = []
    updated_count = 0
    
    for line in lines:
        entry = json.loads(line)
        
        # Add missing scout timing features if not present
        if "last_nat_scout_time" not in entry:
            entry["last_nat_scout_time"] = -1
            entry["nat_present_on_last_scout"] = -1
            updated_count += 1
        
        migrated.append(json.dumps(entry))
    
    # Write back
    LOG_FILE.write_text('\n'.join(migrated) + '\n')
    
    print(f"Migration complete!")
    print(f"  Total entries: {len(lines)}")
    print(f"  Updated: {updated_count}")
    print(f"  Already had scout features: {len(lines) - updated_count}")

if __name__ == "__main__":
    migrate()
