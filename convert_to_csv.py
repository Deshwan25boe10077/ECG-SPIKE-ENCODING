import wfdb
import pandas as pd

# ================== CONFIGURATION ==================
record_name = "210"          # Your record
# ===================================================

print(f"Reading record: {record_name}...")

record = wfdb.rdrecord(record_name, physical=True)

df = pd.DataFrame(record.p_signal, columns=record.sig_name)

sampling_rate = record.fs
df.insert(0, 'time', [i / sampling_rate for i in range(len(df))])

output_file = f"{record_name}.csv"
df.to_csv(output_file, index=False)

print(f"✅ SUCCESS! Created {output_file}")
print(f"   Rows: {len(df):,}")
print(f"   Channels: {record.sig_name}")
print(f"   Duration: {len(df)/sampling_rate:.2f} seconds")