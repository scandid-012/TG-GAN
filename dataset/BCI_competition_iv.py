dataset_path = '../input/bci-competition-iv-dataset-2b/' # Adjust if the folder structure is different
gdf_files = glob.glob(os.path.join(dataset_path, '*.gdf'))
event_files = glob.glob(os.path.join(dataset_path, '*events.txt')) # Or whatever the event file extension is
info_files = glob.glob(os.path.join(dataset_path, '*.txt')) # Look for any readme or info files

print("GDF Files found:", gdf_files)
print("Event Files found (example name):", event_files) # Event files might be named differently or embedded
print("Other TXT/Info Files found:", info_files)
