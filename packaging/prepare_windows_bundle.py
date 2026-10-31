from pathlib import Path
root=Path(__file__).resolve().parent
folder=root/'dist-windows/PossibilityLab';app=folder/'PossibilityLab.exe';engine=folder/'Engine.exe'
if app.exists() and app.stat().st_size>1_000_000:
 app.replace(engine)
assert engine.exists()
print('Prepared engine binary for the offline desktop launcher.')
