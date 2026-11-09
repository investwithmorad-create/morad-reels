# Morad 3D map engine (Map gear, Morad Pulse 2.0)

Renders a 9:16 3D fly-in over real satellite imagery for Ahmed Morad's map and launch reels:

- camera push-in from a city view to the plot, tilt and orbit
- plot boundary draws itself in gold with glow; everything outside dims
- illustrative tower massing rises with a soft shadow (always labelled "Illustrative")
- arc route lines to growth drivers, one at a time, with gold pins and dark-glass labels
- Pulse label (Bebas title, gold rule, Montserrat sub-line), Esri credit, safe-zone layout

## Run

```
pip install --break-system-packages numpy opencv-python-headless pillow
python3 map3d.py scene.json out.mp4 --draft     # 540x960 check
python3 map3d.py scene.json out.mp4             # 1080x1920 final
python3 map3d.py scene.json x --stills=2.5,6.0  # still frames only
```

Imagery tiles (Esri World Imagery, server.arcgisonline.com) download into `cache/` on first run.

## Scene file (see scene_demo.json)

- `origin`: [lon, lat] of the plot (local metres are centred here)
- `layers`: imagery, finest first: `{bbox:[lon0,lat0,lon1,lat1], z}`. Use z17 around the plot, z15 for the district, z13 for the city.
- `plot`: `{polygon:[[lon,lat],...], label}` — boundary from the developer's plan only, never guessed
- `tower`: `{footprint:[[lon,lat],...], height}` — illustrative massing; omit for no tower
- `pins`: `{name, lon, lat, t}` — t = second the route starts
- `label`: `{title, sub, y}` — y 400 for full-screen maps so the label never covers the plot
- `events`: `highlight [in0,in1,out0,out1]`, `boundary [t0,t1]`, `tower [t0,t1]`, `plot_label t`, `label [t0,t1]`
- `camera`: keyframes `{t, lon, lat, dist (m), heading (deg, 0 = north), pitch (deg from straight down)}`
- `note`: small gold note, e.g. "Illustrative massing"

## Limits

- Real 3D buildings for the surrounding city need building-height data (OpenStreetMap via overpass-api.de), which this workspace cannot reach. Allowing that domain in claude.ai Settings > Capabilities would let surrounding towers rise from real heights.
- Esri imagery: credit "Imagery: Esri, Maxar" on screen; check the licence before paid ads.
