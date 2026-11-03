"""Build one auto project:  python3 auto/build.py SONG_ID THEME OUTDIR VSCO_DIR
THEME: kitchen | warship | concert. Produces OUTDIR/{scene.py,events.json,notes.json,music.wav,meta.json,config.json}."""
import sys, os, json, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sid, theme, out, vsco = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
subprocess.run([sys.executable, os.path.join(HERE, 'make_song.py'), sid, out, theme], check=True)
palette = 'band' if theme == 'warship' else 'orchestra'
subprocess.run([sys.executable, os.path.join(HERE, 'render_audio.py'), vsco, palette, out], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', os.path.join(out, 'music_raw.wav'), '-af',
                'acompressor=threshold=-16dB:ratio=2.5:attack=8:release=120,loudnorm=I=-14:TP=-1.2:LRA=10', '-ar', '44100', '-t', '66',
                os.path.join(out, 'music.wav')], check=True)
os.remove(os.path.join(out, 'music_raw.wav')); shutil.rmtree(os.path.join(out, 'work'), ignore_errors=True)
shutil.copy(os.path.join(HERE, 'themes', theme + '.py'), os.path.join(out, 'scene.py'))
json.dump({"nframes": 1980, "chunk": 60, "samp": 5, "pct": 75,
           "vf": "scale=1080:1920:flags=lanczos,format=gbrp,split[a][b];[b]gblur=sigma=6[g];[a][g]blend=all_mode=screen:all_opacity=0.35,format=yuv420p"},
          open(os.path.join(out, 'config.json'), 'w'))
print('built', out)
