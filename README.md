# music-machine-videos

音楽に合わせて動くCG動画（Blender / Python）を、GitHub Actionsで分割レンダリングして作るリポジトリです。

- `projects/<名前>/scene.py` … Blenderのシーン（`python scene.py -- range 開始 終了` でフレームを描画）
- `projects/<名前>/events.json` … 音符のタイミング（映像と音の同期用）
- `projects/<名前>/music.wav` … 音
- `projects/<名前>/config.json` … フレーム数・分割サイズ・画質
- `requests/*.json` に `{"project": "<名前>"}` を追加してpushすると、レンダリングが始まります
- 完成した動画は `outputs/<名前>.mp4` に自動で追加されます

楽曲はすべてパブリックドメイン（著作権切れ）の作品を編曲しています。
