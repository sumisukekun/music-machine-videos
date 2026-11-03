"""TikTok Content Posting API helper for the Ball Music Studio account (own account only).
Tokens are kept encrypted in tokens.enc (AES via openssl, key = secret TOKEN_KEY).
  python3 tools/tiktok.py connect CODE      # exchange an OAuth code from docs/callback.html
  python3 tools/tiktok.py post VIDEO CAPTION # refresh token, read creator info, upload + publish
"""
import os, sys, json, math, time, subprocess, urllib.request, urllib.parse

API = 'https://open.tiktokapis.com'
KEY, SECRET = os.environ['TIKTOK_CLIENT_KEY'], os.environ['TIKTOK_CLIENT_SECRET']
REDIRECT = os.environ.get('TIKTOK_REDIRECT', 'https://sumisukekun.github.io/music-machine-videos/callback.html')
TOK = 'tokens.enc'


def req(method, url, data=None, headers=None, form=False, raw=None):
    h = dict(headers or {})
    body = None
    if raw is not None: body = raw
    elif form: body = urllib.parse.urlencode(data).encode(); h['Content-Type'] = 'application/x-www-form-urlencoded'
    elif data is not None: body = json.dumps(data).encode(); h['Content-Type'] = 'application/json; charset=UTF-8'
    r = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            t = resp.read(); return json.loads(t) if t else {}
    except urllib.error.HTTPError as e:
        t = e.read().decode(); print('HTTP', e.code, t[:500]); raise


def save(tok):
    p = subprocess.run(['openssl', 'enc', '-aes-256-cbc', '-pbkdf2', '-salt', '-pass', 'env:TOKEN_KEY', '-out', TOK],
                       input=json.dumps(tok).encode(), check=True)


def load():
    out = subprocess.run(['openssl', 'enc', '-d', '-aes-256-cbc', '-pbkdf2', '-pass', 'env:TOKEN_KEY', '-in', TOK], capture_output=True, check=True).stdout
    return json.loads(out)


def token(grant):
    d = dict(client_key=KEY, client_secret=SECRET, **grant)
    t = req('POST', API + '/v2/oauth/token/', d, form=True)
    if 'access_token' not in t: print(t); sys.exit(1)
    return {'access_token': t['access_token'], 'refresh_token': t['refresh_token'], 'open_id': t.get('open_id'), 'scope': t.get('scope'), 'at': int(time.time())}


def connect(code):
    tok = token({'code': code, 'grant_type': 'authorization_code', 'redirect_uri': REDIRECT}); save(tok)
    print('connected, scope:', tok['scope'])


def post(video, caption):
    tok = token({'grant_type': 'refresh_token', 'refresh_token': load()['refresh_token']}); save(tok)
    H = {'Authorization': 'Bearer ' + tok['access_token']}
    ci = req('POST', API + '/v2/post/publish/creator_info/query/', {}, H)['data']
    print('creator:', ci.get('creator_username'), 'privacy options:', ci.get('privacy_level_options'), 'max sec:', ci.get('max_video_post_duration_sec'))
    want = os.environ.get('TIKTOK_PRIVACY', 'SELF_ONLY')
    privacy = want if want in ci.get('privacy_level_options', []) else 'SELF_ONLY'
    size = os.path.getsize(video); chunk = min(size, 10_000_000)
    n = max(1, size // chunk)                     # last chunk takes the remainder (TikTok rule)
    init = req('POST', API + '/v2/post/publish/video/init/', {
        'post_info': {'title': caption[:2200], 'privacy_level': privacy, 'disable_duet': False, 'disable_comment': False,
                      'disable_stitch': False, 'video_cover_timestamp_ms': 3000, 'brand_content_toggle': False, 'brand_organic_toggle': False},
        'source_info': {'source': 'FILE_UPLOAD', 'video_size': size, 'chunk_size': chunk, 'total_chunk_count': n}}, H)['data']
    url, pid = init['upload_url'], init['publish_id']
    with open(video, 'rb') as f:
        for i in range(n):
            start = i * chunk; end = size - 1 if i == n - 1 else start + chunk - 1
            f.seek(start); b = f.read(end - start + 1)
            req('PUT', url, raw=b, headers={'Content-Type': 'video/mp4', 'Content-Range': f'bytes {start}-{end}/{size}', 'Content-Length': str(len(b))})
            print(f'chunk {i + 1}/{n} uploaded')
    for _ in range(60):
        st = req('POST', API + '/v2/post/publish/status/fetch/', {'publish_id': pid}, H)['data']
        print('status:', st.get('status'), st.get('fail_reason', ''))
        if st.get('status') in ('PUBLISH_COMPLETE', 'FAILED'): break
        time.sleep(10)
    if st.get('status') != 'PUBLISH_COMPLETE': sys.exit(1)
    print('published as', privacy, pid)


if __name__ == '__main__':
    if sys.argv[1] == 'connect': connect(sys.argv[2])
    elif sys.argv[1] == 'post': post(sys.argv[2], sys.argv[3])
