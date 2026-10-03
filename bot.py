"""Discord music bot. Needs Python 3.10+, FFmpeg, and `pip install -r requirements.txt`."""
import asyncio
import colorsys
import os
import time

import discord
import spotipy
import yt_dlp
from discord import app_commands
from discord.ext import commands
from spotipy.oauth2 import SpotifyClientCredentials
from ytmusicapi import YTMusic

TOKEN = os.getenv("DISCORD_TOKEN")
# Custom YT Music emoji, like <:ytmusic:1234567890> (falls back to 🎵)
YTM_EMOJI = os.getenv("YTM_EMOJI", "🎵")
ADDED_COLOR = 0xFF0033
RAINBOW_STEP = 1 / 12   # fallback: hue jump per change
RAINBOW_DELAY = 1.2     # fallback: seconds between color changes
BEAT_GAP = 1.0          # min seconds between beat color changes (Discord edit limit)
BEAT_LEAD = 0.35        # change color slightly early to cover edit lag
MAX_ANALYZE_SECS = 720  # only analyze the first 12 min of a song

YDL_OPTS = {"format": "bestaudio/best", "noplaylist": True, "quiet": True}
FFMPEG_OPTS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}

ytm = YTMusic()
sp = None
if os.getenv("SPOTIPY_CLIENT_ID") and os.getenv("SPOTIPY_CLIENT_SECRET"):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())


class MusicBot(commands.Bot):
    async def setup_hook(self):
        await self.tree.sync()


bot = MusicBot(command_prefix="!", intents=discord.Intents.default())

queues: dict[int, list[dict]] = {}
panels: dict[int, "Panel"] = {}
volumes: dict[int, float] = {}          # guild id -> 0.0 - 1.0
locks: dict[int, asyncio.Lock] = {}     # one lock per server
starting: set[int] = set()              # servers that are about to start a song
songs: dict[str, dict] = {}             # videoId -> song info (from autocomplete)
search_cache: dict[tuple, tuple] = {}   # (query, limit) -> (time, results)
analyze_lock = asyncio.Semaphore(1)     # one beat analysis at a time (CPU heavy)


def get_lock(gid: int) -> asyncio.Lock:
    return locks.setdefault(gid, asyncio.Lock())


async def run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


# ---------- search ----------
def search_songs(query: str, limit: int = 8) -> list[dict]:
    out = []
    for r in ytm.search(query, filter="songs", limit=limit):
        if not r.get("videoId"):
            continue
        artists = ", ".join(a["name"] for a in r.get("artists", []))
        thumbs = r.get("thumbnails") or []
        song = {
            "label": f"{r['title']} - {artists}",
            "vid": r["videoId"],
            "secs": r.get("duration_seconds") or 0,
            "thumb": thumbs[-1]["url"] if thumbs else None,
        }
        out.append(song)
        if len(songs) > 1000:
            songs.pop(next(iter(songs)))
        songs[song["vid"]] = dict(song)
    return out


async def cached_search(query: str, limit: int = 8) -> list[dict]:
    key = (query.lower().strip(), limit)
    hit = search_cache.get(key)
    if hit and time.monotonic() - hit[0] < 120:
        return [dict(r) for r in hit[1]]
    results = await run(search_songs, query, limit)
    if len(search_cache) > 200:
        search_cache.clear()
    search_cache[key] = (time.monotonic(), results)
    return [dict(r) for r in results]


def spotify_labels(url: str) -> list[str]:
    if "/track/" in url:
        tracks = [sp.track(url)]
    elif "/album/" in url:
        tracks = sp.album_tracks(url)["items"][:25]
    elif "/playlist/" in url:
        items = sp.playlist_items(url, limit=25)["items"]
        tracks = [i["track"] for i in items if i.get("track")]
    else:
        return []
    return [f"{t['name']} - {t['artists'][0]['name']}" for t in tracks]


# ---------- "Added to queue" card ----------
def added_card(items: list[dict]) -> discord.Embed:
    if len(items) == 1:
        it = items[0]
        title = f"**{it['label']}**"
        if it.get("vid"):
            title = f"[{title}](https://music.youtube.com/watch?v={it['vid']})"
        dur = f" - `{fmt(it['secs'])}`" if it.get("secs") else ""
        text = f"{YTM_EMOJI} Added {title}{dur} to the queue."
    else:
        text = f"{YTM_EMOJI} Added **{len(items)} songs** to the queue."
    return discord.Embed(description=text, color=ADDED_COLOR)


async def delete_card(item: dict | None):
    if item and item.get("card"):
        channel, msg_id = item["card"]
        try:
            await channel.get_partial_message(msg_id).delete()
        except Exception:
            pass
        item["card"] = None


async def clear_queue(gid: int):
    for it in queues.get(gid, []):
        await delete_card(it)
    queues[gid] = []


# ---------- beat detection ----------
def pick_strong(times, strengths, gap: float = BEAT_GAP) -> list[tuple[float, float]]:
    """Keep the strongest hits, at least `gap` seconds apart. Returns [(time, 0-1)]."""
    import numpy as np

    if len(times) == 0:
        return []
    peak = float(np.max(strengths))
    if peak <= 0:
        return []
    chosen: list[tuple[float, float]] = []
    for i in np.argsort(-np.asarray(strengths)):  # strongest first
        t = float(times[i])
        if all(abs(t - c[0]) >= gap for c in chosen):
            chosen.append((t, float(strengths[i]) / peak))
    chosen.sort()
    return chosen


def beats_from_pcm(data: bytes):
    import librosa
    import numpy as np

    sr = 22050
    y = np.frombuffer(data, dtype=np.float32)
    if len(y) < sr * 5:
        return None
    env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
    if len(frames) == 0:
        return None
    times = librosa.frames_to_time(frames, sr=sr)
    return pick_strong(times, env[frames]) or None


async def decode_audio(stream: str) -> bytes:
    """Decode the song to mono float PCM with ffmpeg (killed if cancelled)."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", stream, "-t", str(MAX_ANALYZE_SECS),
        "-f", "f32le", "-ac", "1", "-ar", "22050", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        data = await asyncio.wait_for(proc.stdout.read(), timeout=180)
        await proc.wait()
        return data
    except BaseException:
        proc.kill()
        raise


async def analyze(panel: "Panel", stream: str):
    beats = None
    try:
        async with analyze_lock:
            data = await decode_audio(stream)
            beats = await run(beats_from_pcm, data)
    except Exception as e:  # not caught: CancelledError
        print(f"Beat analysis failed: {e!r}")
    panel.beats = beats
    panel.sync = "on" if beats else "off"


# ---------- "Now playing" panel (rainbow + buttons) ----------
class Panel(discord.ui.View):
    def __init__(self, guild: discord.Guild, track: dict):
        super().__init__(timeout=None)
        self.guild = guild
        self.track = track
        self.msg: discord.Message | None = None
        self.task: asyncio.Task | None = None
        self.analysis: asyncio.Task | None = None
        self.t0 = time.monotonic()
        self.paused_at: float | None = None
        self.paused_total = 0.0
        self.hue = 0.0
        self.beats: list[tuple[float, float]] | None = None
        self.sync = "pending"  # pending / on / off

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        vc = self.guild.voice_client
        voice = getattr(interaction.user, "voice", None)
        if vc and voice and voice.channel == vc.channel:
            return True
        await interaction.response.send_message(
            "Join my voice channel to use these buttons.", ephemeral=True
        )
        return False

    def elapsed(self) -> float:
        now = self.paused_at or time.monotonic()
        e = max(0.0, now - self.t0 - self.paused_total)
        secs = self.track.get("secs") or 0
        return min(e, secs) if secs else e

    def embed(self) -> discord.Embed:
        r, g, b = colorsys.hsv_to_rgb(self.hue, 0.85, 1.0)
        color = discord.Colour.from_rgb(int(r * 255), int(g * 255), int(b * 255))

        it = self.track
        title = f"**{it['label']}**"
        if it.get("vid"):
            title = f"[{title}](https://music.youtube.com/watch?v={it['vid']})"

        desc = title
        if it.get("secs"):
            filled = min(14, int(self.elapsed() / it["secs"] * 14))
            bar = "▰" * filled + "▱" * (14 - filled)
            state = "⏸️" if self.paused_at else "▶️"
            desc += f"\n\n{state} `{fmt(self.elapsed())}` {bar} `{fmt(it['secs'])}`"

        vol = volumes.get(self.guild.id, 0.5)
        n = round(vol * 10)
        vbar = "▰" * n + "▱" * (10 - n)
        queue = queues.get(self.guild.id, [])
        up_next = queue[0]["label"] if queue else "Nothing yet"

        e = discord.Embed(title=f"{YTM_EMOJI} Now playing", description=desc, color=color)
        e.add_field(name="🔊 Volume", value=f"`{vbar}` {int(vol * 100)}%", inline=False)
        e.add_field(name="⏭️ Up next", value=up_next[:100], inline=True)
        e.add_field(name="📜 In queue", value=str(len(queue)), inline=True)
        if it.get("thumb"):
            e.set_thumbnail(url=it["thumb"])
        sync = {"pending": "syncing beats…", "on": "beat sync on", "off": "rainbow mode"}
        e.set_footer(text=f"Requested by {it.get('by', '?')} • {sync[self.sync]}")
        return e

    @discord.ui.button(emoji="⏯️", style=discord.ButtonStyle.secondary)
    async def pause_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        vc = self.guild.voice_client
        if vc and vc.is_paused():
            vc.resume()
            if self.paused_at:
                self.paused_total += time.monotonic() - self.paused_at
            self.paused_at = None
        elif vc and vc.is_playing():
            vc.pause()
            self.paused_at = time.monotonic()
        await interaction.response.edit_message(embed=self.embed())

    def _set_volume(self, delta: float):
        vol = min(1.0, max(0.0, volumes.get(self.guild.id, 0.5) + delta))
        volumes[self.guild.id] = vol
        vc = self.guild.voice_client
        if vc and isinstance(vc.source, discord.PCMVolumeTransformer):
            vc.source.volume = vol

    @discord.ui.button(emoji="🔉", style=discord.ButtonStyle.secondary)
    async def vol_down(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._set_volume(-0.1)
        await interaction.response.edit_message(embed=self.embed())

    @discord.ui.button(emoji="🔊", style=discord.ButtonStyle.secondary)
    async def vol_up(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._set_volume(+0.1)
        await interaction.response.edit_message(embed=self.embed())


async def rainbow(panel: Panel):
    """Changes the side color on strong beats once analysis is done.
    Until then (or if analysis fails) it cycles the rainbow."""
    last_edit = 0.0
    idx = 0
    while True:
        await asyncio.sleep(0.2)
        now = time.monotonic()
        if now - last_edit < BEAT_GAP:
            continue
        change = False
        if panel.beats is None:
            if now - last_edit >= RAINBOW_DELAY:
                panel.hue = (panel.hue + RAINBOW_STEP) % 1
                change = True
        elif not panel.paused_at:
            el = panel.elapsed()
            while idx < len(panel.beats) and panel.beats[idx][0] < el - 0.3:
                idx += 1  # skip beats we already missed
            if idx < len(panel.beats) and panel.beats[idx][0] <= el + BEAT_LEAD:
                panel.hue = (panel.hue + 0.06 + 0.25 * panel.beats[idx][1]) % 1
                idx += 1
                change = True
        if not change and now - last_edit < 3:
            continue  # still refresh the progress bar every 3s
        last_edit = now  # set even on failure so we never spam retries
        try:
            await panel.msg.edit(embed=panel.embed())
        except (discord.NotFound, discord.Forbidden):
            return
        except discord.HTTPException:
            pass


async def end_panel(gid: int):
    panel = panels.pop(gid, None)
    if not panel:
        return
    for t in (panel.task, panel.analysis):
        if t:
            t.cancel()
    panel.stop()
    if panel.msg:
        try:
            await panel.msg.delete()
        except Exception:
            pass


# ---------- playback ----------
def get_stream(target: str) -> str:
    with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
        info = ydl.extract_info(target, download=False)
        if info and "entries" in info:
            info = info["entries"][0]
        if not info or not info.get("url"):
            raise RuntimeError("no playable audio found")
        return info["url"]


async def resolve(item: dict) -> str:
    if not item.get("vid"):  # lazy item (Spotify batch)
        found = await cached_search(item["label"], 1)
        if not found:
            return await run(get_stream, f"ytsearch1:{item['label']}")
        item.update(found[0])
    return await run(get_stream, f"https://music.youtube.com/watch?v={item['vid']}")


async def play_next(guild: discord.Guild):
    gid = guild.id
    async with get_lock(gid):
        vc = guild.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            return  # something is already playing, don't start a second one
        await end_panel(gid)  # the finished song's panel disappears

        queue = queues.get(gid, [])
        while vc and queue:
            item = queue.pop(0)
            try:
                stream = await resolve(item)
            except Exception as e:
                print(f"Skipping {item['label']}: {e!r}")
                await delete_card(item)
                try:
                    await item["channel"].send(
                        f"Couldn't play **{item['label']}**, skipping it.", delete_after=10
                    )
                except Exception:
                    pass
                continue
            await delete_card(item)  # "Added" card is replaced by the panel

            def after(error, guild=guild):
                if error:
                    print(f"Player error: {error}")
                asyncio.run_coroutine_threadsafe(play_next(guild), bot.loop)

            source = discord.PCMVolumeTransformer(
                discord.FFmpegPCMAudio(stream, **FFMPEG_OPTS),
                volume=volumes.get(gid, 0.5),
            )
            vc.play(source, after=after)

            panel = Panel(guild, item)
            panels[gid] = panel
            try:
                panel.msg = await item["channel"].send(embed=panel.embed(), view=panel)
            except discord.HTTPException as e:
                print(f"Couldn't post the panel (check my permissions): {e!r}")
            else:
                panel.task = asyncio.create_task(rainbow(panel))
            panel.analysis = asyncio.create_task(analyze(panel, stream))
            return


# ---------- events & commands ----------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.event
async def on_voice_state_update(member, before, after):
    # The bot got disconnected/kicked: drop the queue and panel
    if member.id == bot.user.id and before.channel and not after.channel:
        await clear_queue(member.guild.id)
        await end_panel(member.guild.id)


async def song_autocomplete(interaction: discord.Interaction, typed: str):
    if len(typed.strip()) < 2:
        return []
    try:  # Discord gives autocomplete only 3 seconds
        results = await asyncio.wait_for(cached_search(typed), timeout=2.5)
    except Exception:
        return []
    return [
        app_commands.Choice(name=r["label"][:100], value=f"yt:{r['vid']}")
        for r in results
    ]


@bot.tree.command(name="play", description="Search for a song and play it")
@app_commands.describe(song="Type a song or artist, then pick from the list")
@app_commands.autocomplete(song=song_autocomplete)
@app_commands.guild_only()
async def play(interaction: discord.Interaction, song: str):
    if not interaction.user.voice:
        return await interaction.response.send_message(
            "Join a voice channel first.", ephemeral=True
        )
    await interaction.response.defer()

    items, text = [], None
    if song.startswith("yt:"):  # picked from the list
        vid = song[3:]
        if vid in songs:
            items = [dict(songs[vid])]
        else:
            try:
                d = (await run(ytm.get_song, vid))["videoDetails"]
                thumbs = d.get("thumbnail", {}).get("thumbnails") or []
                items = [{
                    "vid": vid,
                    "label": f"{d['title']} - {d['author']}",
                    "secs": int(d["lengthSeconds"]),
                    "thumb": thumbs[-1]["url"] if thumbs else None,
                }]
            except Exception:
                return await interaction.followup.send("Couldn't load that song.")
    elif "open.spotify.com" in song:
        if not sp:
            return await interaction.followup.send("Spotify isn't set up on this bot.")
        try:
            labels = await run(spotify_labels, song)
        except Exception as e:
            return await interaction.followup.send(f"Spotify error: `{e}`")
        if len(labels) == 1:
            text = labels[0]
        else:
            items = [{"label": l} for l in labels]
    elif song.startswith("http"):
        return await interaction.followup.send("Just type the song name and pick from the list.")
    else:  # typed and hit enter without picking
        text = song

    if text:
        try:
            items = (await cached_search(text, 1))[:1]
        except Exception:
            items = []
    if not items:
        return await interaction.followup.send("Couldn't find that song.")

    for it in items:
        it["by"] = interaction.user.display_name
        it["channel"] = interaction.channel

    guild = interaction.guild
    gid = guild.id
    channel = interaction.user.voice.channel

    async with get_lock(gid):
        vc = guild.voice_client
        busy = bool(vc and (vc.is_playing() or vc.is_paused())) or gid in starting
        if vc and vc.channel != channel and busy:
            # don't let someone drag the bot away mid-song
            return await interaction.followup.send(
                f"I'm already playing in {vc.channel.mention}."
            )
        try:
            if not vc:
                vc = await channel.connect()
            elif vc.channel != channel:
                await vc.move_to(channel)
        except Exception as e:
            print(f"Voice connect failed: {e!r}")
            return await interaction.followup.send(
                "Couldn't join that voice channel. Check my Connect/Speak permissions."
            )

        queued = busy
        if queued:  # something is playing: just add to the queue
            msg = await interaction.followup.send(embed=added_card(items))
            items[0]["card"] = (interaction.channel, msg.id)
        else:
            starting.add(gid)
        queues.setdefault(gid, []).extend(items)

    if not queued:
        try:
            await play_next(guild)
        finally:
            starting.discard(gid)
        if gid in panels:
            try:
                await interaction.delete_original_response()
            except Exception:
                pass
        else:
            await interaction.followup.send("Couldn't play that.")


# Mods only (Manage Server). Regular members can't skip.
@bot.tree.command(name="skip", description="Skip the current song (mods only)")
@app_commands.default_permissions(manage_guild=True)
@app_commands.guild_only()
async def skip(interaction: discord.Interaction):
    vc = interaction.guild.voice_client
    if vc and (vc.is_playing() or vc.is_paused()):
        vc.stop()
        await interaction.response.send_message("Skipped.", ephemeral=True)
    else:
        await interaction.response.send_message("Nothing playing.", ephemeral=True)


@bot.tree.command(name="queue", description="Show the queue")
@app_commands.guild_only()
async def show_queue(interaction: discord.Interaction):
    gid = interaction.guild.id
    lines = []
    if gid in panels:
        lines.append(f"**Now:** {panels[gid].track['label']}")
    for i, t in enumerate(queues.get(gid, [])[:10], 1):
        lines.append(f"{i}. {t['label']}")
    await interaction.response.send_message("\n".join(lines) or "Queue is empty.")


@bot.tree.command(name="stop", description="Stop and clear the queue (mods only)")
@app_commands.default_permissions(manage_guild=True)
@app_commands.guild_only()
async def stop(interaction: discord.Interaction):
    await clear_queue(interaction.guild.id)
    vc = interaction.guild.voice_client
    if vc:
        vc.stop()
    await interaction.response.send_message("Stopped.", ephemeral=True)


@bot.tree.command(name="leave", description="Disconnect the bot (mods only)")
@app_commands.default_permissions(manage_guild=True)
@app_commands.guild_only()
async def leave(interaction: discord.Interaction):
    await clear_queue(interaction.guild.id)
    await end_panel(interaction.guild.id)
    vc = interaction.guild.voice_client
    if vc:
        await vc.disconnect()
        await interaction.response.send_message("Bye 👋", ephemeral=True)
    else:
        await interaction.response.send_message("Not in a voice channel.", ephemeral=True)


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Set the DISCORD_TOKEN environment variable first.")
    bot.run(TOKEN)
