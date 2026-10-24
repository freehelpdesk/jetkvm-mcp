# jetkvm-mcp

An [MCP](https://modelcontextprotocol.io) server that lets an AI agent operate a computer
through a [JetKVM](https://jetkvm.com). The agent sees the computer's HDMI output and
drives it as a USB keyboard and mouse, so it works everywhere a person at the desk could:
BIOS/UEFI setup, boot menus, OS installers, lock screens, and any operating system. Nothing
needs to be installed on the controlled computer.

It talks to the JetKVM the same way the JetKVM web UI does: WebRTC for video and HID,
JSON-RPC for device functions. No firmware changes are needed.

## What the agent can do

| Tool | Purpose |
|---|---|
| `screenshot` | Capture the screen (optionally after a delay). Defines the coordinate space for every mouse tool. |
| `zoom` | View a screen region at full native resolution to read small text. |
| `get_status` | Video signal, USB attachment, keyboard LEDs, power extension, mounted media. |
| `mouse_move`, `click`, `drag`, `scroll`, `mouse_button` | Absolute pointer control; clicks support double/triple and held modifiers. |
| `mouse_move_relative` | Relative movement for environments that ignore absolute pointers. |
| `type_text` | Type text (US keyboard layout); compensates for Caps Lock. |
| `press_keys` | Shortcuts and sequences: `ctrl+alt+delete`, `cmd+space`, `down down enter`. |
| `hold_keys` | Hold a chord for a duration (e.g. a key during boot). |
| `release_all` | Release every key and button; stop typing in progress. |
| `power` | ATX power/reset buttons, DC power, or USB remote wake. |
| `wake_on_lan` | Send a Wake-on-LAN packet from the JetKVM. |
| `list_virtual_media`, `mount_virtual_media`, `unmount_virtual_media` | Attach an ISO/disk image (URL or stored on the JetKVM) as a USB CD-ROM or drive. |
| `device_rpc` | Raw JetKVM JSON-RPC (disabled unless `JETKVM_ENABLE_RAW_RPC=1`). |

## Requirements

- A JetKVM that has finished onboarding, reachable from this machine. WebRTC media
  flows over UDP, so the machine running this server must be able to reach the JetKVM
  directly, e.g. the same LAN, a VPN, or Tailscale. JetKVM Cloud (app.jetkvm.com) is not supported.
- Python 3.11 or newer.

## Install

```sh
cd jetkvm-mcp
python3 -m venv .venv
.venv/bin/pip install -e .
```

Check the connection. This connects, prints the device state, and saves a screenshot:

```sh
JETKVM_URL=http://192.168.1.50 JETKVM_PASSWORD='…' .venv/bin/jetkvm-mcp check -o shot.png
```

## Connect it to an MCP client

The JetKVM has to be reachable from the machine running this server. WebRTC media
does not go through the MCP client, so a remote Grok or Grok Bot still needs this
process on a network that can reach the device.

**Grok Build**

```sh
grok mcp add jetkvm \
  -e JETKVM_URL=http://192.168.1.50 \
  -e JETKVM_PASSWORD='your-jetkvm-password' \
  -- /absolute/path/to/jetkvm-mcp/.venv/bin/jetkvm-mcp
```

The same entry in `~/.grok/config.toml` (or a project `.grok/config.toml`):

```toml
[mcp_servers.jetkvm]
command = "/absolute/path/to/jetkvm-mcp/.venv/bin/jetkvm-mcp"
env = { JETKVM_URL = "http://192.168.1.50", JETKVM_PASSWORD = "${JETKVM_PASSWORD}" }
```

`${JETKVM_PASSWORD}` is read from the environment. Run `grok mcp doctor jetkvm` to check it.

**Grok Bot**

Grok Bot can run this server as a Command on a computer that can reach the JetKVM
(your desktop, or a Bot computer on that network). In the Bot, add a custom MCP
server of type **Command**:

- Command: `/absolute/path/to/jetkvm-mcp/.venv/bin/jetkvm-mcp`
- Environment: `JETKVM_URL` and, if the device has a password, `JETKVM_PASSWORD`

Put the password in the environment, not in the command arguments. A Command
server that needs environment variables runs only in the owner's own chat. You
can hand the Bot this:

> Add a custom Command MCP server named jetkvm that runs `/absolute/path/to/jetkvm-mcp/.venv/bin/jetkvm-mcp` with `JETKVM_URL` and `JETKVM_PASSWORD` in its environment. It operates the computer attached to my JetKVM. Start with a screenshot.

For a Bot computer that cannot reach the JetKVM, serve HTTP on a machine that
can (see below) and add a custom MCP server of type **Remote HTTPS**. The URL
is `https://<your-host>/mcp` and the credential is `JETKVM_HTTP_TOKEN`.

**Grok (grok.com and the API)**

`serve --http` speaks [Streamable HTTP](https://modelcontextprotocol.io), which
Grok's remote MCP connectors use. The listener requires `JETKVM_HTTP_TOKEN`
(at least 16 letters, digits, or `-._~`):

```sh
export JETKVM_HTTP_TOKEN="$(openssl rand -hex 32)"
JETKVM_URL=http://192.168.1.50 JETKVM_PASSWORD='your-jetkvm-password' \
  .venv/bin/jetkvm-mcp serve --http --host 127.0.0.1 --port 8765
```

Grok's servers need a public HTTPS URL. Point a tunnel at that port and give Grok:

`https://<tunnel-host>/mcp`

On [grok.com/connectors](https://grok.com/connectors), add a custom connector with
that URL. On Grok Business and Enterprise an admin adds it first. If the form
cannot send an `Authorization` header, use `https://<tunnel-host>/mcp/$JETKVM_HTTP_TOKEN`
instead. That path is the token; treat the whole URL as a secret.

From the xAI API, pass the token as `authorization`. Grok copies it into the
`Authorization` header; this server accepts that raw value or a `Bearer` scheme.

```python
import os
from xai_sdk import Client
from xai_sdk.tools import mcp

client = Client(api_key=os.environ["XAI_API_KEY"])
chat = client.chat.create(
    model="grok-4.7",
    tools=[
        mcp(
            server_url="https://<tunnel-host>/mcp",
            server_label="jetkvm",
            authorization=os.environ["JETKVM_HTTP_TOKEN"],
        ),
    ],
)
```

`GET /healthz` answers `ok` without the token. Everything under `/mcp` requires it.
`--host 0.0.0.0` listens beyond localhost; keep the token private.

**Claude Code**

```sh
claude mcp add jetkvm \
  -e JETKVM_URL=http://192.168.1.50 \
  -e JETKVM_PASSWORD='your-jetkvm-password' \
  -- /absolute/path/to/jetkvm-mcp/.venv/bin/jetkvm-mcp
```

**Claude Desktop** (`claude_desktop_config.json`) or other clients using the same format:

```json
{
  "mcpServers": {
    "jetkvm": {
      "command": "/absolute/path/to/jetkvm-mcp/.venv/bin/jetkvm-mcp",
      "env": {
        "JETKVM_URL": "http://192.168.1.50",
        "JETKVM_PASSWORD": "your-jetkvm-password"
      }
    }
  }
}
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `JETKVM_URL` | (required) | Device address, e.g. `http://192.168.1.50` or `https://jetkvm.local`. |
| `JETKVM_PASSWORD` | unset | Local web UI password. Omit if the device uses "no password" mode. |
| `JETKVM_VERIFY_TLS` | `true` | Set `false` for the JetKVM's self-signed HTTPS certificate. |
| `JETKVM_MAX_SCREENSHOT_EDGE` | `1920` | Screenshots are downscaled so their long edge fits this size. A 1080p screen is sent at full resolution. Lower values save tokens. |
| `JETKVM_SCREENSHOT_FORMAT` | `jpeg` | `jpeg` or `png`. The source is already compressed video, so JPEG costs little quality. |
| `JETKVM_JPEG_QUALITY` | `85` | JPEG quality. |
| `JETKVM_TYPING_DELAY_MS` | `20` | Default per-key press and release time for `type_text`. |
| `JETKVM_AUTO_SCREENSHOT` | `false` | Return a screenshot after every action, saving a round trip at the cost of tokens. |
| `JETKVM_ENABLE_RAW_RPC` | `false` | Expose the `device_rpc` tool. |
| `JETKVM_HTTP_TOKEN` | unset | Bearer token for `serve --http`. Required there. At least 16 letters, digits, or `-._~`. |
| `JETKVM_ICE_SERVERS` | unset | Comma-separated STUN/TURN URLs, only needed across NAT. |
| `JETKVM_CONNECT_TIMEOUT` | `20` | Seconds to wait for the connection. |
| `JETKVM_LOG_LEVEL` | `WARNING` | Log level (logs go to stderr). |

## Things to know

- **One session at a time.** A JetKVM serves one WebRTC session. When the agent connects,
  an open JetKVM browser tab is disconnected, and opening the web UI later disconnects the
  agent. The agent reconnects on its next action and tells you that it took the session back.
- **Logging in signs out browsers.** The JetKVM keeps a single auth token, so the server's
  login invalidates other browsers' logins (they will ask for the password on reload). The
  server logs in only when it has no valid session cookie.
- **Keyboard layout.** `type_text` sends key positions for a US layout. If the computer uses
  another layout, characters such as `@`, `"` or `z`/`y` may come out differently.
- **Multiple monitors.** The JetKVM sees one display. With several monitors attached, the
  absolute pointer may map across all of them and clicks will land in the wrong place.
- **Video runs while connected.** Like an open web UI tab, the server receives and decodes
  the video stream for as long as it is connected, which costs some CPU and bandwidth.
- **Input requires USB to be attached.** When the computer is off or suspended its USB port
  may not be active; `screenshot` and `get_status` report this.
- **Security.** Anyone who can use this server has full control of the computer, at the
  level of someone physically at its keyboard. Treat its configuration (including
  `JETKVM_PASSWORD` and `JETKVM_HTTP_TOKEN`) accordingly, and keep a human in the loop
  for destructive actions. The HTTP listener admits anyone who has the token.

## Development

```sh
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

The tests include an end-to-end suite that runs the real client and MCP server against
`tests/fake_device.py`, a simulated JetKVM built from the firmware's protocol: it does
the WebSocket signaling with trickled ICE, the `rpc` and `hidrpc` data channels, and an H.264
video track. It cannot catch every difference from real hardware, so run `jetkvm-mcp check`
against a real device after changing the transport.

Layout:

- `src/jetkvm_mcp/client.py`: auth, signaling, WebRTC session, JSON-RPC, HID transport, video frames
- `src/jetkvm_mcp/hidrpc.py`: binary HID protocol codec
- `src/jetkvm_mcp/keyboard.py`: HID usage tables, key-name parsing, text-to-keystroke conversion
- `src/jetkvm_mcp/display.py`: screenshot scaling and pixel-to-HID coordinate mapping
- `src/jetkvm_mcp/server.py`: MCP tools and CLI
