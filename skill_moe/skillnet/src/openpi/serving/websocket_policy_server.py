import asyncio
import logging
import http
import traceback
import time
import numpy as np

from skillnet_client import base_policy as _base_policy
from skillnet_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


logger = logging.getLogger(__name__)

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request = _health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                # ======================================
                start_time = time.monotonic()
                # ===========================================

                obs = msgpack_numpy.unpackb(await websocket.recv())
                
                # ================================================
                infer_time =time.monotonic()
                # =========================================

                return_action = self._policy.infer(obs)
                
                # ============================================
                infer_time = time.monotonic() - infer_time
                # =================================================
                await websocket.send(packer.pack(return_action))

                if isinstance(return_action, tuple):
                    action, attn_traj, router_traj, skill_embeds = return_action
                    return_action = {
                    "actions": np.asarray(action['actions'], dtype=np.float32),
                    "attn": np.asarray(attn_traj, dtype=np.float32) if attn_traj is not None else None,
                    "router": np.asarray(router_traj, dtype=np.float32) if router_traj is not None else None,
                    "skills": np.asarray(skill_embeds, dtype=np.float32) if skill_embeds is not None else None,
                }
                else:
                    action = return_action
                    attn_traj = None
                    skill_embeds = None
                
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }

                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                await websocket.send(packer.pack(return_action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

def _health_check(connection: websockets.asyncio.server.ServerConnection, request: websockets.asyncio.server.Request) -> websockets.asyncio.server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
