"""ZMQ server for UR3 block-pushing policy evaluation."""

import json
import logging
import threading
from typing import Any

from tso_robotics_sockets import (
    CompressionType,
    InferenceRequestKey,
    InferenceResponseKey,
    ServerRoute,
    ServerStatus,
    SocketServer,
    TransportKey,
)
from versatil_constants.shared import ActionComponent
from versatil_constants.ur3 import UR3ProprioKey

from versatil_inference.environment import Environment
from versatil_inference.socket_flags import DEFAULT_CLIENT_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_PROPRIO_KEYS = (
    UR3ProprioKey.EE_POS.value,
    UR3ProprioKey.BLOCK1_POS.value,
    UR3ProprioKey.BLOCK2_POS.value,
)


class UR3BlockPushServer(SocketServer):
    """ZMQ-based server for running UR3 block-pushing environments."""

    def __init__(
        self,
        ip_address: str = "0.0.0.0",
        port: int = 5556,
        compression_type: str = CompressionType.RAW.value,
        seed: int = 42,
        num_trials: int = 50,
        output_folder: str = "",
        max_parallel_envs: int = 10,
        record_video: bool = False,
        normalize_io: bool = False,
        stats_path: str | None = None,
    ):
        super().__init__(ip_address=ip_address, port=port)
        self.compression_type = compression_type
        self.environment = Environment(
            seed=seed,
            num_trials=num_trials,
            output_folder=output_folder,
            max_parallel_envs=max_parallel_envs,
            record_video=record_video,
            normalize_io=normalize_io,
            stats_path=stats_path,
        )
        self._register_routes()
        thread = threading.Thread(target=self.environment.initialize, daemon=True)
        thread.start()

    def _register_routes(self) -> None:
        self.add_route(
            ServerRoute.GET_OBSERVATION.value,
            self.handle_request,
            blocking=True,
        )
        self.add_route(
            ServerRoute.SEND_ACTION.value,
            self.handle_request,
            blocking=True,
        )
        self.add_route(
            ServerRoute.REGISTER_CLIENT.value,
            self.handle_request,
            blocking=True,
        )

    def _handle_register_client(self, request_data: dict) -> tuple[bool, dict]:
        client_name = request_data.get(
            InferenceRequestKey.CLIENT_NAME.value,
            DEFAULT_CLIENT_NAME,
        )
        self.environment.client_name = client_name
        logging.info(f"Client connected: {client_name}")
        return True, {TransportKey.STATUS.value: self.environment.current_status}

    def _handle_get_observation(self, request_data: dict) -> tuple[bool, dict]:
        environment = self.environment
        if environment.current_status != ServerStatus.WAITING_ACTION.value:
            return True, {TransportKey.STATUS.value: environment.current_status}
        latest_observation = environment.get_latest_observation()
        if not latest_observation:
            return True, {TransportKey.STATUS.value: environment.current_status}

        requested_keys = request_data.get(InferenceRequestKey.REQUESTED_KEYS.value, [])
        requested_keys_set = set(requested_keys)

        response: dict[str, Any] = {
            TransportKey.STATUS.value: environment.current_status,
            InferenceResponseKey.RESET_ENVIRONMENT_INDICES.value: (
                environment.consume_reset_indices()
            ),
            InferenceResponseKey.TIMESTEP.value: {
                env_idx: latest_observation[env_idx][
                    InferenceResponseKey.TIMESTEP.value
                ]
                for env_idx in latest_observation
            },
        }

        for key in _PROPRIO_KEYS:
            if key in requested_keys_set:
                response[key] = {
                    env_idx: obs[key].tolist()
                    for env_idx, obs in latest_observation.items()
                    if obs.get(key) is not None
                }

        return True, response

    def _handle_send_action(self, request_data: dict) -> tuple[bool, dict]:
        environment = self.environment
        if environment.current_status != ServerStatus.WAITING_ACTION.value:
            return True, {TransportKey.STATUS.value: environment.current_status}
        raw_actions = request_data.get(InferenceRequestKey.ACTIONS.value, {})
        actions = {
            int(key): self._flatten_action(value) for key, value in raw_actions.items()
        }
        environment.step(actions=actions)
        return True, {TransportKey.STATUS.value: environment.current_status}

    @staticmethod
    def _flatten_action(structured_action: dict[str, list[float]]) -> list[float]:
        """Flatten a structured VersatIL action into the UR3 2D target."""
        flat: list[float] = []
        component_names = [
            ActionComponent.POSITION.value,
            ActionComponent.ORIENTATION.value,
            ActionComponent.GRIPPER.value,
        ]
        custom_component = getattr(ActionComponent, "CUSTOM", None)
        if custom_component is not None:
            component_names.append(custom_component.value)
        for component in component_names:
            if component in structured_action:
                flat.extend(structured_action[component])
        return flat

    def handle_request(self, request_data: dict) -> tuple[bool, dict]:
        route_name = request_data.get(TransportKey.ROUTE_NAME.value)
        match route_name:
            case ServerRoute.GET_OBSERVATION.value:
                return self._handle_get_observation(request_data)
            case ServerRoute.SEND_ACTION.value:
                return self._handle_send_action(request_data)
            case ServerRoute.REGISTER_CLIENT.value:
                return self._handle_register_client(request_data)
            case _:
                return False, {
                    TransportKey.ERROR_MSG.value: f"Unknown route: {route_name}",
                }

    def handle_client_request(self) -> dict:
        message = self.reply_socket.recv_string()
        request = json.loads(message)
        success, response = self.handle_request(request)
        if not success:
            response[TransportKey.STATUS.value] = ServerStatus.ERROR.value
        self.reply_socket.send_string(json.dumps(response))
        if response.get(TransportKey.STATUS.value) == ServerStatus.FINISHED.value:
            self.environment.close()
        return response

    def shutdown(self) -> None:
        logging.info("Shutting down UR3BlockPushServer...")
        self.environment.close()
        logging.info("UR3BlockPushServer shut down complete.")
