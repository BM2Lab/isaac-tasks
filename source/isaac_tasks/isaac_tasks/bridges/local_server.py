"""Private Unix-domain socket endpoint. Single client; disconnect freezes/stops simulation."""
import os
from pathlib import Path
import socket
import stat

from contracts.wire import PROTOCOL_VERSION, read_message, write_message
from contracts.policies import PolicyIOSpec


def serve(session, socket_path, idle_timeout_s=60, max_requests=1000):
    path = Path(socket_path).absolute()
    if len(os.fsencode(path)) > 100:
        raise ValueError("Unix socket path is too long; choose a shorter private directory")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        raise ValueError("Socket parent must be owned by this user with mode 0700")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to replace an existing socket path")
    bound_inode = None
    greeted = False
    seen_ids = set()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.settimeout(idle_timeout_s)
            listener.bind(str(path))
            os.chmod(path, 0o600)
            bound_inode = path.stat().st_ino
            listener.listen(1)
            print(f"V2 simulation ready: {path}", flush=True)
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(idle_timeout_s)
                with connection.makefile("rwb", buffering=0) as stream:
                    for _ in range(max_requests):
                        request = read_message(stream)
                        identity = request.get("id")
                        try:
                            if (set(request) != {"protocol", "id", "op", "payload"}
                                or type(request["protocol"]) is not int or request["protocol"] != PROTOCOL_VERSION
                                or not isinstance(identity, str) or not 1 <= len(identity) <= 64
                                or identity in seen_ids or not isinstance(request["payload"], dict)):
                                raise ValueError("Malformed, duplicated or unsupported request")
                            seen_ids.add(identity)
                            op, payload = request["op"], request["payload"]
                            if op == "hello":
                                if greeted or set(payload) != {"contract"} or PolicyIOSpec.from_dict(payload["contract"]) != session.spec:
                                    raise ValueError("Handshake contract mismatch or duplicate handshake")
                                greeted = True
                                result = {"contract": session.spec.to_dict(), "status": session.status(), "receipt": session.receipt()}
                            elif not greeted:
                                raise ValueError("Matching hello handshake required")
                            elif op == "reset" and set(payload) <= {"seed"}:
                                seed = payload.get("seed")
                                if seed is not None and (type(seed) is not int or seed < 0):
                                    raise ValueError("Reset seed must be a nonnegative integer")
                                result = session.reset(seed)
                            elif op == "observe" and not payload:
                                result = session.observe()
                            elif op == "status" and not payload:
                                result = session.status()
                            elif op == "submit" and set(payload) == {"chunk"}:
                                result = session.submit(payload["chunk"])
                            elif op == "expert_action" and set(payload) == {"context"}:
                                result = session.expert_action(payload["context"])
                            elif op == "advance" and set(payload) == {"context"}:
                                result = session.advance(payload["context"])
                            elif op == "close" and not payload:
                                session.queue.clear()
                                write_message(stream, {"protocol": PROTOCOL_VERSION, "id": identity, "ok": True, "result": {"closed": True}})
                                return
                            else:
                                raise ValueError("Unknown operation or payload fields")
                            response = {"protocol": PROTOCOL_VERSION, "id": identity, "ok": True, "result": result}
                        except (ValueError, TypeError, KeyError) as exc:
                            response = {"protocol": PROTOCOL_VERSION, "id": identity, "ok": False, "error": str(exc)}
                        write_message(stream, response)
                    raise RuntimeError("V2 session request budget exhausted")
    finally:
        session.queue.clear()
        if bound_inode is not None and path.exists() and path.stat().st_ino == bound_inode and stat.S_ISSOCK(path.stat().st_mode):
            path.unlink()
